from torch.cuda import amp

from .base import *
from .base.container import _set_model_state


# <editor-fold desc='训练器原型'>

class Trainer(LossManager, OptimazerManager, ModelInfoManager, IterBasedTemplate, VariableManager):

    def __init__(self,
                 loader: IVirtualDataLoader,
                 total_epoch: Optional[int] = None,
                 total_iter: Optional[int] = None,
                 accu_step: Optional[int] = None,
                 enable_half: bool = False,
                 kwargs_train: Optional[Dict] = None):
        IterBasedTemplate.__init__(
            self, loader=loader, total_epoch=total_epoch, total_iter=total_iter,
            device=DEVICE, processor=None)
        LossManager.__init__(self)
        ModelInfoManager.__init__(self)
        OptimazerManager.__init__(self)
        VariableManager.__init__(self)

        self.kwargs_train = {} if kwargs_train is None else kwargs_train
        self.scaler = amp.GradScaler(enabled=enable_half)
        self.model = None
        self.model_ema = None
        self._accu_step = accu_step
        dist_gain = dist.get_world_size() if dist.is_initialized() else 1.0  # 自动平衡ddp
        self._learn_gain = dist_gain * self.batch_size

    # <editor-fold desc='模型管理'>

    def set_state(self, model):
        return _set_model_state(model, train_mode=True, enable_half=self.scaler.is_enabled())

    # </editor-fold>

    # <editor-fold desc='参数管理'>
    @property
    def img_size(self) -> Tuple[int, int]:
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size: Tuple[int, int]) -> NoReturn:
        if self.model is not None:
            self.model.img_size = img_size
        self.loader.img_size = img_size

    @property
    def accu_step(self) -> int:
        return self._accu_step

    @accu_step.setter
    def accu_step(self, accu_step: int) -> NoReturn:
        self._accu_step = accu_step

    @property
    def learn_gain(self) -> float:
        return self._learn_gain

    @property
    def enable_half(self) -> bool:
        return self.scaler.is_enabled()

    # </editor-fold>

    # <editor-fold desc='optimizer处理'>
    def build_optimizers(self, model):
        for name, pkd_module in model.pkd_modules.items():
            for actor in self.get_actors(OptimizerBuildActor):
                if actor.module_name is None or actor.module_name == name:
                    self.optimizers[name] = actor.act_build_optimizer(pkd_module)
        return self

    def optimizer_step(self, module_name: Optional[str] = None):
        if self.accu_step is not None and self.ind_iter % self.accu_step != 0:
            return self
        for name_opt, optimizer in self.optimizers.items():
            if module_name is not None and not module_name == name_opt:
                continue
            self.scaler.unscale_(optimizer)
        for actor in self.get_actors(BeforeOptimizeActor):
            actor.act_before_optimize(self, module_name=module_name)
        for name_opt, optimizer in self.optimizers.items():
            if module_name is not None and not module_name == name_opt:
                continue
            self.scaler.step(optimizer)  # optimizer.step
            self.scaler.update()
        for actor in self.get_actors(AfterOptimizeActor):
            actor.act_after_optimize(self, module_name=module_name)
        return self

    def optimizer_zero_grad(self, module_name: Optional[str] = None):
        if self.accu_step is not None and self.ind_iter % self.accu_step != 0:
            return self
        for name_opt, optimizer in self.optimizers.items():
            if module_name is not None and not module_name == name_opt:
                continue
            optimizer.zero_grad()
        return self

    def update_backward_loss(self, loss: Union[dict, torch.Tensor], name: str = 'Loss'):
        loss_sum, names, losses = LossManager.process_loss(loss)
        self.update_loss(name, loss_sum)
        self.update_losses(names, losses)
        if self.learn_gain is not None:
            loss_sum = loss_sum * self.learn_gain
        self.scaler.scale(loss_sum).backward()
        return self

    # </editor-fold>

    # <editor-fold desc='权重保存管理'>

    def _pth_kwargs(self):
        return dict(ind_iter=self.ind_iter + 1, ind_epoch=self.ind_epoch + 1)

    def save_pth_state(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE, default_name='state',
                       extend=EXTENDS.DCT, **kwargs):
        kwargs.update(self._pth_kwargs())
        save_pth_state = format_save_pth(
            save_pth=save_pth, formatter=formatter, extend=extend, default_name=default_name, **kwargs)
        return save_pth_state

    def save_state(self, save_pth: str, **kwargs):
        save_pth_state = self.save_pth_state(save_pth=save_pth, **kwargs)
        state = {
            SVKEYS.IND_EPOCH: self.ind_epoch,
            SVKEYS.IND_ITER: self.ind_iter,
            SVKEYS.IND_ITER_INEP: self.ind_iter_inep,
            SVKEYS.RUNNING: self.running,
        }
        save_json(save_pth_state, state)

    def load_state(self, save_pth: str, **kwargs):
        save_pth_state = self.save_pth_state(save_pth=save_pth, **kwargs)
        if os.path.exists(save_pth_state):
            state = load_json(save_pth_state)
            self._ind_epoch = state[SVKEYS.IND_EPOCH]
            if state[SVKEYS.IND_ITER_INEP] == self.total_iter_inep:
                self._ind_epoch = self._ind_epoch + 1
            self._ind_iter = state[SVKEYS.IND_ITER]
            self._running = state[SVKEYS.RUNNING]
        return self

    def save_pth_model(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE, default_name='model',
                       extend=EXTENDS.MODEL_WEIGHT, **kwargs):
        kwargs.update(self._pth_kwargs())
        save_pth_model = format_save_pth(
            save_pth=save_pth, formatter=formatter, extend=extend, default_name=default_name, **kwargs)
        return save_pth_model

    def save_model(self, save_pth: str, **kwargs):
        save_pth_model = self.save_pth_model(save_pth=save_pth, **kwargs)
        if self.model is not None:
            self.model.save(save_pth_model)
        return self

    def load_model(self, save_pth: str, **kwargs):
        save_pth_model = self.save_pth_model(save_pth=save_pth, **kwargs)
        if self.model is not None and os.path.exists(save_pth_model):
            self.model.load(save_pth_model, broadcast=self.broadcast)
        return self

    def save_pth_model_ema(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE, default_name='model',
                           extend=EXTENDS.MODEL_WEIGHT, appendix='_ema', **kwargs):
        kwargs.update(self._pth_kwargs())
        save_pth_model = format_save_pth(
            appendix=appendix, save_pth=save_pth, formatter=formatter, extend=extend,
            default_name=default_name, **kwargs)
        return save_pth_model

    def save_model_ema(self, save_pth: str, **kwargs):
        save_pth_model_ema = self.save_pth_model_ema(save_pth=save_pth, **kwargs)
        if self.model_ema is not None:
            self.model_ema.save(save_pth_model_ema)
        return self

    def load_model_ema(self, save_pth: str, **kwargs):
        save_pth_model_ema = self.save_pth_model_ema(save_pth=save_pth, **kwargs)
        if self.model_ema is not None and os.path.exists(save_pth_model_ema):
            self.model_ema.load(save_pth_model_ema)
        return self

    def save_pth_optim(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE, default_name='optimizer',
                       extend=EXTENDS.OPTIMIZER_WEIGHT, **kwargs):
        kwargs.update(self._pth_kwargs())
        save_pth_optm = format_save_pth(
            save_pth=save_pth, formatter=formatter, extend=extend,
            default_name=default_name, **kwargs)
        return save_pth_optm

    def save_optimizer(self, save_pth: str, **kwargs):
        save_pth_optm = self.save_pth_optim(save_pth=save_pth, **kwargs)
        opt_dct = {}
        for name, optimizer in self.optimizers.items():
            opt_dct[name] = optimizer.state_dict()
        torch.save(opt_dct, save_pth_optm)
        return self

    def load_optimizer(self, save_pth: str, **kwargs):
        save_pth_optm = self.save_pth_optim(save_pth=save_pth, **kwargs)
        if os.path.exists(save_pth_optm):
            opt_dct = torch.load(save_pth_optm)
            for name, optimizer in self.optimizers.items():
                optimizer.load_state_dict(opt_dct[name])
        return self

    def save_pth_dct(self, save_pth: str, **kwargs):
        return {
            SVKEYS.SAVE_PTH_OPTIM: self.save_pth_optim(save_pth=save_pth, **kwargs),
            SVKEYS.SAVE_PTH_MODEL: self.save_pth_model(save_pth=save_pth, **kwargs),
            SVKEYS.SAVE_PTH_MODEL_EMA: self.save_pth_model_ema(save_pth=save_pth, **kwargs),
        }

    def save(self, save_pth: str, **kwargs):
        self.save_model(save_pth, **kwargs)
        self.save_model_ema(save_pth, **kwargs)
        self.save_optimizer(save_pth, **kwargs)
        self.save_state(save_pth)
        for actor in self.get_actors(AfterSaveActor):
            actor.act_after_save(self, save_pth)
        return self

    def load(self, save_pth: str, **kwargs):
        self.load_model(save_pth, **kwargs)
        self.load_model_ema(save_pth, **kwargs)
        self.load_optimizer(save_pth, **kwargs)
        self.load_state(save_pth)
        for actor in self.get_actors(AfterLoadActor):
            actor.act_after_load(self, save_pth)
        return self

    # </editor-fold>

    # <editor-fold desc='训练执行'>

    def act_init(self, model, *args, **kwargs):
        self.model = model
        self.set_state(model)
        self.device = model.device
        self.running = True
        self.build_optimizers(model)
        self.processor = model.labels2tars
        self.add_actor(model)
        self.model.act_init_train(self)
        return None

    def act_return(self):
        return None

    def act_ending(self):
        return None

    def act_iter(self):
        imgs, targets = self.batch_data
        self.model.act_iter_train(self, imgs, targets, **self.kwargs_train)
    # </editor-fold>


# </editor-fold>

# <editor-fold desc='梯度管理'>
class GradClipActor(BeforeOptimizeActor):
    def __init__(self, grad_norm: float = 10, module_name: Optional[str] = None, ):
        self.grad_norm = grad_norm
        self.module_name = module_name

    @property
    def lev_before_optimize(self):
        return 1

    def act_before_optimize(self, container, module_name: Optional[str] = None, **kwargs):
        for name_opt, optimizer in container.optimizers.items():
            if module_name is not None and not module_name == name_opt:
                continue
            if self.module_name is not None and not self.module_name == name_opt:
                continue
            if self.grad_norm > 0:
                for para_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(para_group['params'], max_norm=self.grad_norm)


class CheckGradActor(BeforeOptimizeActor):
    def __init__(self, norm_thres: float = 10, module_name: Optional[str] = None, ):
        self.norm_thres = norm_thres
        self.module_name = module_name

    @property
    def lev_before_optimize(self):
        return 3

    def act_before_optimize(self, container, module_name: Optional[str] = None, **kwargs):
        for name, para in container.model.named_parameters():
            norm = torch.linalg.norm(para.grad)
            if norm > self.norm_thres or torch.isnan(norm).item():
                container.broadcast(name + '\t' + '%8.4f' % norm.item())


class CheckParaActor(AfterOptimizeActor):
    def __init__(self, norm_thres: float = 10, module_name: Optional[str] = None, ):
        self.norm_thres = norm_thres
        self.module_name = module_name

    @property
    def lev_after_optimize(self):
        return 1

    def act_after_optimize(self, container, module_name: Optional[str] = None, **kwargs):
        for name, para in container.model.named_parameters():
            norm = torch.linalg.norm(para)
            if norm > self.norm_thres or torch.isnan(norm).item():
                container.broadcast(name + '\t' + '%8.4f' % norm.item())


class IterBasedGradCollectActor(BeforeOptimizeActor, IntervalTrigger):
    def __init__(self, step: int, offset: int = 0, first: bool = False, last: bool = False,
                 para_names: Optional[Sequence[str]] = None, ignore_nograd: bool = True):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.para_names = para_names
        self.ignore_nograd = ignore_nograd

    @property
    def lev_before_optimize(self):
        return 3

    def act_before_optimize(self, trainer, module_name: Optional[str] = None, **kwargs):
        if trainer.main_proc and self.trigger(ind=trainer.ind_iter, total=trainer.total_iter):
            names = []
            size_strs = []
            norm_strs = []
            intv_strs = []
            dist_strs = []
            for name, para in trainer.model.named_parameters():
                if self.para_names is not None and name not in self.para_names:
                    continue
                grad = para.grad
                if grad is None and self.ignore_nograd:
                    continue
                elif grad is None:
                    size_str = 'None'
                    dist_str = ''
                    intv_str = ''
                    norm_str = ''
                else:
                    size_str = 'Size ' + str(list(grad.size()))
                    dist_str = 'Dist %8.4f' % (torch.mean(grad).item()) + ' ~ %-8.4f' % (torch.std(grad).item())
                    intv_str = 'Intv %8.4f' % (torch.min(grad).item()) + ' - %-8.4f' % (torch.max(grad).item())
                    norm_str = 'Norm %8.4f' % (torch.linalg.vector_norm(grad).item())
                names.append(name)
                size_strs.append(size_str)
                norm_strs.append(norm_str)
                intv_strs.append(intv_str)
                dist_strs.append(dist_str)
            if len(names) == 0:
                return self
            maxl_name = max([len(v) for v in names])
            maxl_size = max([len(v) for v in size_strs])
            maxl_norm = max([len(v) for v in norm_strs])
            maxl_intv = max([len(v) for v in intv_strs])
            maxl_dist = max([len(v) for v in dist_strs])
            for name, size_str, norm_str, intv_str, dist_str \
                    in zip(names, size_strs, norm_strs, intv_strs, dist_strs):
                msg = '< Grad > ' + name.ljust(maxl_name) + ' --- ' \
                      + size_str.ljust(maxl_size) + ' | ' \
                      + dist_str.ljust(maxl_dist) + ' | ' \
                      + norm_str.ljust(maxl_norm) + ' | ' \
                      + intv_str.ljust(maxl_intv) + ' | '
                trainer.broadcast(msg)
        return self


# </editor-fold>

# <editor-fold desc='保存管理'>
class ResumeActor(InitialActor):
    def __init__(self, save_pth: str, formatter: str = FORMATTER.SAVE_PTH_SIMPLE):
        self.save_pth = save_pth
        self.formatter = formatter

    def act_init(self, trainer: Trainer, **kwargs):
        save_pth_model = trainer.save_pth_model(self.save_pth, formatter=self.formatter)
        save_pth_optim = trainer.save_pth_optim(self.save_pth, formatter=self.formatter)
        save_pth_state = trainer.save_pth_state(self.save_pth, formatter=self.formatter)
        if os.path.exists(save_pth_model):
            trainer.broadcast('Resume checkpoint from')
            names = ['Model', 'Optimizer', 'State']
            save_pths = [save_pth_model, save_pth_optim, save_pth_state]
            maxl_name = max([len(name) for name in names])
            maxl_svpth = max([len(save_pth) for save_pth in save_pths])
            for name, save_pth in zip(names, save_pths):
                trainer.broadcast('| ' + name.ljust(maxl_name) + ' <- ' + save_pth.ljust(maxl_svpth) + ' |')
            trainer.load(self.save_pth, formatter=self.formatter)
        else:
            trainer.broadcast('Resume failed from non-exist ' + save_pth_model)


class BaseSaveActor(SavePathCollector):

    def __init__(self, save_pth: str, num_keep: int = 3, formatter: str = FORMATTER.SAVE_PTH_EPOCH):
        SavePathCollector.__init__(self, num_keep=num_keep, order_by=SVKEYS.IND_ITER, order_ascend=True)
        self.save_pth = save_pth
        self.formatter = formatter

    def _save(self, trainer: Trainer):
        spth_dct = trainer.save_pth_dct(save_pth=self.save_pth, formatter=self.formatter)
        save_pth = spth_dct[SVKEYS.SAVE_PTH_MODEL]
        prop = {
            SVKEYS.SOURCE: self.__class__.__name__,
            SVKEYS.ACTIVE: True,
            SVKEYS.IND_ITER: trainer.ind_iter,
            SVKEYS.IND_EPOCH: trainer.ind_epoch,
            SVKEYS.SAVE_PTH: save_pth,
        }
        prop.update(spth_dct)

        save_pth_last, prop_last = self.update_save_pth(save_pth, prop)
        if prop_last is not None:
            prop_last[SVKEYS.ACTIVE] = False
        trainer.update_info(prop)

        if save_pth_last is not None and not save_pth_last == save_pth:
            remove_pths_ifexist([prop_last[k] for k in spth_dct.keys()])
        trainer.save(save_pth=self.save_pth, formatter=self.formatter)


class EpochBasedSaveActor(IntervalTrigger, BaseSaveActor, AfterEpochActor):

    def __init__(self, save_pth: str, step: int = 5, offset: int = 0, first: bool = False, last: bool = False,
                 num_keep: int = 1, formatter=FORMATTER.SAVE_PTH_SIMPLE, lev: int = 1):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        BaseSaveActor.__init__(self, save_pth, num_keep=num_keep, formatter=formatter)
        self.lev = lev

    def act_after_epoch(self, trainer, **kwargs):
        if trainer.main_proc and self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            self._save(trainer)

    @property
    def lev_after_epoch(self):
        return self.lev


class IterBasedSaver(IntervalTrigger, BaseSaveActor, AfterIterActor):

    def __init__(self, save_pth: str, step: int = 100, offset: int = 0, first: bool = False, last: bool = False,
                 num_keep: int = 1, formatter=FORMATTER.SAVE_PTH_SIMPLE, lev: int = 1):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        BaseSaveActor.__init__(self, save_pth, num_keep=num_keep, formatter=formatter)
        self.lev = lev

    def act_after_iter(self, trainer, **kwargs):
        if trainer.main_proc and self.trigger(ind=trainer.ind_iter, total=trainer.total_iter):
            self._save(trainer)

    @property
    def lev_after_iter(self):
        return self.lev


# </editor-fold>

# <editor-fold desc='EMA管理'>
class BaseEMAActor(Actor):

    def __init__(self, ema_ratio: float = 0.9):
        self.ema_ratio = ema_ratio

    def apply_ema(self, trainer):
        ema_ratio = self.ema_ratio
        if trainer.model_ema is None:
            model_ema = copy.deepcopy(trainer.model)
            model_ema.eval()
            for para in model_ema.parameters():
                para.requires_grad = False
            trainer.model_ema = model_ema
            return self
        else:
            sd = trainer.model.state_dict()
            sd_ema = trainer.model_ema.state_dict(keep_vars=True)
            for name in sd.keys():
                para = sd[name]
                para_ema = sd_ema[name]
                if para.dtype in [torch.float, torch.half]:
                    para_ema.data = ema_ratio * para_ema.data + (1 - ema_ratio) * para.data
            return self


class IterBasedEMAActor(BaseEMAActor, IntervalTrigger, AfterIterActor):

    def __init__(self, ema_ratio: float = 0.9, step: int = 50, offset: int = 0, first: bool = False,
                 last: bool = False, lev: int = 1):
        BaseEMAActor.__init__(self, ema_ratio=ema_ratio)
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self._lev = lev

    def act_after_iter(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_iter_inep, total=trainer.total_iter_inep):
            self.apply_ema(trainer)

    @property
    def lev_after_iter(self):
        return self._lev


class EpochBasedEMAActor(BaseEMAActor, IntervalTrigger, AfterEpochActor):

    def __init__(self, ema_ratio: float = 0.9, step: int = 50, offset: int = 0, first: bool = False,
                 last: bool = False, lev: int = 1):
        BaseEMAActor.__init__(self, ema_ratio=ema_ratio)
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self._lev = lev

    def act_after_epoch(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            self.apply_ema(trainer)

    @property
    def lev_after_epoch(self):
        return self._lev


# </editor-fold>

# <editor-fold desc='显示管理'>


class TrainerCollectActor(IntervalTrigger, BeforeEpochActor, AfterEpochActor, AfterIterActor):

    def __init__(self, step: int = 50, offset: int = 0, first: bool = False, last: bool = False, title: str = 'Train'):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.title = title

    @staticmethod
    def get_opt_msg(trainer, **kwargs):
        lrs = []
        modules_names = []
        group_indexs = []
        for module_name, opt in trainer.optimizers.items():
            for index, param_group in enumerate(opt.param_groups):
                lrs.append(param_group['lr'])
                modules_names.append(module_name)
                group_indexs.append(index)

        if len(set(lrs)) == 1:
            return 'Lr %-7.6f ' % lrs[0]
        else:
            msg = 'Lr %-7.6f ' % (sum(lrs) / len(lrs))
            msg += ''.join([module_name + '[%d]' % index + ' %-7.6f ' % lr
                            for module_name, index, lr in zip(modules_names, group_indexs, lrs)])
            return msg

    @staticmethod
    def get_loss_msg(trainer, **kwargs):
        return ''.join([name + ' %-5.4f ' % val for name, val in trainer._loss_dct.items()])

    @staticmethod
    def get_time_msg(trainer, **kwargs):
        return ''.join([name + ' %-5.4f ' % val for name, val in trainer._period_dct.items()])

    def act_after_iter(self, trainer: Trainer, **kwargs):
        if not trainer.main_proc or not self.trigger(ind=trainer.ind_iter_inep, total=len(trainer)):
            return None
        msg = []
        msg.append('Iter %06d ' % (trainer.ind_iter + 1) + '[ %04d ] ' % (trainer.ind_iter_inep + 1))
        msg.append(self.get_opt_msg(trainer, **kwargs))
        msg.append(self.get_loss_msg(trainer, **kwargs))
        msg.append(self.get_time_msg(trainer, **kwargs))
        for var_scope in trainer._var_dct.values():
            msg.append(''.join([name + ' ' + str(value) + ' ' for name, value in var_scope.items()]))
        msg = '| '.join(msg) + '|'
        trainer.broadcast(msg)

    def act_before_epoch(self, trainer: Trainer, **kwargs):
        if not trainer.main_proc:
            return None
        msg = ['< ' + self.title + ' >']
        msg.append('Epoch %3d' % (trainer.ind_epoch + 1))
        msg.append('Data %d' % trainer.num_data)
        msg.append('Batch %d' % trainer.num_batch)
        msg_batch = 'BatchSize %d' % trainer.batch_size
        if dist.is_initialized():
            msg_batch += '[x%d GPU]' % dist.get_world_size()
        if trainer.accu_step > 1:
            msg_batch += '[x%d Accu]' % trainer.accu_step
        msg.append(msg_batch)
        msg.append('ImgSize ' + str(trainer.img_size))
        if trainer.learn_gain is not None and not trainer.learn_gain == 1.0:
            msg.append('Gain %-2.1f' % trainer.learn_gain)
        if trainer.enable_half:
            msg.append('[Half]')
        msg.append('ETA ' + sec2msg(trainer.eta))
        trainer.broadcast('  '.join(msg))

    @property
    def lev_after_epoch(self):
        return 5

    @property
    def lev_after_iter(self):
        return 5

    def act_after_epoch(self, trainer: Trainer, **kwargs):

        infos_save = [info for info in trainer.infos if SVKEYS.SAVE_PTH in info.keys()
                      and info[SVKEYS.ACTIVE]]
        if len(infos_save) == 0:
            return None
        infos_save.sort(key=lambda info: info.get(SVKEYS.PERFORMANCE, 0), reverse=True)
        sources = set([info[SVKEYS.SOURCE] for info in infos_save])
        msg = '< Save > Num %d' % len(infos_save) + \
              ' From ' + ', '.join(list(sources))
        trainer.broadcast(msg)
        max_pth_len = max([len(info[SVKEYS.SAVE_PTH]) for info in infos_save])
        for info in infos_save:
            msg = 'Epoch %3d |' % (info.get(SVKEYS.IND_EPOCH, 0) + 1) + \
                  ' Path ' + info[SVKEYS.SAVE_PTH].ljust(max_pth_len) + ' |'
            if SVKEYS.PERFORMANCE in info.keys():
                msg += ' Perfmce %8.4f |' % info[SVKEYS.PERFORMANCE]
            trainer.broadcast(msg)


# </editor-fold>

# <editor-fold desc='运行数据记录'>

class TrainerRecordActor(Actor):

    def __init__(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE):
        self._record_iter = pd.DataFrame()
        self._record_info = pd.DataFrame()
        self.formatter = formatter
        self.save_pth = save_pth

    @staticmethod
    def get_cols_loss(trainer: Trainer):
        names, vals = zip(*trainer._loss_dct.items())
        return list(names), list(vals)

    @staticmethod
    def get_cols_period(trainer: Trainer):
        names, vals = zip(*trainer._period_dct.items())
        return list(names), list(vals)

    @staticmethod
    def get_cols_index(trainer: Trainer):
        names = [SVKEYS.IND_EPOCH, SVKEYS.IND_ITER]
        vals = [trainer.ind_epoch, trainer.ind_iter]
        return names, vals

    @staticmethod
    def get_cols_var(trainer: Trainer):
        names = []
        vals = []
        for scope_name, scope in trainer._var_dct.items():
            for name, val in scope_name.items():
                names.append(name)
                vals.append(val)
        return names, vals

    @staticmethod
    def get_cols_opt(trainer: Trainer):
        lrs = []
        modules_names = []
        group_indexs = []
        for module_name, opt in trainer.optimizers.items():
            for index, param_group in enumerate(opt.param_groups):
                lrs.append(param_group['lr'])
                modules_names.append(module_name)
                group_indexs.append(index)

        if len(lrs) == 1:
            return ['Lr'], lrs
        else:
            names = ['Lr-' + m_name + '-' + str(g) for m_name, g in zip(modules_names, group_indexs)]
            return names, lrs

    def _add_cols_iter(self, trainer: Trainer):
        names_index, vals_index = TrainerRecordActor.get_cols_index(trainer)
        names_loss, vals_loss = TrainerRecordActor.get_cols_loss(trainer)
        names_period, vals_period = TrainerRecordActor.get_cols_period(trainer)
        names_opt, vals_opt = TrainerRecordActor.get_cols_opt(trainer)
        names = names_index + names_opt + names_loss + names_period
        vals = vals_index + vals_opt + vals_loss + vals_period
        row = pd.DataFrame(dict(zip(names, vals)), columns=names, index=[0])
        self._record_iter = pd.concat([self._record_iter, row])
        return row

    def _add_cols_info(self, trainer: Trainer, info):
        if SVKEYS.PERFORMANCE in info.keys():
            row = pd.DataFrame(dict(info), index=[0])
            self._record_info = pd.concat([self._record_info, row])
            return row
        else:
            return None

    def _save(self, trainer: Trainer):
        save_pth = format_save_pth(self.save_pth, default_name='train', ind_iter=trainer.ind_iter + 1,
                                   ind_epoch=trainer.ind_epoch + 1,
                                   appendix='_record', formatter=self.formatter, extend=EXTENDS.EXCEL)
        try:
            writer = pd.ExcelWriter(save_pth, date_format=None, mode='w')
            self._record_iter.to_excel(writer, sheet_name='Iter', index=False)
            self._record_info.to_excel(writer, sheet_name='Info', index=False)
            writer.close()
        except Exception as e:
            pass


class IterBasedTrainerRecordActor(TrainerRecordActor, IntervalTrigger, AfterIterActor, AfterSaveActor,
                                  AfterAddInfoActor):

    def __init__(self, save_pth: str, step: int, offset: int = 0, first: bool = False, last: bool = False,
                 formatter=FORMATTER.SAVE_PTH_SIMPLE):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        TrainerRecordActor.__init__(self, save_pth=save_pth, formatter=formatter)

    def act_after_iter(self, trainer: Trainer, **kwargs):
        if not trainer.main_proc or not self.trigger(ind=trainer.ind_iter_inep, total=trainer.total_iter_inep):
            return None
        self._add_cols_iter(trainer)

    def act_after_save(self, trainer: Trainer, save_pth, **kwargs):
        if not trainer.main_proc:
            return None
        self._save(trainer)

    def act_after_addinfo(self, trainer: Trainer, info, **kwargs):
        if not trainer.main_proc:
            return None
        self._add_cols_info(trainer, info)

    @property
    def lev_after_iter(self):
        return 0


class EpochBasedTrainerRecordActor(TrainerRecordActor, IntervalTrigger, AfterEpochActor, AfterSaveActor,
                                   AfterAddInfoActor):
    def __init__(self, save_pth: str, step: int, offset: int = 0, first: bool = False, last: bool = False,
                 formatter=FORMATTER.SAVE_PTH_SIMPLE):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        TrainerRecordActor.__init__(self, save_pth=save_pth, formatter=formatter)

    def act_after_epoch(self, trainer: Trainer, **kwargs):
        if not trainer.main_proc or not self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            return None
        self._add_cols_iter(trainer)

    def act_after_save(self, trainer: Trainer, save_pth, **kwargs):
        if not trainer.main_proc:
            return None
        self._save(trainer)

    def act_after_addinfo(self, trainer: Trainer, info, **kwargs):
        if not trainer.main_proc:
            return None
        self._add_cols_info(trainer, info)

    @property
    def lev_after_epoch(self):
        return 0

# </editor-fold>


# if __name__ == '__main__':

#     loader=
