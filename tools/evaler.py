from torch.cuda.amp import autocast

from datas import COCOWriter
from .base import *
from .base.container import _set_model_state
from .metric import *
from .metric.cocostd import _summarize_coco_eval, _eval_coco_obj


# <editor-fold desc='Evaler类封装'>


class Evaler(IterBasedTemplate, VariableManager):

    def __init__(self,
                 loader: IMDataLoader,
                 total_epoch: int = 1,
                 device: torch.device = DEVICE,
                 enable_half: bool = True,
                 kwargs_eval: Optional[Dict] = None):
        IterBasedTemplate.__init__(
            self, total_epoch=total_epoch, total_iter=total_epoch * len(loader), loader=loader,
            device=device, processor=None)
        VariableManager.__init__(self)
        self.cind2name = loader.cind2name
        self.kwargs_eval = {} if kwargs_eval is None else kwargs_eval
        self.labels_cmb = None
        self.eval_results = None
        self.report = None
        self.enable_half = enable_half

    def set_state(self, model):
        return _set_model_state(model, train_mode=False, enable_half=self.enable_half)

    def act_init(self, model, *args, **kwargs):
        if isinstance(model, EvalableModel):
            self.model = model
            if isinstance(model, HasDevice):
                self.device = model.device
            else:
                self.device = DEVICE
            self.set_state(model)
            self.running = True
            self.labels_cmb = []
            self.eval_results = []
            self.ind_iter = 0
            self.ind_epoch = 0
            self.add_actor(model)
            self.model.act_init_eval(self)
        elif isinstance(model, list) or isinstance(model, tuple):
            self.labels_cmb = model
            self.eval_results = [self._eval_label_pair(m, d) for m, d in self.labels_cmb]
            self.running = False
        elif isinstance(model, str):
            self.broadcast('Using test cache at ' + model)
            buffer = load_pkl(model, extend=EXTENDS.CACHE)
            self.labels_cmb = buffer['labels_cmb']
            self.eval_results = [self._eval_label_pair(m, d) for m, d in self.labels_cmb]
            self.running = False
        else:
            raise Exception('fmt err ' + model.__class__.__name__)
        return None

    def act_return(self):
        if self.main_proc and self.report is not None:
            val = self._eval_metric(self.report)
        else:
            val = 0
        return val

    def act_ending(self):
        self.eval_results = all_extend_object_list(self.eval_results)
        if self.main_proc:
            self.update_time(TIMENODE.BEFORE_CALC)
            if len(self.eval_results) > 0:
                self.report = self._eval_sum_report(self.eval_results)
            else:
                self.broadcast('No eval result')
                self.report = None
            self.update_time(TIMENODE.AFTER_CALC)
        return None

    def act_iter(self):
        imgs, labels_ds = self.batch_data
        with torch.no_grad():
            with autocast(enabled=self.enable_half):
                self.update_time(TIMENODE.BEFORE_INFER)
                imgs_md, labels_md = self.model.act_iter_eval(self, imgs, labels_ds, **self.kwargs_eval)
                self.update_time(TIMENODE.AFTER_INFER)
        self.infered_data = (imgs_md, labels_md)
        for label_ds, label_md in zip(labels_ds, labels_md):
            if label_md is not None and label_ds is not None:
                label_md.info_from(label_ds)
            self.eval_results.append(self._eval_label_pair(label_md, label_ds))
        return None

    @abstractmethod
    def _eval_sum_report(self, eval_results):
        print('*** core calc opr ***')
        pass

    @abstractmethod
    def _eval_metric(self, report):
        print('*** core calc opr ***')
        pass

    @abstractmethod
    def _eval_label_pair(self, label_md, label_ds):
        print('*** core calc opr ***')
        pass


# </editor-fold>

# <editor-fold desc='Evaler扩展'>
class EvalerReportActor(EndingActor):
    def __init__(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE):
        self.save_pth = format_save_pth(
            save_pth=save_pth, formatter=formatter, extend=EXTENDS.EXCEL,
            default_name='eval', appendix='_report')

    def act_ending(self, evaler: Evaler, **kwargs):
        if evaler.main_proc and len(self.save_pth) > 0:
            try:
                evaler.broadcast('Save report to ' + self.save_pth)
                evaler.report.to_excel(self.save_pth, index=False)
            except Exception as e:
                pass


class EvalerLabelCollectActor(EndingActor, AfterIterActor):
    def __init__(self, with_recover: bool = False):
        self.with_recover = with_recover

    def act_ending(self, evaler: Evaler, **kwargs):
        evaler.labels_cmb = all_extend_object_list(evaler.labels_cmb)
        return None

    def act_after_iter(self, evaler, **kwargs):
        imgs_ds, labels_ds = evaler.batch_data
        imgs_md, labels_md = evaler.infered_data
        for label_ds, label_md in zip(labels_ds, labels_md):
            if label_md is not None and self.with_recover:
                label_md.recover()
            if label_ds is not None and self.with_recover:
                label_ds.recover()
            evaler.labels_cmb.append((label_md, label_ds))
        return None


class EvalerCacheActor(EvalerLabelCollectActor, InitialActor):
    def act_init(self, evaler: Evaler, **kwargs):
        if evaler.main_proc and os.path.exists(self.save_pth):
            evaler.broadcast('Use cache at ' + self.save_pth)
            evaler.labels_cmb = load_pkl(file_pth=self.save_pth)
            evaler.eval_results = [evaler._eval_label_pair(m, d) for m, d in evaler.labels_cmb]
            evaler.running = False

    def __init__(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE, with_recover: bool = False):
        EvalerLabelCollectActor.__init__(self, with_recover=with_recover)
        self.save_pth = format_save_pth(
            save_pth=save_pth, formatter=formatter, extend=EXTENDS.CACHE,
            default_name='eval', appendix='_cache')

    def act_ending(self, evaler: Evaler, **kwargs):
        EvalerLabelCollectActor.act_ending(self, evaler)
        if evaler.main_proc and len(self.save_pth) > 0 and not os.path.exists(self.save_pth):
            evaler.broadcast('Save cache to ' + self.save_pth)
            save_pkl(file_pth=self.save_pth, obj=evaler.labels_cmb)


class EvalerCollectActor(IntervalTrigger, BeforeEpochActor, AfterIterActor, AfterCycleActor, EndingActor):

    def __init__(self, step: int = 50, offset: int = 0, first: bool = False, last: bool = False, title: str = 'Eval'):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.periods_infer = []
        self.title = title

    def get_time_msg(self, evaler: Evaler, **kwargs) -> str:
        return ''.join([name + ' %-5.4f ' % val for name, val in evaler._period_dct.items()])

    def act_after_iter(self, evaler: Evaler, **kwargs):
        self.periods_infer.append(evaler.collect_period(PERIOD.INFER))
        if not evaler.main_proc or not self.trigger(evaler.ind_iter_inep, len(evaler)):
            return None
        msg = []
        msg.append('Iter %06d ' % (evaler.ind_iter + 1) + '[ %04d ] ' % (evaler.ind_iter_inep + 1))
        msg.append(self.get_time_msg(evaler, **kwargs))
        msg.append('ETA ' + sec2msg(evaler.eta) + ' ')
        for var_scope in evaler._var_dct.values():
            msg.append(''.join([name + ' ' + str(value) + ' ' for name, value in var_scope.items()]))
        msg = '| '.join(msg) + '|'
        evaler.broadcast(msg)

    def act_before_epoch(self, evaler, **kwargs):
        if not evaler.main_proc:
            return None
        msg = ['< ' + self.title + ' >']
        msg.append('Epoch %d' % (evaler.ind_epoch + 1))
        msg.append('Data %d' % evaler.num_data)
        msg.append('Batch %d' % evaler.num_batch)
        msg_batch = 'BatchSize %d' % evaler.batch_size
        if dist.is_initialized():
            msg_batch += '[x%d GPU]' % dist.get_world_size()
        msg.append(msg_batch)
        msg.append('ImgSize ' + str(evaler.img_size))
        if evaler.enable_half:
            msg.append('[Half]')
        msg.append('ETA ' + sec2msg(evaler.eta))
        evaler.broadcast('  '.join(msg))

    def act_after_cycle(self, evaler, **kwargs):
        if not evaler.main_proc:
            return None
        mean_infer = np.mean(self.periods_infer)
        msg = 'Average infer time %f' % (mean_infer / evaler.batch_size) \
              + ' [ %f' % mean_infer + ' / %d ]' % evaler.batch_size
        evaler.broadcast(msg)

    def act_ending(self, evaler: Evaler, **kwargs):
        if evaler.main_proc and evaler.report is not None:
            evaler.broadcast_dataframe(evaler.report)


class EvalerBuffer(AfterCycleActor):

    def __init__(self, cache_pth):
        self.cache_pth = cache_pth

    def act_after_cycle(self, evaler, **kwargs):
        cache_pth = ensure_extend(self.cache_pth, EXTENDS.CACHE)
        save_pkl(obj=evaler.labels_cmb, file_pth=cache_pth)


class EvalActor(SavePathCollector):

    def __init__(self, evaler: Evaler, save_pth: str = '', num_keep: int = 1,
                 formatter=FORMATTER.SAVE_PTH_PRFM,
                 select_ema=False):
        SavePathCollector.__init__(self, num_keep=num_keep, order_by=SVKEYS.PERFORMANCE, order_ascend=True)
        self.evaler = evaler
        self.save_pth = save_pth
        self.formatter = formatter
        self.select_ema = select_ema

    def _proc_info(self, trainer, perfmce, key, save_pth):
        prop = {
            SVKEYS.SOURCE: self.__class__.__name__,
            SVKEYS.ACTIVE: True,
            SVKEYS.IND_ITER: trainer.ind_iter,
            SVKEYS.IND_EPOCH: trainer.ind_epoch,
            SVKEYS.SAVE_PTH: save_pth,
            key: save_pth,
            SVKEYS.PERFORMANCE: perfmce
        }
        save_pth_last, prop_last = self.update_save_pth(save_pth, prop)
        if prop_last is not None:
            prop_last[SVKEYS.ACTIVE] = False
        trainer.update_info(prop)
        return save_pth_last, prop_last

    def _eval_and_save(self, trainer):
        if self.select_ema:
            model = trainer.model_ema
            save_pth_func = trainer.save_pth_model_ema
            save_func = trainer.save_model_ema
            key = SVKEYS.SAVE_PTH_MODEL_EMA
        else:
            model = trainer.model
            save_pth_func = trainer.save_pth_model
            save_func = trainer.save_model
            key = SVKEYS.SAVE_PTH_MODEL

        if model is None:
            return None
        perfmce = self.evaler.start(model)
        trainer.set_state(model)
        save_pth = save_pth_func(save_pth=self.save_pth, formatter=self.formatter, perfmce=perfmce)
        if not trainer.main_proc or len(self.save_pth) == 0:
            return None
        save_pth_last, prop_last = self._proc_info(
            trainer, perfmce, save_pth=save_pth, key=key)
        if save_pth_last is not None and not save_pth_last == save_pth:
            remove_pth_ifexist(save_pth_last)
        if not save_pth_last == save_pth:
            save_func(save_pth=self.save_pth, formatter=self.formatter, perfmce=perfmce)
        return None


class EpochBasedEvalActor(EvalActor, IntervalTrigger, AfterEpochActor):

    def __init__(self, evaler: Evaler, step: int, save_pth: str = '', offset: int = 0, first: bool = False,
                 last: bool = False, num_keep: int = 1, formatter=FORMATTER.SAVE_PTH_PRFM, select_ema: bool = False,
                 lev: int = 1):
        EvalActor.__init__(self, evaler, num_keep=num_keep, save_pth=save_pth, formatter=formatter,
                           select_ema=select_ema)
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.lev = lev

    def act_after_epoch(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            self._eval_and_save(trainer)

    @property
    def lev_after_epoch(self):
        return self.lev


class IterBasedEvalActor(EvalActor, IntervalTrigger, AfterIterActor):

    def __init__(self, evaler: Evaler, step: int, save_pth: str = '', offset: int = 0, first: bool = False,
                 last: bool = False, num_keep: int = 1, formatter=FORMATTER.SAVE_PTH_PRFM, select_ema: bool = False,
                 lev: int = 1):
        EvalActor.__init__(self, evaler, num_keep=num_keep, save_pth=save_pth, formatter=formatter,
                           select_ema=select_ema)
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.lev = lev

    def act_after_iter(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_iter, total=trainer.total_iter):
            self._eval_and_save(trainer)

    @property
    def lev_after_iter(self):
        return self.lev


# </editor-fold>

# <editor-fold desc='Evaler类型'>

class AccuracyEvaler(Evaler):

    def __init__(self, loader: IMDataLoader, total_epoch: int = 1, device: torch.device = DEVICE,
                 enable_half: bool = True, kwargs_eval: Optional[Dict] = None,
                 top_nums: Tuple[int, ...] = (1, 5), ):
        Evaler.__init__(self, loader=loader, total_epoch=total_epoch, device=device,
                        enable_half=enable_half, kwargs_eval=kwargs_eval)
        self.top_nums = top_nums

    def _eval_label_pair(self, label_md, label_ds):
        return eval_pair_tpacc(label_md, label_ds, top_nums=self.top_nums)

    def _eval_sum_report(self, results):
        cinds_ct_md, cinds_ds = [np.array(res) for res in zip(*results)]
        ns_ds, accs = acc_top_nums(cinds_ct_md, cinds_ds, top_nums=self.top_nums, num_cls=self.loader.num_cls)
        data = AccuracyEvaler.report_data(ns_ds, accs, top_nums=self.top_nums, cind2name=self.loader.cind2name, )
        return data

    def _eval_metric(self, report):
        return report.iloc[-1, -1]

    @staticmethod
    def report_data(ns_ds, accs, top_nums: Tuple[int, ...] = (1, 5), cind2name: Optional[Callable] = None,
                    ignore_empty: bool = True):
        num_cls = len(ns_ds)
        data = pd.DataFrame(columns=['Class', 'Target', ] + ['AccT%d' % n for n in top_nums])
        for i in range(num_cls):
            name = cind2name(i) if cind2name is not None else i
            if ignore_empty and ns_ds[i] == 0:
                continue
            row_dct = {
                'Class': name,
                'Target': ns_ds[i],
            }
            for j, n in enumerate(top_nums):
                row_dct['AccT%d' % n] = accs[i, j]
            data = pd.concat([data, pd.DataFrame(row_dct, index=[0])])
        if len(data) > 1:
            n_ds = np.sum(ns_ds)
            ns_ds_valid = np.array(data['Target'])
            row_dct = {
                'Class': 'Total',
                'Target': n_ds
            }
            for j, n in enumerate(top_nums):
                n_col = 'AccT%d' % n
                row_dct[n_col] = np.sum(np.array(data[n_col]) * ns_ds_valid) / n_ds
            data = pd.concat([data, pd.DataFrame(row_dct, index=[0])])
        return data


class F1Evaler(Evaler):

    def _eval_label_pair(self, label_md, label_ds):
        cind_md = label_md.category.cindN
        cind_ds = label_ds.category.cindN
        return cind_md, cind_ds

    def _eval_sum_report(self, results):
        cinds_md, cinds_ds = [np.array(res) for res in zip(*results)]
        tp, tn, fp, fn = confusion_per_class(cinds_md=cinds_md, cinds_ds=cinds_ds, num_cls=self.loader.num_cls)
        data = F1Evaler.report_data(tp, tn, fp, fn, cind2name=self.loader.cind2name,
                                    ignore_empty=True)
        return data

    def _eval_metric(self, report):
        return np.array(report['F1'])[-1]

    @staticmethod
    def report_data(tps, tns, fps, fns, cind2name: Optional[Callable] = None, ignore_empty: bool = True):
        precs, recls, f1s, accs = f1_from_confusion(tps, tns, fps, fns)
        ns_ds = tps + fns
        ns_md = tps + fps
        num_cls = len(tps)
        data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'TP', 'Percison', 'Recall', 'F1', 'Accuracy'])
        for i in range(num_cls):
            name = cind2name(i) if cind2name is not None else i
            if ignore_empty and ns_ds[i] == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name,
                'Target': ns_ds[i],
                'Pred': ns_md[i],
                'TP': tps[i],
                'Percison': precs[i],
                'Recall': recls[i],
                'F1': f1s[i],
                'Accuracy': accs[i]
            }, index=[0])])
        if len(data) > 1:
            lb_ds_sum = np.maximum(np.sum(np.array(ns_ds)), 1)
            lb_md_sum = np.maximum(np.sum(np.array(ns_md)), 1)
            ncls_valid = np.maximum(np.sum(ns_ds > 0), 1)
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total',
                'Target': lb_ds_sum,
                'Pred': np.sum(np.array(data['Pred'])),
                'TP': np.sum(np.array(data['TP'])),
                'Percison': np.sum(np.array(data['Percison']) * np.array(data['Pred'])) / lb_md_sum,
                'Recall': np.sum(np.array(data['Recall']) * np.array(data['Target'])) / lb_ds_sum,
                'F1': np.sum(data['F1']) / ncls_valid,
                'Accuracy': np.sum(data['Accuracy']) / ncls_valid,
            }, index=[0])])
        return data


class AUCEvaler(Evaler):

    def _eval_metric(self, report):
        return np.array(report['AUC'])[-1]

    def _eval_label_pair(self, label_md, label_ds):
        chot_md = OneHotCategory.convert(label_md.category)._chotN
        cind_ds = label_ds.category.cindN
        return chot_md, cind_ds

    def _eval_sum_report(self, results):
        chots_md, cinds_ds = [np.array(res) for res in zip(*results)]
        cinds_md = chotN2cindN(chots_md)
        tp, tn, fp, fn = confusion_per_class(cinds_md=cinds_md, cinds_ds=cinds_ds, num_cls=self.loader.num_cls)
        aucs = auc_per_class(chots_md, cinds_ds, num_cls=self.loader.num_cls)
        data = AUCEvaler.report_data(tp, tn, fp, fn, aucs, cind2name=self.loader.cind2name, ignore_empty=True)
        return data

    @staticmethod
    def report_data(tps, tns, fps, fns, aucs, cind2name: Optional[Callable] = None, ignore_empty: bool = True):
        ns_ds = tps + fns
        ns_md = tps + fps
        num_cls = len(tps)
        data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'TP', 'AUC'])
        for i in range(num_cls):
            name = cind2name(i) if cind2name is not None else i
            if ignore_empty and ns_ds[i] == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name,
                'Target': ns_ds[i],
                'Pred': ns_md[i],
                'TP': tps[i],
                'AUC': aucs[i]
            }, index=[0])])
        if len(data) > 1:
            ncls_valid = np.maximum(np.sum(ns_ds > 0), 1)
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total',
                'Target': np.sum(data['Target']),
                'Pred': np.sum(data['Pred']),
                'TP': np.sum(data['TP']),
                'AUC': np.sum(data['AUC']) / ncls_valid
            }, index=[0])])
        return data


class BoxVOCEvaler(Evaler):

    def __init__(self, loader: IMDataLoader, total_epoch: int = 1, device: torch.device = DEVICE,
                 enable_half: bool = True, kwargs_eval: Optional[Dict] = None,
                 iou_thres: float = 0.5, ignore_class: bool = False):
        Evaler.__init__(self, loader=loader, total_epoch=total_epoch, device=device,
                        enable_half=enable_half, kwargs_eval=kwargs_eval)
        self.iou_thres = iou_thres
        self.ignore_class = ignore_class

    def _eval_label_pair(self, label_md, label_ds):
        return eval_pair_vocap(
            label_md, label_ds, label_ropr_mat=label_ropr_mat_box, iou_thres=self.iou_thres,
            ignore_class=self.ignore_class)

    def _eval_sum_report(self, results):
        results = [np.concatenate(res, axis=0) for res in zip(*results)]
        cinds_ds_acpt, cinds_md_acpt, confs_md_acpt, masks_md_pos, masks_md_neg = results
        ap = ap_per_class(cinds_ds_acpt, cinds_md_acpt, confs_md_acpt, masks_md_pos, masks_md_neg,
                          num_cls=self.loader.num_cls, interp=False)
        n_ds = np.bincount(cinds_ds_acpt, minlength=self.loader.num_cls)
        n_md = np.bincount(cinds_md_acpt, minlength=self.loader.num_cls)

        tp, tn, fp, fn = confusion_per_class_det(
            cinds_md=cinds_md_acpt, masks_md_pos=masks_md_pos,
            masks_md_neg=masks_md_neg, num_cls=self.loader.num_cls)
        prec, recl, f1 = precrecl_from_confusion_det(tp, n_md, n_ds)
        data = BoxVOCEvaler.report_data(tp, n_md, n_ds, prec, recl, f1, ap,
                                        cind2name=self.loader.cind2name, iou_thres=self.iou_thres)
        return data

    def _eval_metric(self, report):
        return report.iloc[-1, -1]

    @staticmethod
    def report_data(tps, ns_md, ns_ds, precs, recls, f1s, aps, cind2name: Optional[Callable] = None,
                    ignore_empty: bool = True, iou_thres: float = 0.5):
        ap_name = 'AP%2d' % (iou_thres * 100)
        num_cls = len(aps)
        data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'TP', 'Recall', 'Precision', 'F1', ap_name])
        for cind in range(num_cls):
            name = cind2name(cind) if cind2name is not None else str(cind)
            if ignore_empty and ns_ds[cind] == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name,
                'Target': ns_ds[cind],
                'Pred': ns_md[cind],
                'Precision': precs[cind],
                'F1': f1s[cind],
                'TP': tps[cind],
                'Recall': recls[cind],
                ap_name: aps[cind]
            }, index=[0])])
        if len(data) > 1:
            ncls_valid = np.sum(np.array(data['Target']) > 0)
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total',
                'Target': np.sum(data['Target']),
                'Pred': np.sum(data['Pred']),
                'TP': np.sum(data['TP']),
                'Recall': np.sum(data['Recall']) / ncls_valid,
                'Precision': np.sum(data['Precision']) / ncls_valid,
                'F1': np.sum(data['F1']) / ncls_valid,
                ap_name: np.sum(data[ap_name]) / ncls_valid
            }, index=[0])])
        return data


class InstVOCEvaler(BoxVOCEvaler):

    def _eval_label_pair(self, label_md, label_ds):
        return eval_pair_vocap(
            label_md, label_ds, label_ropr_mat=label_ropr_mat_inst, iou_thres=self.iou_thres,
            ignore_class=self.ignore_class)


class BoxCOCOEvaler(Evaler):

    def __init__(self, loader: IMDataLoader, total_epoch: int = 1, device: torch.device = DEVICE,
                 enable_half: bool = True, kwargs_eval: Optional[Dict] = None,
                 iou_thress: Sequence[float] = (0.5, 0.55, 0.6),
                 ignore_class: bool = False):
        Evaler.__init__(self, loader=loader, total_epoch=total_epoch, device=device,
                        enable_half=enable_half, kwargs_eval=kwargs_eval)
        self.iou_thress = iou_thress
        self.ignore_class = ignore_class

    def _eval_label_pair(self, label_md, label_ds):
        return eval_pair_cocoap(
            label_md, label_ds, label_ropr_mat=label_ropr_mat_box, iou_thress=self.iou_thress,
            ignore_class=self.ignore_class)

    def _eval_sum_report(self, results):
        results = [np.concatenate(res, axis=0) for res in zip(*results)]
        cinds_ds_acpt, cinds_md_acpt, confs_md_acpt, masks_md_pos, masks_md_neg = results
        aps = []
        for j in range(len(self.iou_thress)):
            ap = ap_per_class(cinds_ds_acpt, cinds_md_acpt, confs_md_acpt, masks_md_pos[:, j], masks_md_neg[:, j],
                              num_cls=self.loader.num_cls, interp=False)
            aps.append(ap)
        aps = np.stack(aps, axis=-1)
        n_ds = np.bincount(cinds_ds_acpt, minlength=self.loader.num_cls)
        data = BoxCOCOEvaler.report_data(n_ds, aps, cind2name=self.loader.cind2name, )
        return data

    def _eval_metric(self, report):
        return np.array(report['AP'])[-1]

    @staticmethod
    def report_data(ns_ds, aps, cind2name: Optional[Callable] = None,
                    iou_thress: Sequence[float] = (0.5, 0.55, 0.6), ignore_empty: bool = True):
        num_cls = len(ns_ds)
        ap_aver = np.mean(aps, axis=-1)
        names_ap = ['AP@%2d' % (iou_thres * 100) for iou_thres in iou_thress]
        data = pd.DataFrame(columns=['Class', 'Target', 'AP'] + names_ap)
        for cind in range(num_cls):
            name = cind2name(cind) if cind2name is not None else str(cind)
            if ignore_empty and ns_ds[cind] == 0:
                continue
            row_dct = {
                'Class': name,
                'Target': ns_ds[cind],
                'AP': ap_aver[cind]
            }
            for name_ap, ap in zip(names_ap, aps[cind]):
                row_dct[name_ap] = ap
            data = pd.concat([data, pd.DataFrame(row_dct, index=[0])])

        if len(data) > 1:
            ncls_valid = np.sum(np.array(data['Target']) > 0)
            row_dct = {
                'Class': 'Total',
                'Target': np.sum(data['Target']),
                'AP': np.sum(data['AP']) / ncls_valid
            }
            for name_ap, ap in zip(names_ap, aps):
                row_dct[name_ap] = np.sum(data[name_ap]) / ncls_valid
            data = pd.concat([data, pd.DataFrame(row_dct, index=[0])])
        return data


class InstCOCOEvaler(BoxCOCOEvaler):
    def _eval_label_pair(self, label_md, label_ds):
        return eval_pair_cocoap(
            label_md, label_ds, label_ropr_mat=label_ropr_mat_inst, iou_thress=self.iou_thress,
            ignore_class=self.ignore_class)


class BoxCOCOSTDEvaler(Evaler):
    def _eval_metric(self, report):
        return np.array(report['AP'])[-1]

    def __init__(self, loader: IMDataLoader, total_epoch: int = 1, device: torch.device = DEVICE,
                 enable_half: bool = True, kwargs_eval: Optional[Dict] = None,
                 ignore_class: bool = False):
        Evaler.__init__(self, loader=loader, total_epoch=total_epoch, device=device,
                        enable_half=enable_half, kwargs_eval=kwargs_eval)
        self.ignore_class = ignore_class

    def _eval_label_pair(self, label_md, label_ds):
        anno_md = COCOWriter.label2json_anno(label_md, with_score=True, with_rgn=False)
        img_info, anno_ds = COCOWriter.label2json_item(label_ds, with_score=False, with_rgn=False)
        return (img_info, anno_md, anno_ds)

    def _eval_sum_report(self, results):
        img_infos, annos_md, annos_ds = zip(*results)
        json_dct = COCOWriter.json_itemsjson_dct(list(zip(img_infos, annos_ds)), img_id_init=0)
        # coco_ds = COCOWriter.json_dct2coco_obj(json_dct)
        annos_md_flat = []
        for i, anno_md in enumerate(annos_md):
            for a_md in anno_md:
                a_md['image_id'] = i
                annos_md_flat.append(a_md)

        data = _eval_coco_obj(coco_dct_md=annos_md_flat,
                              coco_dct_lb=json_dct, eval_type='bbox', ignore_class=self.ignore_class)
        # coco_pd = coco_ds.loadRes(annos_md_flat)
        # coco_eval = COCOeval(coco_ds, coco_pd, 'bbox')
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()
        # data = _summarize_coco_eval(coco_eval, cind2name=self.loader.cind2name)
        return data


class InstCOCOSTDEvaler(BoxCOCOSTDEvaler):

    def _eval_label_pair(self, label_md, label_ds):
        anno_md = COCOWriter.label2json_item(label_md, with_score=True, with_rgn=False)
        img_info, anno_ds = COCOWriter.label2json_anno(label_ds, with_score=False, with_rgn=False)
        return (img_info, anno_md, anno_ds)

    def _eval_sum_report(self, results):
        img_infos, annos_md, annos_ds = zip(*results)
        json_dct = COCOWriter.json_itemsjson_dct(list(zip(img_infos, annos_ds)), img_id_init=0)
        coco_ds = COCODataset.json_dct2coco_obj(json_dct)
        annos_md_flat = []
        for i, anno_md in enumerate(annos_md):
            for a_md in anno_md:
                a_md['image_id'] = i
                annos_md_flat.append(a_md)
        coco_pd = coco_ds.loadRes(annos_md_flat)
        coco_eval = COCOeval(coco_ds, coco_pd, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        data = _summarize_coco_eval(coco_eval, cind2name=self.loader.cind2name)
        return data

    # def _eval_sum_report(self, results):
    #     labels_md, labels_ds, = zip(*results)
    #     anno_md = COCODataset.labels2json_lst(labels_md, with_score=True, with_rgn=True)
    #     coco_ds = COCODataset.labels2coco_obj(labels_ds, with_score=False, with_rgn=True)
    #     coco_pd = coco_ds.loadRes(anno_md)
    #     coco_eval = COCOeval(coco_ds, coco_pd, 'segm')
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()
    #     data = _summarize_coco_eval(coco_eval, cind2name=self.loader.cind2name)
    #     return data


class MIOUEvaler(Evaler):

    def _eval_metric(self, report):
        return np.array(report['IOU'])[-1]

    def _eval_label_pair(self, label_md, label_ds):
        return eval_pair_miou(label_md, label_ds, num_cls=self.loader.num_cls)

    def _eval_sum_report(self, results):
        tp, fp, tn, fn = [np.array(res) for res in zip(*results)]
        report = MIOUEvaler.report_data(tp, fp, tn, fn,
                                        cind2name=self.loader.cind2name, ignore_empty=True)
        return report

    @staticmethod
    def report_data(tps, fps, tns, fns, cind2name=None, ignore_empty=True):
        num_cls = tps.shape[-1]
        nlbs_ds, nlbs_md, precs, recls, f1s, accs, ious, dices = \
            eval_sum_miou_eff(tps, fps, tns, fns, scale=1000)
        data = pd.DataFrame(
            columns=['Class', 'Target(k)', 'Pred(k)', 'Percison', 'Recall', 'F1', 'Accuracy', 'IOU', 'Dice'])
        for i in range(num_cls):
            name = cind2name(i) if cind2name is not None else i
            if ignore_empty and nlbs_ds[i] == 0:
                continue
            data = pd.concat([data, pd.DataFrame({
                'Class': name,
                'Target(k)': nlbs_ds[i],
                'Pred(k)': nlbs_md[i],
                'Dice': dices[i],
                'Percison': precs[i],
                'Recall': recls[i],
                'F1': f1s[i],
                'Accuracy': accs[i],
                'IOU': ious[i]
            }, index=[0])])
        if len(data) > 1:
            nlb_ds_sum = np.maximum(np.sum(np.array(nlbs_ds)), 1)
            nlb_md_sum = np.maximum(np.sum(np.array(nlbs_md)), 1)
            ncls_valid = np.maximum(np.sum(nlbs_ds > 0), 1)
            data = pd.concat([data, pd.DataFrame({
                'Class': 'Total',
                'Target(k)': np.sum(data['Target(k)']),
                'Pred(k)': np.sum(data['Pred(k)']),
                'Percison': np.sum(np.array(data['Percison']) * nlbs_md) / nlb_md_sum,
                'Recall': np.sum(np.array(data['Recall']) * nlbs_ds) / nlb_ds_sum,
                'F1': np.sum(data['F1']) / ncls_valid,
                'Accuracy': np.average(data['Accuracy']),
                'IOU': np.sum(data['IOU']) / ncls_valid,
                'Dice': np.sum(data['Dice']) / ncls_valid,
            }, index=[0])])
        return data

# </editor-fold>
