from utils import *


class SVKEYS:
    IND_ITER = 'ind_iter'
    IND_ITER_INEP = 'ind_iter_inep'
    IND_EPOCH = 'ind_epoch'
    RUNNING = 'running'
    PERFORMANCE = 'performance'
    SAVE_PTH = 'save_pth'
    SAVE_PTH_MODEL = 'save_pth_model'
    SAVE_PTH_MODEL_EMA = 'save_pth_model_ema'
    SAVE_PTH_OPTIM = 'save_pth_optim'
    SAVE_PTH_STATE = 'save_pth_state'
    SOURCE = 'source'
    ACTIVE = 'active'
    INFOS = 'infos'
    TIME_DCT = 'time_dct'
    TIME_INV = 'time_inv'
    PERIOD_DCT = 'period_dct'
    PERIOD_DCT_AUTO = 'period_dct_auto'
    LOSS_DCT = 'loss_dct'
    VAR_DCT = 'var_dct'
    SVAE_PTHS_PROPS = 'save_pths_props'
    CLASS_NAME = 'class_name'


# <editor-fold desc='事件动作'>


class Actor(metaclass=ABCMeta):

    def act_add(self, container, **kwargs):
        pass


class IActorContainer(metaclass=ABCMeta):

    @abstractmethod
    def get_actors(self, actor_type: type) -> Sequence:
        pass


class InitialActor(Actor):

    @abstractmethod
    def act_init(self, container, **kwargs):
        pass

    @property
    def lev_init(self):
        return 1


class EndingActor(Actor):

    @abstractmethod
    def act_ending(self, container, **kwargs):
        pass

    @property
    def lev_ending(self):
        return 1


class BeforeIterActor(Actor):

    @abstractmethod
    def act_before_iter(self, container, **kwargs):
        pass

    @property
    def lev_before_iter(self):
        return 1


class AfterIterActor(Actor):
    @abstractmethod
    def act_after_iter(self, container, **kwargs):
        pass

    @property
    def lev_after_iter(self):
        return 1


class BeforeEpochActor(Actor):

    @abstractmethod
    def act_before_epoch(self, container, **kwargs):
        pass

    @property
    def lev_before_epoch(self):
        return 1


class AfterCycleActor(Actor):
    @abstractmethod
    def act_after_cycle(self, container, **kwargs):
        pass

    @property
    def lev_after_cycle(self):
        return 1


class BeforeCycleActor(Actor):

    @abstractmethod
    def act_before_cycle(self, container, **kwargs):
        pass

    @property
    def lev_before_cycle(self):
        return 1


class AfterEpochActor(Actor):
    @abstractmethod
    def act_after_epoch(self, container, **kwargs):
        pass

    @property
    def lev_after_epoch(self):
        return 1


class AfterSaveActor(Actor):

    @abstractmethod
    def act_after_save(self, container, save_pth, **kwargs):
        pass

    @property
    def lev_after_save(self):
        return 1


class AfterLoadActor(Actor):
    @abstractmethod
    def act_after_load(self, container, save_pth, **kwargs):
        pass

    @property
    def lev_after_load(self):
        return 1


class AtBroadcastActor(Actor):
    @abstractmethod
    def act_broadcast(self, container, msg, **kwargs):
        pass

    @property
    def lev_broadcast(self):
        return 1


class BeforeAddInfoActor(Actor):
    @abstractmethod
    def act_before_addinfo(self, container, info, **kwargs):
        pass

    @property
    def lev_before_addinfo(self):
        return 1


class AfterAddInfoActor(Actor):
    @abstractmethod
    def act_after_addinfo(self, container, info, **kwargs):
        pass

    @property
    def lev_after_addinfo(self):
        return 1


class AfterBackwardActor(Actor):
    @abstractmethod
    def act_after_backward(self, container, info, **kwargs):
        pass

    @property
    def lev_after_backward(self):
        return 1


class BeforeOptimizeActor(Actor):
    @abstractmethod
    def act_before_optimize(self, container, module_name: Optional[str] = None, **kwargs):
        pass

    @property
    def lev_before_optimize(self):
        return 1


class AfterOptimizeActor(Actor):
    @abstractmethod
    def act_after_optimize(self, container, info, **kwargs):
        pass

    @property
    def lev_after_optimize(self):
        return 1


class OptimizerBuildActor(Actor):
    def __init__(self, module_name: Optional[str] = None):
        self._module_name = module_name

    @property
    def module_name(self):
        return self._module_name

    @abstractmethod
    def act_build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        pass

    @property
    def lev_build_optimizer(self, ):
        return 1


class BeforeRemoveInfoActor(Actor):
    @abstractmethod
    def act_before_removeinfo(self, container, info, **kwargs):
        pass

    @property
    def lev_before_removeinfo(self):
        return 1


class AfterRemoveInfoActor(Actor):
    @abstractmethod
    def act_after_removeinfo(self, container, info, **kwargs):
        pass

    @property
    def lev_after_removeinfo(self):
        return 1


_REGISTER_ACTOR_LEVEL = {
    InitialActor: 'lev_init',
    EndingActor: 'lev_ending',
    BeforeIterActor: 'lev_before_iter',
    AfterIterActor: 'lev_after_iter',
    BeforeEpochActor: 'lev_before_epoch',
    AfterEpochActor: 'lev_after_epoch',
    AfterSaveActor: 'lev_after_save',
    AfterLoadActor: 'lev_after_load',
    AtBroadcastActor: 'lev_broadcast',
    BeforeCycleActor: 'lev_before_cycle',
    AfterCycleActor: 'lev_after_cycle',
    BeforeAddInfoActor: 'lev_before_addinfo',
    BeforeRemoveInfoActor: 'lev_before_removeinfo',
    AfterAddInfoActor: 'lev_after_addinfo',
    AfterRemoveInfoActor: 'lev_after_removeinfo',
    OptimizerBuildActor: 'lev_build_optimizer',
    AfterBackwardActor: 'lev_after_backward',
    BeforeOptimizeActor: 'lev_before_optimize',
    AfterOptimizeActor: 'lev_after_optimize'
}


class ActorContainer(IActorContainer):

    def get_actors(self, actor_type: type) -> Sequence:
        return self._actors_dct[actor_type]

    def __init__(self):
        self._actors_dct = OrderedDict([(at, []) for at in _REGISTER_ACTOR_LEVEL.keys()])
        self._actors = []

    @property
    def actors(self) -> Sequence:
        return tuple(self._actors)

    def add_actor(self, actor):
        for actor_type, actors in self._actors_dct.items():
            if isinstance(actor, actor_type):
                actors.append(actor)
                lev_name = _REGISTER_ACTOR_LEVEL[actor_type]
                actors.sort(key=lambda at: getattr(at, lev_name))
        if isinstance(actor, Actor):
            actor.act_add(self)
        self._actors.append(actor)
        return self

    def extract_dct(self):
        seq = []
        for actor in self._actors:
            seq.append(actor.extract_dct())
        return seq

    def refrom_dct(self, seq: List):
        for actor, dct in zip(self.actors, seq):
            actor.refrom_dct(dct)
        return self


# </editor-fold>

# <editor-fold desc='输出管理'>


class BroadcastManager(IActorContainer):

    def broadcast(self, msg: str, **kwargs):
        for actor in self.get_actors(AtBroadcastActor):
            actor.act_broadcast(self, msg, **kwargs)
        return self

    def broadcast_dataframe(self, data: pd.DataFrame, **kwargs):
        msgs = dataframe2strs(data, inter_col='\t', divider=2)
        for msg in msgs:
            self.broadcast(msg, **kwargs)
        return self


class PrintBasedBroadcastActor(PrintBasedBroadcaster, AtBroadcastActor):

    def act_broadcast(self, container, msg, **kwargs):
        self.__call__(msg)


class LogBasedBroadcastActor(LogBasedBroadcaster, AtBroadcastActor):

    def act_broadcast(self, container, msg, **kwargs):
        self.__call__(msg)


# </editor-fold>

# <editor-fold desc='时间管理'>
def odct2dct(odct: OrderedDict) -> List:
    seq = []
    for key, val in odct.items():
        seq.append((key, val))
    return seq


def dct2odct(seq: Sequence) -> OrderedDict:
    odct = OrderedDict(seq)
    return odct


class TimeManager(IActorContainer):

    def __init__(self):
        self._time_dct = {}
        self._period_dct = OrderedDict()
        self._period_dct_auto = dict()
        self._time_inv = dict()

    def update_time(self, *names: str, time_cur: float = None):
        time_cur = time.time() if time_cur is None else time_cur
        for tn in names:
            self._time_dct[tn] = time_cur
        for tn in names:
            if tn in self._time_inv.keys():
                pn = self._time_inv[tn]
                per = self.collect_period(self._period_dct_auto[pn])
                if per >= 0:
                    self._period_dct[pn] = per
        return self

    def collect_period(self, period_pair: tuple) -> float:
        return self._time_dct.get(period_pair[1], 0) - self._time_dct.get(period_pair[0], 0)

    def regist_period_auto(self, name: str, period_pair: tuple):
        self._period_dct[name] = self.collect_period(period_pair)
        self._period_dct_auto[name] = period_pair
        self._time_inv[period_pair[1]] = name
        self._time_inv[period_pair[0]] = name
        return self

    def regist_periods_auto(self, names: Sequence[str], period_pairs: Sequence[tuple]):
        for name, period_pair in zip(names, period_pairs):
            self.regist_period_auto(name, period_pair)
        return self

    def update_period(self, name: str, period_pair: tuple):
        self._period_dct[name] = self.collect_period(period_pair)
        return self

    def update_periods(self, names: Sequence[str], period_pairs: Sequence[tuple]):
        for name, period_pair in zip(names, period_pairs):
            self.update_period(name, period_pair)
        return self

    def extract_dct(self):
        return {
            SVKEYS.TIME_DCT: self._time_dct,
            SVKEYS.TIME_INV: self._time_inv,
            SVKEYS.PERIOD_DCT: odct2dct(self._period_dct),
            SVKEYS.PERIOD_DCT_AUTO: self._period_dct_auto,
        }

    def refrom_dct(self, dct: Dict):
        self._time_dct = dct[SVKEYS.TIME_DCT]
        self._time_inv = dct[SVKEYS.TIME_INV]
        self._period_dct = dct2odct(dct[SVKEYS.PERIOD_DCT])
        self._period_dct_auto = dct[SVKEYS.PERIOD_DCT_AUTO]
        return self


# </editor-fold>


# <editor-fold desc='loss管理'>

class LossManager(IActorContainer):
    def __init__(self):
        self._loss_dct = OrderedDict()

    @staticmethod
    def check_losses(losses: Sequence[Union[torch.Tensor]], names: Sequence[str]) -> bool:
        for loss, name in zip(losses, names):
            if torch.isnan(loss):
                BROADCAST('nan in loss ' + str(name))
                raise Exception('err loss')
            if torch.isinf(loss):
                BROADCAST('inf in loss ' + str(name))
                raise Exception('err loss')
        return True

    @staticmethod
    def get_total_loss(loss: Union[dict, torch.Tensor]) -> float:
        if isinstance(loss, dict):
            total = 0.0
            for name_i, loss_i in loss.items():
                assert not torch.isnan(loss_i), 'nan occur in ' + name_i
                assert not torch.isinf(loss_i), 'inf occur in ' + name_i
                total += loss_i.item()
            return total
        elif isinstance(loss, torch.Tensor):
            assert not torch.isnan(loss), 'nan occur in Loss'
            assert not torch.isinf(loss), 'inf occur in Loss'
            return loss.item()
        else:
            raise Exception('err loss')

    @staticmethod
    def process_loss(loss: Union[dict, torch.Tensor]) \
            -> (torch.Tensor, List[str], List[torch.Tensor]):
        if isinstance(loss, dict):
            losses, names = [], []
            for name_i, loss_i in loss.items():
                assert not torch.isnan(loss_i), 'nan occur in ' + name_i
                assert not torch.isinf(loss_i), 'inf occur in ' + name_i
                losses.append(loss_i)
                names.append(name_i)
            loss = sum(losses)
            return loss, names, losses
        elif isinstance(loss, torch.Tensor):
            assert not torch.isnan(loss), 'nan occur in Loss'
            assert not torch.isinf(loss), 'inf occur in Loss'
            return loss, [], []
        else:
            raise Exception('err loss')

    def update_loss(self, name: str, loss: Union[torch.Tensor, float]):
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        self._loss_dct[name] = loss
        return self

    def update_losses(self, names: Sequence[str], losses: Sequence[Union[torch.Tensor, float]]):
        for name, loss in zip(names, losses):
            self.update_loss(name, loss)
        return self

    def extract_dct(self):
        return {
            SVKEYS.LOSS_DCT: odct2dct(self._loss_dct),
        }

    def refrom_dct(self, dct: Dict):
        self._loss_dct = dct2odct(dct[SVKEYS.LOSS_DCT])
        return self


# </editor-fold>

# <editor-fold desc='参数存储管理'>
class VariableManager(IActorContainer):
    def __init__(self):
        self._var_dct = OrderedDict()

    def add_var_scope(self, scope_name: str):
        if scope_name not in self._var_dct.keys():
            self._var_dct[scope_name] = OrderedDict()

    def update_var(self, scope_name: str, var_name: str, value):
        self._var_dct[scope_name][var_name] = value

    def extract_dct(self):
        return {
            SVKEYS.VAR_DCT: odct2dct(self._var_dct),
        }

    def refrom_dct(self, dct: Dict):
        self._var_dct = dct2odct(dct[SVKEYS.VAR_DCT])
        return self


# </editor-fold>

# <editor-fold desc='保存管理'>


class ModelInfoManager(IActorContainer):

    def __init__(self):
        self._infos = []

    @property
    def infos(self):
        return tuple(self._infos)

    def update_info(self, info):
        for actor in self.get_actors(BeforeAddInfoActor):
            actor.act_before_addinfo(self, info)
        self._infos.append(info)
        for actor in self.get_actors(AfterAddInfoActor):
            actor.act_after_addinfo(self, info)
        return self

    def _remove_info_index(self, index: int):
        for actor in self.get_actors(BeforeRemoveInfoActor):
            actor.act_before_removeinfo(self, self._infos[index])
        info = self._infos.pop(index)
        for actor in self.get_actors(AfterRemoveInfoActor):
            actor.act_after_addinfo(self, info)
        return self

    def remove_info(self, info):
        return self._remove_info_index(self._infos.index(info))

    def remove_info_fltr(self, fltr: Callable):
        inds_rmv = []
        for i, info in enumerate(self._infos):
            if fltr(info):
                inds_rmv.append(i)
        for i in reversed(inds_rmv):
            self._remove_info_index(i)
        return self

    def extract_dct(self):
        return {
            SVKEYS.INFOS: self._infos,
        }

    def refrom_dct(self, dct: Dict):
        self._infos = dct[SVKEYS.INFOS]
        return self


class SavePathCollector():

    def extract_dct(self):
        return {
            SVKEYS.SVAE_PTHS_PROPS: odct2dct(self._save_pths_props),
        }

    def refrom_dct(self, dct: Dict):
        self._save_pths_props = dct2odct(dct[SVKEYS.SVAE_PTHS_PROPS])
        return self

    def __init__(self, num_keep: int = 3, order_by: Optional[str] = None, order_ascend: bool = True):
        self._num_keep = num_keep
        self._save_pths_props = OrderedDict()
        self._order_by = order_by
        self._order_ascend = order_ascend

    @property
    def order_by(self) -> Optional[str]:
        return self._order_by

    @property
    def num_keep(self) -> int:
        return self._num_keep

    @property
    def save_pths_props(self) -> OrderedDict:
        return self._save_pths_props

    @property
    def save_pths(self) -> List[str]:
        return list(self._save_pths_props.keys())

    @property
    def props(self) -> List[dict]:
        return list(self._save_pths_props.values())

    def update_save_pth(self, save_pth: str, prop: Dict) -> (Optional[str], Optional[Dict]):
        prop_last = None
        if save_pth in self._save_pths_props.keys():
            prop_last = self._save_pths_props[save_pth]
        self._save_pths_props[save_pth] = prop
        if self.order_by is not None and len(self.order_by) > 0:
            self._save_pths_props = OrderedDict(sorted(
                self._save_pths_props.items(), key=lambda t: t[1].get(self.order_by, 0), reverse=self._order_ascend))
        if 0 < self.num_keep < len(self._save_pths_props):
            save_pth_last, prop_last = self._save_pths_props.popitem(last=True)
            return save_pth_last, prop_last
        else:
            return None, prop_last

    def remove_save_pth(self, save_pth: Optional[str]) -> Optional[dict]:
        return self._save_pths_props.pop(save_pth, None)


# </editor-fold>


# <editor-fold desc='工具原型'>

class LRScheduler(Actor):
    pass


class IMScheduler(Actor):
    pass

# </editor-fold>
