from .define import *


# <editor-fold desc='矢量曲线'>

class ScalableFunc():
    def __init__(self, num_base: int = 10, scale: int = 1):
        self.num_base = num_base
        self.scale = scale

    @property
    def scale(self) -> int:
        return self._scale

    @scale.setter
    def scale(self, scale: int) -> NoReturn:
        self._scale = scale

    @property
    def num_base(self) -> int:
        return self._num_base

    @num_base.setter
    def num_base(self, num_base: int) -> NoReturn:
        self._num_base = num_base

    @property
    def num_scaled(self) -> int:
        return self.num_base * self.scale

    @abstractmethod
    def __getitem__(self, ind_scaled: int):
        pass


class ScalableSize(ScalableFunc):
    pass


class ConstSize(ScalableSize):
    def __init__(self, size: Tuple[int, int], num_base: int = 10, scale: int = 1):
        ScalableSize.__init__(self, num_base=num_base, scale=scale)
        self.size = size

    def __getitem__(self, ind_scaled: int) -> Tuple[int, int]:
        return self.size


class RandSize(ScalableSize):

    def __init__(self, min_size: Tuple[int, int], max_size: Tuple[int, int], devisor: int = 32,
                 keep_ratio: bool = True, num_base_keep: int = 1, max_first: bool = True,
                 max_last: bool = True, num_base: int = 10, scale: int = 1):
        super().__init__(num_base=num_base, scale=scale)
        self.min_size = min_size
        self.max_size = max_size
        self.devisor = devisor
        self.keep_ratio = keep_ratio
        self.num_base_keep = num_base_keep
        self.max_first = max_first
        self.max_last = max_last

        self.max_w, self.max_h = int(math.floor(max_size[0] / devisor)), int(math.floor(max_size[1] / devisor))
        self.min_w, self.min_h = int(math.ceil(min_size[0] / devisor)), int(math.ceil(min_size[1] / devisor))
        self._last_size = self._rand_size()
        self._keeped = 0

    @property
    def num_scaled_keep(self) -> int:
        return self.num_base_keep * self.scale

    def _rand_size(self) -> Tuple[int, int]:
        w = random.randint(self.min_w, self.max_w)
        if self.keep_ratio:
            h = int(1.0 * (w - self.min_w) / (self.max_w - self.min_w) * (self.max_h - self.min_h) + self.min_h)
        else:
            h = random.randint(self.min_h, self.max_h)
        return (w * self.devisor, h * self.devisor)

    def __getitem__(self, ind_scaled: int) -> Tuple[int, int]:
        if (self.max_first and ind_scaled <= 0) \
                or (self.max_last and ind_scaled >= self.num_scaled - self.num_scaled_keep):
            size = self.max_size
        elif self._keeped < self.num_scaled_keep:
            size = self._last_size
        else:
            size = self._rand_size()
            self._keeped = 0
        self._keeped = self._keeped + 1
        self._last_size = size
        return size


class ScalableCurve(ScalableFunc):
    @property
    def vals(self) -> list:
        vals = []
        for i in range(self.num_scaled):
            vals.append(self.__getitem__(i))
        return vals

    def __imul__(self, other):
        return self


class ComposedCurve(ScalableCurve):

    @property
    def num_base(self):
        return sum([curve.num_base for curve in self.curves])

    @property
    def scale(self) -> int:
        return self.num_scaled // max(self.num_base, 1)

    @scale.setter
    def scale(self, scale: int) -> NoReturn:
        num_scaled = 0
        milestones = []
        for curve in self.curves:
            curve.scale = scale
            milestones.append(num_scaled)
            num_scaled += curve.num_scaled
        self._milestones = milestones

    @property
    def num_scaled(self) -> int:
        return sum([curve.num_scaled for curve in self.curves])

    def __init__(self, *curves: ScalableCurve, scale: int = 1):
        self.curves = curves
        ScalableCurve.__init__(self, num_base=0, scale=scale)

    def __imul__(self, other):
        for curve in self.curves:
            curve.__imul__(other)
        return self

    def __getitem__(self, ind_scaled: int) -> float:
        for i in range(len(self._milestones) - 1, -1, -1):
            if ind_scaled >= self._milestones[i]:
                return self.curves[i].__getitem__(ind_scaled - self._milestones[i])
        raise Exception('milestones err')


class MultiStepCurve(ScalableCurve):

    def __init__(self, value_init: float = 0.1, base_milestones: Tuple[int, ...] = (0, 1),
                 gamma: float = 0.1, num_base: int = 10, scale: int = 1):
        if isinstance(base_milestones, int):
            base_milestones = [base_milestones]
        self.base_milestones = list(base_milestones)
        self.gamma = gamma
        self.value_init = value_init
        ScalableCurve.__init__(self, num_base=num_base, scale=scale)

    def __imul__(self, other: float):
        self.value_init *= other
        return self

    def __getitem__(self, ind_scaled: int) -> float:
        lr = self.value_init
        for milestone in self.smilestones:
            lr = lr * self.gamma if ind_scaled >= milestone else lr
        return lr

    @property
    def scale(self) -> int:
        return self._scale

    @scale.setter
    def scale(self, scale: int) -> NoReturn:
        self._scale = scale
        smilestones = []
        for i in range(len(self.base_milestones)):
            smilestones.append(self.base_milestones[i] * scale)
        self.smilestones = smilestones


class PowerCurve(ScalableCurve):
    def __init__(self, value_init: float = 0.1, value_end: float = 1e-8, num_base: int = 10,
                 scale: int = 1, pow: int = 2):
        super(PowerCurve, self).__init__(num_base=num_base, scale=scale)
        self.value_init = value_init
        self.value_end = value_end
        self.pow = pow

    def __getitem__(self, ind_scaled: int) -> float:
        alpha = (ind_scaled / self.num_scaled) ** self.pow
        val = (1 - alpha) * self.value_init + alpha * self.value_end
        return val

    def __imul__(self, other: float):
        self.value_init *= other
        self.value_end *= other
        return self


class ExponentialCurve(ScalableCurve):
    def __init__(self, value_init: float = 0.1, value_end: float = 1e-8, num_base: int = 10, scale: int = 1):
        super(ExponentialCurve, self).__init__(num_base=num_base, scale=scale)
        self.value_init_log = math.log(value_init)
        self.value_end_log = math.log(value_end)

    def __getitem__(self, ind_scaled: int) -> float:
        alpha = ind_scaled / self.num_scaled
        val_log = (1 - alpha) * self.value_init_log + alpha * self.value_end_log
        return math.exp(val_log)

    def __imul__(self, other: float):
        self.value_init_log += math.log(other)
        self.value_end_log += math.log(other)
        return self


class CosCurve(ScalableCurve):

    def __init__(self, val_init: float = 0.1, val_end: float = 1e-8, num_base: int = 10, scale: int = 1):
        super(CosCurve, self).__init__(num_base=num_base, scale=scale)
        self.val_init = val_init
        self.val_end = val_end

    def __getitem__(self, ind_scaled: int) -> float:
        alpha = ind_scaled / self.num_scaled
        lr = self.val_end + (self.val_init - self.val_end) * 0.5 * (1.0 + math.cos(math.pi * alpha))
        return lr

    def __imul__(self, other: float):
        self.val_init *= other
        self.val_end *= other
        return self


class ConstCurve(ScalableCurve):
    def __init__(self, value: float = 0.1, num_base: int = 10, scale: int = 1):
        super(ConstCurve, self).__init__(num_base=num_base, scale=scale)
        self.value = value

    def __getitem__(self, ind_scaled: int):
        return self.value

    def __imul__(self, other: float):
        self.value *= other
        return self


# </editor-fold>

# <editor-fold desc='学习率'>
class CurveBasedLRScheduler(LRScheduler):

    def __init__(self, curve: ScalableCurve, module_name: Optional[str] = None, group_index: Optional[int] = None):
        self.curve = curve
        self.module_name = module_name
        self.group_index = group_index


class EpochBasedLRScheduler(CurveBasedLRScheduler, BeforeEpochActor):

    def act_add(self, trainer, **kwargs):
        self.curve.scale = 1
        trainer.total_epoch = self.curve.num_base

    def act_before_epoch(self, trainer, **kwargs):
        learning_rate = self.curve[trainer.ind_epoch]
        trainer.optimizer_lr_set(learning_rate, module_name=self.module_name, group_index=self.group_index)

    @staticmethod
    def Const(lr: float = 0.1, num_epoch: int = 10, module_name: Optional[str] = None,
              group_index: Optional[int] = None):
        return EpochBasedLRScheduler(ConstCurve(value=lr, num_base=num_epoch),
                                     module_name=module_name, group_index=group_index)

    @staticmethod
    def Cos(lr_init: float = 0.1, lr_end: float = 1e-8, num_epoch=10, module_name: Optional[str] = None,
            group_index: Optional[int] = None):
        return EpochBasedLRScheduler(CosCurve(val_init=lr_init, val_end=lr_end, num_base=num_epoch),
                                     module_name=module_name, group_index=group_index)

    @staticmethod
    def WarmCos(lr_init: float = 0.1, lr_end: float = 1e-8, num_epoch: int = 10, num_warm: int = 1,
                module_name: Optional[str] = None, group_index: Optional[int] = None):
        curve = ComposedCurve(
            PowerCurve(value_init=0, value_end=lr_init, num_base=num_warm),
            CosCurve(val_init=lr_init, val_end=lr_end, num_base=max(0, num_epoch - num_warm))
        )
        return EpochBasedLRScheduler(curve, module_name=module_name, group_index=group_index)

    @staticmethod
    def MultiStep(lr_init: float = 0.1, milestones: tuple[int, ...] = (0, 1), gamma: float = 0.1,
                  num_epoch: int = 10, module_name: Optional[str] = None, group_index: Optional[int] = None):
        return EpochBasedLRScheduler(MultiStepCurve(
            value_init=lr_init, base_milestones=milestones, gamma=gamma, num_base=num_epoch), module_name, group_index)


class EpochBasedConsecutiveLRScheduler(CurveBasedLRScheduler, BeforeIterActor):

    def act_add(self, trainer, **kwargs):
        self.curve.scale = len(trainer.loader)
        trainer.total_epoch = self.curve.num_base

    def act_before_iter(self, trainer, **kwargs):
        learning_rate = self.curve[trainer.ind_iter]
        trainer.optimizer_lr_set(learning_rate, name=self.module_name, group_index=self.group_index)

    @staticmethod
    def Const(lr: float = 0.1, num_epoch=10, module_name: Optional[str] = None, group_index: Optional[int] = None):
        return EpochBasedLRScheduler(ConstCurve(value=lr, num_base=num_epoch), module_name, group_index)

    @staticmethod
    def Cos(lr_init: float = 0.1, lr_end: float = 1e-8, num_epoch: int = 10,
            module_name: Optional[str] = None, group_index: Optional[int] = None):
        return EpochBasedLRScheduler(CosCurve(val_init=lr_init, val_end=lr_end, num_base=num_epoch), module_name,
                                     group_index)

    @staticmethod
    def WarmCos(lr_init: float = 0.1, lr_end: float = 1e-8, num_epoch: int = 10, num_warm: int = 1,
                module_name: Optional[str] = None, group_index: Optional[int] = None):
        curve = ComposedCurve(
            PowerCurve(value_init=0, value_end=lr_init, num_base=num_warm),
            CosCurve(val_init=lr_init, val_end=lr_end, num_base=max(0, num_epoch - num_warm))
        )
        return EpochBasedConsecutiveLRScheduler(curve, module_name, group_index)

    @staticmethod
    def MultiStep(lr_init: float = 0.1, milestones: Tuple[int, ...] = (0, 1), gamma: float = 0.1, num_epoch: int = 10,
                  module_name: Optional[str] = None, group_index: Optional[int] = None):
        return EpochBasedLRScheduler(MultiStepCurve(
            value_init=lr_init, base_milestones=milestones, gamma=gamma, num_base=num_epoch), module_name, group_index)


class IterBasedLRScheduler(CurveBasedLRScheduler, BeforeIterActor):

    def act_add(self, trainer, **kwargs):
        self.curve.scale = 1
        trainer.total_iter = self.curve.num_base

    def act_before_iter(self, trainer, **kwargs):
        learning_rate = self.curve[trainer.ind_iter]
        trainer.optimizer_lr_set(learning_rate, module_name=self.module_name, group_index=self.group_index)

    @staticmethod
    def Const(lr: float = 0.1, num_iter: int = 10, module_name: Optional[str] = None,
              group_index: Optional[int] = None):
        return IterBasedLRScheduler(ConstCurve(value=lr, num_base=num_iter), module_name, group_index)

    @staticmethod
    def Cos(lr_init: float = 0.1, lr_end: float = 1e-8, num_iter: int = 10,
            module_name: Optional[str] = None, group_index: Optional[int] = None):
        return IterBasedLRScheduler(CosCurve(val_init=lr_init, val_end=lr_end, num_base=num_iter), module_name,
                                    group_index)

    @staticmethod
    def MultiStep(lr_init: float = 0.1, milestones: Tuple[int, ...] = (0, 1), gamma: float = 0.1, num_iter: int = 10,
                  module_name: Optional[str] = None, group_index: Optional[int] = None):
        return IterBasedLRScheduler(MultiStepCurve(
            value_init=lr_init, base_milestones=milestones, gamma=gamma, num_base=num_iter), module_name, group_index)


class FuncBasedMScheduler(IMScheduler):

    def __init__(self, scsize: ScalableSize):
        self.scsize = scsize


class EpochBasedIMScheduler(FuncBasedMScheduler, BeforeEpochActor):

    def act_add(self, trainer, **kwargs):
        self.scsize.scale = 1
        trainer.total_epoch = self.scsize.num_base

    def act_before_epoch(self, trainer, **kwargs):
        trainer.img_size = self.scsize[trainer.ind_epoch]

    @staticmethod
    def Const(img_size: Tuple[int, int] = (32, 32), num_epoch: int = 10):
        return EpochBasedIMScheduler(ConstSize(size=img_size, num_base=num_epoch, scale=1))

    @staticmethod
    def Rand(min_size: Tuple[int, int], max_size: Tuple[int, int], devisor: int = 32, keep_ratio: bool = True,
             num_keep: int = 1, max_first: bool = True, max_last: bool = True, num_epoch: int = 10):
        return EpochBasedIMScheduler(RandSize(
            min_size=min_size, max_size=max_size, devisor=devisor, keep_ratio=keep_ratio,
            num_base_keep=num_keep, max_first=max_first, max_last=max_last,
            num_base=num_epoch, scale=1))


class IterBasedIMScheduler(FuncBasedMScheduler, BeforeIterActor):

    def act_add(self, trainer, **kwargs):
        self.scsize.scale = 1
        trainer.total_iter = self.scsize.num_base

    def act_before_iter(self, trainer, **kwargs):
        trainer.img_size = self.scsize[trainer.ind_iter]

    @staticmethod
    def Const(img_size: Tuple[int, int] = (32, 32), num_iter: int = 10):
        return IterBasedIMScheduler(ConstSize(size=img_size, num_base=num_iter, scale=1))

    @staticmethod
    def Rand(min_size: Tuple[int, int], max_size: Tuple[int, int], devisor: int = 32,
             keep_ratio: bool = True, num_keep: int = 1, max_first: bool = True, max_last: bool = True,
             num_iter: int = 10):
        return IterBasedIMScheduler(RandSize(
            min_size=min_size, max_size=max_size, devisor=devisor, keep_ratio=keep_ratio,
            num_base_keep=num_keep, max_first=max_first, max_last=max_last,
            num_base=num_iter, scale=1))

# </editor-fold>


# if __name__ == '__main__':
#     from visual import *
#
#     curve = ScalableCurve.WARM_COS(val_init=0.0025, warm_epoch=1, num_biter=300)
#     # curve = ScalableCurve.WARM_STEP(val_init=0.1, warm_epoch=50, milestones=(20, 30), gamma=0.1, num_biter=100)
#     curve.scale = 100
#     plt.plot(curve.lr_list)
