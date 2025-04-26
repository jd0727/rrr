from abc import abstractmethod

from ..define import *
from utils.functional import *
from ..interface import SettableImageSize, SettableSize

# <editor-fold desc='原型接口'>






# 可移动标签 自定义增广方法接口
class Expandable(metaclass=ABCMeta):

    @abstractmethod
    def expend_(self, scale: np.ndarray = SCALE_IDENTIIY):
        pass

    def expend(self, scale: np.ndarray = SCALE_IDENTIIY):
        return copy.deepcopy(self).expend_(scale)


class HasArea(metaclass=ABCMeta):

    @property
    @abstractmethod
    def area(self) -> float:
        pass


class Clipable(metaclass=ABCMeta):
    @abstractmethod
    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        pass

    def clip(self, xyxyN_rgn: np.ndarray, **kwargs):
        return copy.deepcopy(self).clip_(xyxyN_rgn, **kwargs)


class Movable(SettableSize):
    @abstractmethod
    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        pass

    def linear(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        return copy.deepcopy(self).linear_(size, biasN, scaleN, **kwargs)

    @abstractmethod
    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        pass

    def perspective(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        return copy.deepcopy(self).perspective_(size, homographyN, **kwargs)


class PointsExtractable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_xysN(self) -> int:
        pass

    @abstractmethod
    def extract_xysN(self) -> np.ndarray:
        pass

    @abstractmethod
    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        pass


class PointsExtractableBySLOTS(PointsExtractable):
    __slots__ = []

    @property
    def num_xysN(self):
        num_pnt = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, PointsExtractable):
                num_pnt += attr.num_xysN
        return num_pnt

    def extract_xysN(self):
        xysN = []
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, PointsExtractable):
                xysN.append(attr.extract_xysN())
        xysN = np.concatenate(xysN, axis=0) if len(xysN) > 0 else None
        return xysN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        ptr = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, PointsExtractable):
                dt = attr.num_xysN
                attr.refrom_xysN(xysN[ptr:ptr + dt], size, **kwargs)
                ptr = ptr + dt
        return self


class PointsExtractableByIterable(PointsExtractable, Iterable):

    @property
    def num_xysN(self):
        num_pnt = 0
        for name in self:
            attr = getattr(self, name)
            if isinstance(attr, PointsExtractable):
                num_pnt += attr.num_xysN
        return num_pnt

    def extract_xysN(self):
        xysN = []
        for name in self:
            attr = getattr(self, name)
            if isinstance(attr, PointsExtractable):
                xysN.append(attr.extract_xysN())
        xysN = np.concatenate(xysN, axis=0) if len(xysN) > 0 else None
        return xysN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        ptr = 0
        for name in self:
            attr = getattr(self, name)
            if isinstance(attr, PointsExtractable):
                dt = attr.num_xysN
                attr.refrom_xysN(xysN[ptr:ptr + dt], size, **kwargs)
                ptr = ptr + dt
        return self


class BoolMaskExtractable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_bool_chan(self) -> int:
        pass

    @abstractmethod
    def extract_maskNb(self) -> np.ndarray:
        pass

    @abstractmethod
    def refrom_maskNb(self, maskNb, **kwargs):
        pass

    def extract_maskNb_enc(self, index: int) -> np.ndarray:
        maskNb = self.extract_maskNb()
        inds = np.arange(index, index + maskNb.shape[-1])
        return np.max(maskNb * inds, axis=2, keepdims=True)

    def refrom_maskNb_enc(self, maskNb_enc: np.ndarray, index: int, **kwargs):
        inds = np.arange(index, index + self.num_bool_chan)
        maskNb = maskNb_enc == inds
        self.refrom_maskNb(maskNb, **kwargs)
        return self


class BoolMaskExtractableBySLOTS(BoolMaskExtractable):
    __slots__ = []

    @property
    def num_bool_chan(self):
        num_bool_chan = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, BoolMaskExtractable):
                num_bool_chan += attr.num_bool_chan
        return num_bool_chan

    def extract_maskNb(self) -> np.ndarray:
        maskNb = []
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, BoolMaskExtractable):
                maskNb.append(attr.extract_maskNb())
        maskNb = np.concatenate(maskNb, axis=-1) if len(maskNb) > 0 else None
        return maskNb

    def refrom_maskNb(self, maskNb: np.ndarray, **kwargs):
        ptr = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, BoolMaskExtractable):
                dt = attr.num_bool_chan
                attr.refrom_maskNb(maskNb[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self


class BoolMaskExtractableByIterable(BoolMaskExtractable, Iterable):

    @property
    def num_bool_chan(self):
        num_bool_chan = 0
        for item in self:
            if isinstance(item, BoolMaskExtractable):
                num_bool_chan += item.num_bool_chan
        return num_bool_chan

    def extract_maskNb(self) -> np.ndarray:
        maskNb = []
        for item in self:
            if isinstance(item, BoolMaskExtractable):
                maskNb.append(item.extract_maskNb())
        maskNb = np.concatenate(maskNb, axis=-1) if len(maskNb) > 0 else None
        return maskNb

    def refrom_maskNb(self, maskNb: np.ndarray, **kwargs):
        ptr = 0
        for item in self:
            if isinstance(item, BoolMaskExtractable):
                dt = item.num_bool_chan
                item.refrom_maskNb(maskNb[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self


class ValMaskExtractable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def num_chan(self) -> int:
        pass

    @abstractmethod
    def extract_maskN(self) -> np.ndarray:
        pass

    @abstractmethod
    def refrom_maskN(self, maskN: np.ndarray, **kwargs):
        pass


class ValMaskExtractableBySLOTS(ValMaskExtractable):
    __slots__ = []

    @property
    def num_chan(self):
        num_chan = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                num_chan += attr.num_chan
        return num_chan

    def extract_maskN(self) -> np.ndarray:
        maskN = []
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                maskN.append(attr.extract_maskN())
        maskN = np.concatenate(maskN, axis=-1) if len(maskN) > 0 else None
        return maskN

    def refrom_maskN(self, maskN: np.ndarray, **kwargs):
        ptr = 0
        for name in self.__slots__:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                dt = attr.num_chan
                attr.refrom_maskN(maskN[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self


class ValMaskExtractableByIterable(ValMaskExtractable, Iterable):

    @property
    def num_chan(self):
        num_chan = 0
        for name in self:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                num_chan += attr.num_chan
        return num_chan

    def extract_maskN(self) -> np.ndarray:
        maskN = []
        for name in self:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                maskN.append(attr.extract_maskN())
        maskN = np.concatenate(maskN, axis=-1) if len(maskN) > 0 else None
        return maskN

    def refrom_maskN(self, maskN: np.ndarray, **kwargs):
        ptr = 0
        for name in self:
            attr = getattr(self, name)
            if isinstance(attr, ValMaskExtractable):
                dt = attr.num_chan
                attr.refrom_maskN(maskN[..., ptr:ptr + dt], **kwargs)
                ptr = ptr + dt
        return self


# 作为图像标签可被其余增广包扩展
class Augmentable(BoolMaskExtractable, PointsExtractable, ValMaskExtractable, SettableImageSize):
    pass


class Measurable(metaclass=ABCMeta):
    @property
    @abstractmethod
    def measure(self) -> float:
        pass


class AspectMeasurable(Measurable):

    @property
    @abstractmethod
    def measure_min(self) -> float:
        pass

    @property
    @abstractmethod
    def measure_max(self) -> float:
        pass

    @property
    def aspect(self) -> float:
        return self.measure_max / max(self.measure_min, 1e-7)


class HasXYXYN(metaclass=ABCMeta):

    @property
    @abstractmethod
    def xyxyN(self) -> np.ndarray:
        pass


class HasXYXYSN(metaclass=ABCMeta):

    @property
    @abstractmethod
    def xyxysN(self) -> np.ndarray:
        pass


class HasWHN(metaclass=ABCMeta):

    @property
    @abstractmethod
    def whN(self) -> np.ndarray:
        pass


class HasMaskNbAbs(metaclass=ABCMeta):

    @property
    @abstractmethod
    def maskNb(self) -> np.ndarray:
        pass


class HasMaskNAbs(metaclass=ABCMeta):

    @property
    @abstractmethod
    def maskN(self) -> np.ndarray:
        pass


class HasXYSNI(metaclass=ABCMeta):

    @property
    @abstractmethod
    def xysNi(self) -> np.ndarray:
        pass


class AspectMeasurableFromXYSNI(AspectMeasurable, HasXYSNI):

    @property
    def measure(self):
        return self.xysNi.shape[0]

    @property
    def whN_proj(self):
        xys = self.xysNi
        mat = xysN2rot2N(xys)
        xys_proj = xys @ mat
        return np.max(xys_proj, axis=0) - np.min(xys_proj, axis=0)

    @property
    def measure_min(self):
        return np.min(self.whN_proj)

    @property
    def measure_max(self):
        return np.max(self.whN_proj)

    @property
    def aspect(self):
        wh = self.whN_proj
        return np.max(wh) / max(np.min(wh), 1e-8)


# </editor-fold>
class ColoredVertex():
    def __init__(self, vcolorsN: Optional[np.ndarray] = None):
        self._vcolorsN = np.array(vcolorsN).astype(np.float32) if vcolorsN is not None else None

    @property
    def vcolorsN(self) -> np.ndarray:
        return self._vcolorsN
