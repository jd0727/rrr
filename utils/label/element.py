from .define import *
from ..interface import HasNumClass


# <editor-fold desc='分类标签'>

def cindN2chotN(cindN: np.ndarray, num_cls: int, conf: float = 1) -> np.ndarray:
    chotN = np.zeros(shape=num_cls)
    chotN[cindN] = conf
    return chotN


def chotN2cindN(chotN: np.ndarray) -> np.ndarray:
    return np.argmax(chotN, axis=-1)


class HasConfidence(metaclass=ABCMeta):
    @property
    @abstractmethod
    def confN(self) -> np.ndarray:
        pass


class HasClassIndex(metaclass=ABCMeta):
    @property
    @abstractmethod
    def cindN(self) -> np.ndarray:
        pass


class Category(Convertable, HasNumClass, HasConfidence, HasClassIndex):
    REGISTER_COVERT = Register()

    @abstractmethod
    def conf_scale_(self, scale: float):
        pass

    @abstractmethod
    def conf_scale(self, scale: float):
        pass

    @property
    def cind(self):
        return int(self.cindN)


class IndexCategory(Category, Convertable):
    REGISTER_COVERT = Register()
    __slots__ = ('_cindN', '_confN', '_num_cls')

    def __init__(self, cindN: Union[np.ndarray, int], num_cls: int, confN: Union[np.ndarray, float] = 1.0):
        self._num_cls = num_cls
        self._cindN = np.array(cindN).astype(np.int32)
        self._confN = np.array(confN).astype(float)

    def __repr__(self):
        return '<' + str(self._cindN) + '>'

    def __eq__(self, other):
        other = IndexCategory.convert(other)
        return self._cindN == other._cindN and self.confN == other.confN

    @property
    def num_cls(self) -> int:
        return self._num_cls

    @property
    def confN(self):
        return self._confN

    @property
    def cindN(self):
        return self._cindN



    @confN.setter
    def confN(self, confN: np.ndarray):
        self._confN = confN

    def conf_scale_(self, scale):
        self._confN *= scale

    def conf_scale(self, scale: float):
        return IndexCategory(self._cindN, self._num_cls, self.confN * scale)


class OneHotCategory(Category, Convertable):
    @property
    def cindN(self) -> int:
        return np.argmax(self._chotN)

    @property
    def num_cls(self) -> int:
        return len(self._chotN)

    REGISTER_COVERT = Register()
    __slots__ = ('_chotN',)

    def __init__(self, chotN: np.ndarray):
        self._chotN = np.array(chotN)

    def conf_scale_(self, scale):
        self._chotN *= scale

    def conf_scale(self, scale: float):
        return OneHotCategory(self._chotN * scale)

    def cvt_cindcates(self, ntop_cls: int = 1):
        cinds = np.argsort(-self._chotN, )[:ntop_cls]
        cates = [IndexCategory(cindN=cind, confN=self._chotN[cind], num_cls=self.num_cls) for cind in cinds]
        return cates

    @property
    def confN(self):
        return np.max(self._chotN)

    @property
    def chotN(self) -> np.ndarray:
        return self._chotN

    def __repr__(self):
        return '<' + str(np.argmax(self._chotN)) + '>'

    def __eq__(self, other):
        other = OneHotCategory.convert(other)
        return np.all(self._chotN == other._chotN)


@IndexCategory.REGISTER_COVERT.registry(int)
def _int2index_category(category: int):
    return IndexCategory(cindN=category, num_cls=category + 1, confN=1.0)


@IndexCategory.REGISTER_COVERT.registry(OneHotCategory)
def _onehot_category2index_category(category: OneHotCategory):
    return IndexCategory(cindN=chotN2cindN(category._chotN), num_cls=category.num_cls,
                         confN=np.max(category._chotN))


@OneHotCategory.REGISTER_COVERT.registry(IndexCategory)
def _index_category2onehot_category(category: IndexCategory):
    return OneHotCategory(chotN=cindN2chotN(category._cindN, num_cls=category._num_cls))


@OneHotCategory.REGISTER_COVERT.registry(List, Tuple, np.ndarray)
def _list_tuple_arr2onehot_category(category: Union[List, Tuple, np.ndarray]):
    return OneHotCategory(chotN=category)


# </editor-fold>


# <editor-fold desc='几何元素'>
class XYSSurface(Movable, AspectMeasurable, HasXYXYN, PointsExtractable, Convertable):

    @property
    def xysN(self) -> np.ndarray:
        return self._xysN

    @property
    def surfsN(self) -> np.ndarray:
        return self._surfsN

    @property
    def edgesN(self) -> np.ndarray:
        surf_roll = np.roll(self._surfsN, shift=1, axis=-1)
        edges = np.stack([self._surfsN, surf_roll], axis=-1)
        edges = np.sort(edges, axis=-1)
        edges = np.unique(edges.reshape(-1, 2), axis=0)
        return edges

    @property
    def xyxyN(self) -> np.ndarray:
        return xysN2xyxyN(self._xysN)

    @property
    def measure(self):
        return np.sqrt(np.prod(np.max(self._xysN, axis=0) - np.min(self._xysN, axis=0)))

    @property
    def measure_min(self):
        return np.min(np.max(self._xysN, axis=0) - np.min(self._xysN, axis=0))

    @property
    def measure_max(self):
        return np.max(np.max(self._xysN, axis=0) - np.min(self._xysN, axis=0))

    REGISTER_COVERT = Register()
    __slots__ = ('_xysN', '_surfsN', '_size')

    def __init__(self, xysN: np.ndarray, surfsN: np.ndarray, size: tuple):
        self._xysN = np.array(xysN).astype(np.float32)
        self._surfsN = np.array(surfsN).astype(np.int32)
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self._xysN = xysN_clip(self._xysN, xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xysN' + str(self._xysN)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self._xysN = self._xysN * scaleN + biasN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xysN = xysN_perspective(self._xysN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def __eq__(self, other):
        other = XYSGraph.convert(other)
        return np.all(self._xysN == other._xysN) and np.all(self._surfsN == other._surfsN)

    @property
    def num_xysN(self):
        return self._xysN.shape[0]

    def extract_xysN(self):
        return self._xysN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xysN = xysN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self


class XYSColoredSurface(XYSSurface, ColoredVertex):
    def __init__(self, xysN: np.ndarray, surfsN: np.ndarray, size: tuple, vcolorsN: Optional[np.ndarray] = None):
        XYSSurface.__init__(self, xysN=xysN, surfsN=surfsN, size=size)
        ColoredVertex.__init__(self, vcolorsN=vcolorsN)


class XYSGraph(Movable, AspectMeasurable, HasXYXYN, PointsExtractable, Convertable):
    @property
    def xyxyN(self) -> np.ndarray:
        return xysN2xyxyN(self._xysN)

    @property
    def xysN(self) -> np.ndarray:
        return self._xysN

    @property
    def edgesN(self) -> np.ndarray:
        return self._edgesN

    @property
    def measure(self):
        return np.sqrt(np.prod(np.max(self._xysN, axis=0) - np.min(self._xysN, axis=0)))

    @property
    def measure_min(self):
        return np.min(np.max(self._xysN, axis=0) - np.min(self._xysN, axis=0))

    @property
    def measure_max(self):
        return np.max(np.max(self._xysN, axis=0) - np.min(self._xysN, axis=0))

    REGISTER_COVERT = Register()
    __slots__ = ('_xysN', '_edgesN', '_size')

    def __init__(self, xysN: np.ndarray, edgesN: np.ndarray, size: tuple):
        self._xysN = np.array(xysN).astype(np.float32)
        self._edgesN = np.array(edgesN).astype(np.int32)
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self._xysN = xysN_clip(self._xysN, xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xysN' + str(self._xysN)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self._xysN = self._xysN * scaleN + biasN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xysN = xysN_perspective(self._xysN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def __eq__(self, other):
        other = XYSGraph.convert(other)
        return np.all(self._xysN == other._xysN) and np.all(self._edgesN == other._surfsN)

    @property
    def num_xysN(self):
        return self._xysN.shape[0]

    def extract_xysN(self):
        return self._xysN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xysN = xysN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self


class XYSPoint(Movable, AspectMeasurable, Clipable, HasXYXYN, PointsExtractable, Convertable):
    @property
    def xyxyN(self) -> np.ndarray:
        return xysN2xyxyN(self._xysN)

    @property
    def xysN(self) -> np.ndarray:
        return self._xysN

    @property
    def measure(self):
        if self.xysN.shape[0] == 0:
            return 0.0
        else:
            return np.sqrt(np.prod(np.max(self.xysN, axis=0) - np.min(self.xysN, axis=0)))

    @property
    def measure_min(self):
        if self.xysN.shape[0] == 0:
            return 0.0
        else:
            return np.min(np.max(self.xysN, axis=0) - np.min(self.xysN, axis=0))

    @property
    def measure_max(self):
        if self.xysN.shape[0] == 0:
            return 0.0
        else:
            return np.max(np.max(self.xysN, axis=0) - np.min(self.xysN, axis=0))

    REGISTER_COVERT = Register()
    __slots__ = ('_xysN', '_size')

    def __init__(self, xysN: np.ndarray, size: tuple):
        self._xysN = np.array(xysN).astype(np.float32)
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        valid = np.all((self._xysN > xyxyN_rgn[:2]) * (self._xysN < xyxyN_rgn[2:]), axis=-1)
        self._xysN = self._xysN[valid]
        return self

    def __repr__(self):
        return 'xysN' + str(self.xysN)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self._xysN = self._xysN * scaleN + biasN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xysN = xysN_perspective(self._xysN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def __eq__(self, other):
        other = XYSPoint.convert(other)
        return np.allclose(self._xysN, other._xysN)

    @property
    def num_xysN(self):
        return self.xysN.shape[0]

    def extract_xysN(self):
        return self.xysN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xysN = xysN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self


class XYXYBorder(Movable, Clipable, AspectMeasurable, HasArea, HasMaskNbAbs, HasXYXYN,
                 HasXYSNI, HasWHN, Expandable, PointsExtractable, Convertable):
    @property
    def xyxyN(self) -> np.ndarray:
        return self._xyxyN

    REGISTER_COVERT = Register()

    @property
    def xysNi(self) -> np.ndarray:
        return xyxyN2xysNi(self._xyxyN, self._size)

    @property
    def measure(self) -> float:
        return np.sqrt(np.prod(self.whN))

    @property
    def whN(self):
        return np.maximum(self._xyxyN[2:4] - self._xyxyN[:2], 0)

    @property
    def measure_min(self) -> float:
        return np.min(self.whN)

    @property
    def measure_max(self) -> float:
        return np.max(self.whN)

    @property
    def aspect(self) -> float:
        wh = self.whN
        return np.max(wh) / np.clip(np.min(wh), a_min=1e-7, a_max=None)

    @property
    def area(self) -> float:
        return np.prod(self.whN)

    WIDTH = 4
    __slots__ = ('_xyxyN', '_size')

    def __init__(self, xyxyN, size):
        self._xyxyN = np.array(xyxyN).astype(np.float32)
        self._size = tuple(size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self._xyxyN = xyxyN_clip(self._xyxyN, xyxyN_rgn=xyxyN_rgn)
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        xyp = xyxyN2xypN(self._xyxyN)
        xyp = xyp * scaleN + biasN
        self._xyxyN = xysN2xyxyN(xyp)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xyxyN = xyxyN_perspective(self._xyxyN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend_(self, scale: np.ndarray = SCALE_IDENTIIY):
        xywh = xyxyN2xywhN(self._xyxyN)
        xywh[2:4] *= scale
        self._xyxyN = xywhN2xyxyN(xywh)
        return self

    def __repr__(self):
        return 'xyxyN' + str(self._xyxyN)

    @property
    def num_xysN(self):
        return NUM_WH2XYS_SAMP

    def extract_xysN(self):
        return xyxyN2xysN_samp(self._xyxyN)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xyxyN = xysN2xyxyN(xysN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        maskN = xyxyN2maskNb(self._xyxyN, self._size)
        return maskN

    def __eq__(self, other):
        return np.all(self._xyxyN == XYXYBorder.convert(other)._xyxyN)


class XYWHBorder(Movable, Clipable, AspectMeasurable, HasArea, HasMaskNbAbs, HasXYXYN,
                 HasXYSNI, HasWHN, Expandable, PointsExtractable, Convertable):
    @property
    def xyxyN(self) -> np.ndarray:
        return xywhN2xyxyN(self._xywhN)

    @property
    def xywhN(self) -> np.ndarray:
        return self._xywhN

    @property
    def whN(self) -> np.ndarray:
        return self._xywhN[2:]

    REGISTER_COVERT = Register()

    WIDTH = 4
    __slots__ = ('_xywhN', '_size')

    def __init__(self, xywhN, size):
        self._xywhN = np.array(xywhN).astype(np.float32)
        self._size = tuple(size)

    @property
    def xysNi(self) -> np.ndarray:
        return xywhN2xysNi(self._xywhN, self._size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    @property
    def measure(self):
        return np.sqrt(np.prod(np.maximum(self._xywhN[2:4], 0)))

    @property
    def measure_min(self):
        return np.min(np.maximum(self._xywhN[2:4], 0))

    @property
    def measure_max(self):
        return np.max(np.maximum(self._xywhN[2:4], 0))

    @property
    def area(self):
        return np.prod(np.maximum(self._xywhN[2:4], 0))

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self._xywhN = xywhN_clip(self._xywhN, xyxyN_rgn=xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xywhN' + str(self._xywhN)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        xyp = xywhN2xypN(self._xywhN)
        xyp = xyp * scaleN + biasN
        self._xywhN = xysN2xywhN(xyp)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xywhN = xywhN_perspective(self._xywhN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend_(self, scale: np.ndarray = SCALE_IDENTIIY):
        self._xywhN[2:4] *= scale
        return self

    def __eq__(self, other):
        return np.all(self._xywhN == XYWHBorder.convert(other)._xywhN)

    @property
    def num_xysN(self):
        return NUM_WH2XYS_SAMP

    def extract_xysN(self):
        return xywhN2xysN_samp(self._xywhN)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xywhN = xysN2xywhN(xysN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        maskN = xywhN2maskNb(self._xywhN, self._size)
        return maskN


class XYWHABorder(Movable, Clipable, AspectMeasurable, HasArea, HasMaskNbAbs, HasXYXYN,
                  HasXYSNI, HasWHN, Expandable, PointsExtractable, Convertable):
    REGISTER_COVERT = Register()

    @property
    def xyxyN(self) -> np.ndarray:
        return xysN2xyxyN(xywhaN2xypN(self._xywhaN))

    @property
    def xysNi(self) -> np.ndarray:
        return maskNb2xysNi(self.maskNb)

    @property
    def xywhaN(self) -> np.ndarray:
        return self._xywhaN

    @property
    def whN(self) -> np.ndarray:
        xyxy = self.xyxyN
        return np.maximum(xyxy[2:4] - xyxy[:2], 0)

    WIDTH = 5
    __slots__ = ('_xywhaN', '_size')

    def __init__(self, xywhaN: np.ndarray, size: tuple):
        self._xywhaN = np.array(xywhaN).astype(np.float32)
        self._size = tuple(size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    @property
    def measure(self):
        return np.sqrt(np.prod(np.maximum(self._xywhaN[2:4], 0)))

    @property
    def measure_min(self):
        return np.min(np.maximum(self._xywhaN[2:4], 0))

    @property
    def measure_max(self):
        return np.max(np.maximum(self._xywhaN[2:4], 0))

    @property
    def area(self):
        return np.prod(np.maximum(self._xywhaN[2:4], 0))

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self._xywhaN = xywhaN_clip(self._xywhaN, xyxyN_rgn=xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xywhaN' + str(self._xywhaN)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        xyp = xywhaN2xysN_samp(self._xywhaN)
        self._xywhaN = xysN2xywhaN(xyp * scaleN + biasN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xywhaN = xywhaN_perspective(self._xywhaN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend_(self, scale: np.ndarray = SCALE_IDENTIIY):
        self._xywhaN[2:4] *= scale
        return self

    def __eq__(self, other):
        return np.all(self._xywhaN == XYWHABorder.convert(other)._xywhaN)

    @property
    def num_xysN(self):
        return NUM_WH2XYS_SAMP

    def extract_xysN(self):
        return xywhaN2xysN_samp(self._xywhaN)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xywhaN = xysN2xywhaN(xysN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        return xywhaN2maskNb(self._xywhaN, self._size)


class XYPBorder(Movable, Clipable, AspectMeasurableFromXYSNI, HasArea, HasMaskNbAbs, HasXYXYN,
                HasWHN, Expandable, PointsExtractable, Convertable):
    @property
    def xyxyN(self) -> np.ndarray:
        return xysN2xyxyN(self._xypN)

    @property
    def xysNi(self) -> np.ndarray:
        return maskNb2xysNi(self.maskNb)

    @property
    def whN(self) -> np.ndarray:
        return np.max(self._xypN, axis=0) - np.min(self._xypN, axis=0)

    REGISTER_COVERT = Register()

    __slots__ = ('_xypN', '_size')

    def __init__(self, xypN, size):
        self._xypN = np.array(xypN).astype(np.float32)
        self._size = tuple(size)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    @property
    def measure(self):
        return np.sqrt(xypN2areaN(self._xypN))

    @property
    def whN_proj(self):
        mat = xysN2rot2N(self._xypN)
        xypN_proj = self._xypN @ mat
        return np.max(xypN_proj, axis=0) - np.min(xypN_proj, axis=0)

    @property
    def measure_min(self):
        return np.min(self.whN_proj)

    @property
    def measure_max(self):
        return np.max(self.whN_proj)

    @property
    def aspect(self):
        wh = self.whN_proj
        return np.max(wh) / max(np.min(wh), 1e-7)

    @property
    def area(self):
        return xypN2areaN(self._xypN)

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        # print('AAA', self._xypN.shape)
        # if len(self._xypN.shape) == 3:
        #     a = 3
        self._xypN = xypN_clip(self._xypN, xyxyN_rgn=xyxyN_rgn)
        return self

    def __repr__(self):
        return 'xypN' + str(self._xypN)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self._xypN = self._xypN * scaleN + biasN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._xypN = xysN_perspective(self._xypN, homographyN=homographyN)
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def expend_(self, scale: np.ndarray = SCALE_IDENTIIY):
        xy = np.mean(self._xypN, axis=0)
        self._xypN = (self._xypN - xy) * scale + xy
        return self

    def __eq__(self, other):
        return np.all(self._xypN == XYPBorder.convert(other)._xypN)

    @property
    def num_xysN(self):
        return self._xypN.shape[0]

    def extract_xysN(self):
        return self._xypN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self._xypN = xysN
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @property
    def maskNb(self):
        return xypN2maskNb(self._xypN, self._size)


class AbsBoolRegion(Movable, Clipable, AspectMeasurableFromXYSNI, HasArea, HasMaskNbAbs, BoolMaskExtractable,
                    Convertable):
    @property
    def xysNi(self) -> np.ndarray:
        iys, ixs = np.nonzero(self._maskNb_abs)
        ixys = np.stack([ixs, iys], axis=1)
        return ixys

    __slots__ = ('_maskNb_abs',)
    CONF_THRES = 0.5

    def __init__(self, maskNb_abs: np.ndarray):
        self._maskNb_abs = np.array(maskNb_abs).astype(bool)

    @property
    def size(self):
        shape = self._maskNb_abs.shape
        return (shape[1], shape[0])

    @size.setter
    def size(self, size):
        if not tuple(size) == self.size:
            A = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32)
            self._maskNb_abs = cv2.warpAffine(self._maskNb_abs.astype(np.float32), A, size) > 0.5

    @property
    def maskNb(self):
        return self._maskNb_abs

    @property
    def num_bool_chan(self) -> int:
        return 1

    def extract_maskNb(self):
        return self._maskNb_abs[..., None]

    def refrom_maskNb(self, maskNb, **kwargs):
        self._maskNb_abs = maskNb[..., 0]
        return self

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        if np.all(xyxyN_rgn == np.array((0, 0, self._maskNb_abs.shape[1], self._maskNb_abs.shape[0]))):
            return self
        maskNb = xyxyN2maskNb(xyxyN_rgn, size=self.size)
        self._maskNb_abs = self._maskNb_abs * maskNb
        return self

    @property
    def measure(self):
        return np.sqrt(np.sum(self._maskNb_abs))

    @property
    def area(self):
        return np.sum(self._maskNb_abs)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY,
                resample=cv2.INTER_CUBIC, **kwargs):
        A = np.array([[scaleN[0], 0, biasN[0]], [0, scaleN[1], biasN[1]]])
        self._maskNb_abs = cv2.warpAffine(self._maskNb_abs.astype(np.float32), A, size,
                                          flags=resample) > AbsBoolRegion.CONF_THRES
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, resample=cv2.INTER_CUBIC,
                     **kwargs):
        self._maskNb_abs = cv2.warpPerspective(self._maskNb_abs.astype(np.float32), homographyN, size,
                                               flags=resample) > AbsBoolRegion.CONF_THRES
        return self

    def __eq__(self, other):
        return np.all(self._maskNb_abs == AbsBoolRegion.convert(other)._maskNb_abs)

    def __repr__(self):
        return 'bmsk' + str(self.size)


class AbsValRegion(Movable, Clipable, AspectMeasurableFromXYSNI, HasArea, HasMaskNbAbs,
                   HasMaskNAbs, PointsExtractable, Convertable):
    @property
    def xysNi(self) -> np.ndarray:
        iys, ixs = np.nonzero(self._maskN_abs > self._conf_thres)
        ixys = np.stack([ixs, iys], axis=1)
        return ixys

    __slots__ = ('_maskN_abs', '_conf_thres')

    def __init__(self, maskN_abs, conf_thres: float = 0.5):
        self._maskN_abs = np.array(maskN_abs).astype(np.float32)
        self._conf_thres = conf_thres

    @property
    def size(self):
        shape = self._maskN_abs.shape
        return (shape[1], shape[0])

    @size.setter
    def size(self, size):
        if not size == self.size:
            A = np.array([[1, 0, 0], [0, 1, 0]]).astype(np.float32)
            self._maskN_abs = cv2.warpAffine(self._maskN_abs, A, size)

    @property
    def maskN(self):
        return self._maskN_abs

    @property
    def conf_thres(self) -> float:
        return self._conf_thres

    @property
    def maskNb(self):
        return self._maskN_abs > self._conf_thres

    @property
    def num_xysN(self) -> int:
        return 4

    def extract_xysN(self):
        return xyxyN2xypN(np.array([0, 0, self._maskN_abs.shape[1], self._maskN_abs.shape[0]])).astype(np.float32)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, resample=cv2.INTER_CUBIC, **kwargs):
        xyp_ori = self.extract_xysN()
        H = cv2.getPerspectiveTransform(xyp_ori, xysN.astype(np.float32))
        self._maskN_abs = cv2.warpPerspective(self._maskN_abs, H, size, flags=resample)
        return self

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        if np.all(xyxyN_rgn == np.array([0, 0, self._maskN_abs.shape[1], self._maskN_abs.shape[0]])):
            return self
        maskNb = xyxyN2maskNb(xyxyN_rgn, size=self.size)
        self._maskN_abs = self._maskN_abs * maskNb
        return self

    @property
    def measure(self):
        return np.sqrt(np.sum(self._maskN_abs > self._conf_thres))

    @property
    def area(self):
        return np.sum(self._maskN_abs > self._conf_thres)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY,
                resample=cv2.INTER_CUBIC, **kwargs):
        A = np.array([[scaleN[0], 0, biasN[0]], [0, scaleN[1], biasN[1]]])
        self._maskN_abs = cv2.warpAffine(self._maskN_abs, A, size, flags=resample)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, resample=cv2.INTER_CUBIC,
                     **kwargs):
        self._maskN_abs = cv2.warpPerspective(self._maskN_abs, homographyN, size, flags=resample)
        return self

    def __eq__(self, other):
        other = AbsValRegion.convert(other)
        return np.all(self._maskN_abs == other._maskN_abs) and self._conf_thres == other._conf_thres

    def __repr__(self):
        return 'amsk' + str(self.size)


class NailValRegion(Movable, Clipable, AspectMeasurableFromXYSNI, HasArea, HasMaskNbAbs,
                    HasMaskNAbs, PointsExtractable, Convertable):
    __slots__ = ('_maskN_nail', '_scaleN', '_conf_thres')

    def __init__(self, maskN_nail: np.ndarray, size: Tuple, conf_thres: float = 0.5):
        self._maskN_nail = np.array(maskN_nail).astype(np.float32)
        self._size = size
        self._conf_thres = conf_thres

    @property
    def xysNi(self) -> np.ndarray:
        iys, ixs = np.nonzero(self._maskN_nail > self._conf_thres)
        scaler = np.array(self._size) / np.array(self.nail_size)
        ixys = np.stack([ixs, iys], axis=1) * scaler
        return ixys.astype(np.int32)

    @property
    def size(self):
        return self._size

    @property
    def nail_size(self):
        shape = self._maskN_nail.shape
        return (shape[1], shape[0])

    @size.setter
    def size(self, size):
        self._size = size

    @property
    def maskN(self):
        return cv2.resize(self._maskN_nail, self._size)

    @property
    def conf_thres(self) -> float:
        return self._conf_thres

    @property
    def maskNb(self):
        return cv2.resize(self._maskN_nail, self._size) > self._conf_thres

    @property
    def num_xysN(self) -> int:
        return 4

    @property
    def measure(self):
        scaler = np.prod(np.array(self._size) / np.array(self.nail_size))
        return np.sqrt(np.sum(self._maskN_nail > self._conf_thres) * scaler)

    @property
    def area(self):
        scaler = np.prod(np.array(self._size) / np.array(self.nail_size))
        return np.sum(self._maskN_nail > self._conf_thres) * scaler

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY,
                resample=cv2.INTER_CUBIC, **kwargs):
        nail_size = self.nail_size
        _scaler = np.array(self._size) / np.array(nail_size)
        new_nail_size = (round(size[0] / _scaler[0]), round(size[1] / _scaler[1]))
        A = np.array([[scaleN[0], 0, biasN[0] * _scaler[0]], [0, scaleN[1], biasN[1] * _scaler[1]]])
        self._maskN_nail = cv2.warpAffine(self._maskN_nail, A, new_nail_size, flags=resample)
        self._size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, resample=cv2.INTER_CUBIC,
                     **kwargs):
        nail_size = self.nail_size
        _scaler = np.array(self._size) / np.array(nail_size)
        new_nail_size = (round(size[0] / _scaler[0]), round(size[1] / _scaler[1]))
        scale_ext = np.concatenate([_scaler, [1]], axis=0)
        homographyN = homographyN / (scale_ext[:, None] / scale_ext[None, :])
        self._maskN_nail = cv2.warpPerspective(self._maskN_nail, homographyN, new_nail_size, flags=resample)
        self._size = size
        return self

    def extract_xysN(self):
        return xyxyN2xypN(np.array([0, 0, self._size[0], self._size[1]])).astype(np.float32)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, resample=cv2.INTER_CUBIC, **kwargs):
        xyp_ori = self.extract_xysN()
        H = cv2.getPerspectiveTransform(xyp_ori, xysN.astype(np.float32))
        self.perspective_(size=size, homographyN=H, resample=resample)
        return self

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        if np.all(xyxyN_rgn == np.array([0, 0, self._size[0], self._size[1]])):
            return self
        nail_size = self.nail_size
        _scaler = np.array(self._size) / np.array(nail_size)
        maskNb = xyxyN2maskNb(xyxyN_rgn / np.tile(_scaler, reps=2), size=nail_size)
        self._maskN_nail = self._maskN_nail * maskNb
        return self

    def __eq__(self, other):
        other = NailValRegion.convert(other)
        return self._size == other._size and np.all(self._maskN_nail == other._maskN_nail) \
               and self._conf_thres == other._conf_thres

    def __repr__(self):
        return 'nmsk' + str(self.size)


class RefValRegion(Movable, Clipable, AspectMeasurableFromXYSNI, HasArea, HasMaskNbAbs, HasWHN, HasXYXYN,
                   HasMaskNAbs, PointsExtractable, Convertable):
    REGISTER_COVERT = Register()
    __slots__ = ('_maskN_ref', '_xyN', '_size', '_conf_thres')

    def __init__(self, maskN_ref, xyN, size, conf_thres: float = 0.5):
        self._maskN_ref = np.array(maskN_ref).astype(np.float32)
        self._conf_thres = conf_thres
        self._xyN = np.array(xyN).astype(np.int32)
        self._size = size

    @property
    def xysNi(self) -> np.ndarray:
        iys, ixs = np.nonzero(self._maskN_ref > self._conf_thres)
        ixys_ref = np.stack([ixs, iys], axis=1)
        return self._xyN + ixys_ref

    @property
    def maskNb_ref(self):
        return self._maskN_ref > self._conf_thres

    @property
    def xyN(self) -> np.ndarray:
        return self._xyN

    @property
    def conf_thres(self) -> float:
        return self._conf_thres

    @property
    def whN(self):
        return np.array((self._maskN_ref.shape[1], self._maskN_ref.shape[0]))

    @property
    def measure(self):
        return np.sqrt(np.sum(self._maskN_ref > self._conf_thres))

    @property
    def area(self):
        return np.sum(self._maskN_ref > self._conf_thres)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    @property
    def xyxyN(self):
        return np.concatenate([self._xyN, self._xyN + self.whN], axis=0).astype(np.float32)

    @property
    def maskNb(self):
        maskNb = np.zeros(shape=(self.size[1], self.size[0]), dtype=bool)
        shape_ref = self._maskN_ref.shape
        maskNb[self._xyN[1]:self._xyN[1] + shape_ref[0], self._xyN[0]:self._xyN[0] + shape_ref[1]] = \
            self.maskNb_ref[0:self.size[1] - self._xyN[1], 0:self.size[0] - self._xyN[0]]
        return maskNb

    @property
    def maskN(self):
        maskN = np.zeros(shape=(self.size[1], self.size[0]), dtype=np.float32)
        shape_ref = self._maskN_ref.shape
        maskN[self._xyN[1]:self._xyN[1] + shape_ref[0], self._xyN[0]:self._xyN[0] + shape_ref[1]] = \
            self._maskN_ref[0:self.size[1] - self._xyN[1], 0:self.size[0] - self._xyN[0]]
        return maskN

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        xyxyN_rgn = xyxyN_rgn.astype(np.int32)
        whN = self.whN
        xyN_max = self._xyN + whN
        if np.any(self._xyN < xyxyN_rgn[:2]) or np.any(xyN_max > xyxyN_rgn[2:4]):
            if np.any(xyN_max <= xyxyN_rgn[:2]) or np.any(self._xyN >= xyxyN_rgn[2:4]):
                self._maskN_ref = np.zeros(shape=(0, 0))
                return self
            xyxyN_rgn_ref = xyxyN_rgn - self._xyN[[0, 1, 0, 1]]
            xyxyN_rgn_ref = xyxyN_clip(xyxyN_rgn_ref, np.array([0, 0, whN[0], whN[1]]))
            self._maskN_ref = self._maskN_ref[xyxyN_rgn_ref[1]:xyxyN_rgn_ref[3], xyxyN_rgn_ref[0]:xyxyN_rgn_ref[2]]
            self._xyN = np.maximum(self._xyN, xyxyN_rgn[:2])
        return self

    @property
    def num_xysN(self) -> int:
        return 4

    def extract_xysN(self):
        return xyxyN2xypN(self.xyxyN)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, resample=cv2.INTER_CUBIC, **kwargs):
        if np.prod(self.whN) == 0:  # 避免出现无法求解的矩阵
            return self
        xypN_ori = xyxyN2xypN(self.xyxyN)
        xyxyN = np.round(xysN2xyxyN(xysN)).astype(np.int32)
        H = cv2.getPerspectiveTransform((xypN_ori - self._xyN).astype(np.float32),
                                        (xysN - xyxyN[:2]).astype(np.float32))
        size_ref = tuple(xyxyN[2:] - xyxyN[:2])
        self._maskN_ref = cv2.warpPerspective(self._maskN_ref.astype(np.float32), H, size_ref, flags=resample)
        self._xyN = xyxyN[:2]
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY,
                resample=cv2.INTER_CUBIC, **kwargs):
        xyxyN_lin = xysN2xyxyN(xyxyN2xypN(self.xyxyN) * np.array(scaleN) + np.array(biasN))
        xyxyN_lin = np.round(xyxyN_lin).astype(np.int32)
        Ap = np.array([[scaleN[0], 0, 0], [0, scaleN[1], 0]])
        size_ref = tuple(xyxyN_lin[2:] - xyxyN_lin[:2])
        self._maskN_ref = cv2.warpAffine(self._maskN_ref.astype(np.float32), Ap, size_ref, flags=resample)
        self._xyN = xyxyN_lin[:2]
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, resample=cv2.INTER_CUBIC,
                     **kwargs):
        xypN_ori = xyxyN2xypN(self.xyxyN)
        xypN_persp = xysN_perspective(xypN_ori, homographyN)
        xyxyN_persp = np.round(xysN2xyxyN(xypN_persp)).astype(np.int32)
        Hp = cv2.getPerspectiveTransform((xypN_ori - self._xyN).astype(np.float32),
                                         (xypN_persp - xyxyN_persp[:2]).astype(np.float32))
        size_ref = tuple(xyxyN_persp[2:] - xyxyN_persp[:2])
        self._maskN_ref = cv2.warpPerspective(self._maskN_ref.astype(np.float32), Hp, size_ref, flags=resample)
        self._xyN = xyxyN_persp[:2]
        self.clip_(np.array([0, 0, size[0], size[1]]))
        self.size = size
        return self

    @staticmethod
    def _maskNb2xyxyN(maskNb):
        ys, xs = np.where(maskNb)
        xyxy = np.array([np.min(xs), np.min(ys), np.max(xs) + 1, np.max(ys) + 1]).astype(np.int32) \
            if len(ys) > 0 else np.array([0, 0, 1, 1]).astype(np.int32)
        return xyxy

    @staticmethod
    def from_maskNb_xyxyN(maskNb, xyxyN):
        size = (maskNb.shape[1], maskNb.shape[0])
        xyxyN = xyxyN_clip(xyxyN, np.array([0, 0, maskNb.shape[1], maskNb.shape[0]])).astype(np.int32)
        maskNb_ref = maskNb[xyxyN[1]:xyxyN[3], xyxyN[0]:xyxyN[2]]
        return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxyN[:2], size=size, conf_thres=0.5)

    def __repr__(self):
        return 'rmsk' + str(self.whN)

    def __eq__(self, other):
        other = RefValRegion.convert(other)
        return np.all(self._maskN_ref == other._maskN_ref) and np.all(self._xyN == other._xyN)


# </editor-fold>

# <editor-fold desc='注册json变换'>
REGISTRY_JSON_ENCDEC_BY_INIT(IndexCategory)
REGISTRY_JSON_ENCDEC_BY_INIT(OneHotCategory)
REGISTRY_JSON_ENCDEC_BY_INIT(XYSSurface)
REGISTRY_JSON_ENCDEC_BY_INIT(XYSColoredSurface)
REGISTRY_JSON_ENCDEC_BY_INIT(XYSGraph)
REGISTRY_JSON_ENCDEC_BY_INIT(XYSPoint)
REGISTRY_JSON_ENCDEC_BY_INIT(XYXYBorder)
REGISTRY_JSON_ENCDEC_BY_INIT(XYWHBorder)
REGISTRY_JSON_ENCDEC_BY_INIT(XYWHABorder)
REGISTRY_JSON_ENCDEC_BY_INIT(XYPBorder)
REGISTRY_JSON_ENCDEC_BY_INIT(AbsBoolRegion)
REGISTRY_JSON_ENCDEC_BY_INIT(AbsValRegion)
REGISTRY_JSON_ENCDEC_BY_INIT(RefValRegion)


# </editor-fold>

# <editor-fold desc='几何元素相互转化'>

@XYSGraph.REGISTER_COVERT.registry(XYSSurface)
def _vsurf2d2vgraph2d(vsurf2d: XYSSurface):
    return XYSGraph(xysN=vsurf2d.xysN, edgesN=vsurf2d.edgesN, size=vsurf2d.size)


@XYWHBorder.REGISTER_COVERT.registry(RefValRegion)
def _refval_region2xywh_border(border: RefValRegion):
    return XYWHBorder(xysN2xywhN(border.xysNi), size=border.size)


@XYXYBorder.REGISTER_COVERT.registry(RefValRegion)
def _refval_region2xyxy_border(border: RefValRegion):
    return XYXYBorder(xysN2xyxyN(border.xysNi), size=border.size)


@XYXYBorder.REGISTER_COVERT.registry(XYWHBorder)
def _xywh_border2xyxy_border(border: XYWHBorder):
    return XYXYBorder(xywhN2xyxyN(border._xywhN), border.size)


@XYXYBorder.REGISTER_COVERT.registry(Sequence)
def _seq2xyxy_border(border: Sequence):
    border = np.array(border)
    size = tuple(border[2:4].astype(np.int32))
    return XYXYBorder(border, size=size)


@XYXYBorder.REGISTER_COVERT.registry(XYWHABorder)
def _xywha_border2xyxy_border(border: XYWHABorder):
    xyxy = xywhaN2xyxyN(border._xywhaN)
    xyxy = xyxyN_clip(xyxy, xyxyN_rgn=np.array([0, 0, border.size[0], border.size[1]]))
    return XYXYBorder(xyxy, border.size)


@XYXYBorder.REGISTER_COVERT.registry(XYPBorder)
def _xyp_border2xyxy_border(border: XYPBorder):
    return XYXYBorder(xysN2xyxyN(border._xypN), border.size)


@XYXYBorder.REGISTER_COVERT.registry(XYSGraph)
def _xyg_border2xyxy_border(border: XYSGraph):
    return XYXYBorder(xysN2xyxyN(border._xysN), border.size)


@XYXYBorder.REGISTER_COVERT.registry(XYSPoint)
def _xys_point2xyxy_border(border: XYSPoint):
    return XYXYBorder(xysN2xyxyN(border._xysN), border.size)


@XYWHBorder.REGISTER_COVERT.registry(XYXYBorder)
def _xyxy_border2xywh_border(border: XYXYBorder):
    return XYWHBorder(xyxyN2xywhN(border._xyxyN), size=border.size)


@XYWHBorder.REGISTER_COVERT.registry(XYWHABorder)
def _xywha_border2xywh_border(border: XYWHABorder):
    xyxy = xywhaN2xyxyN(border._xywhaN)
    xyxy = xyxyN_clip(xyxy, xyxyN_rgn=np.array([0, 0, border.size[0], border.size[1]]))
    return XYWHBorder(xyxyN2xywhN(xyxy), size=border.size)


@XYWHBorder.REGISTER_COVERT.registry(XYPBorder)
def _xyp_border2xywh_border(border: XYPBorder):
    return XYWHBorder(xysN2xywhN(border._xypN), size=border.size)


@XYWHBorder.REGISTER_COVERT.registry(Sequence)
def _seq2xywh_border(border: Sequence):
    border = np.array(border)
    size = tuple((border[:2] + border[2:4] / 2).astype(np.int32))
    return XYWHBorder(border, size=size)


@XYWHABorder.REGISTER_COVERT.registry(XYPBorder)
def _xyp_border2xywha_border(border: XYPBorder):
    return XYWHABorder(xysN2xywhaN(border._xypN), size=border.size)


@XYWHABorder.REGISTER_COVERT.registry(XYSPoint)
def _xys_border2xywha_border(border: XYSPoint):
    return XYWHABorder(xysN2xywhaN(border.xysN), size=border.size)


@XYWHABorder.REGISTER_COVERT.registry(RefValRegion)
def _refval_region2xywha_border(border: RefValRegion):
    return XYWHABorder(xysN2xywhaN(border.xysNi), size=border.size)


@XYWHABorder.REGISTER_COVERT.registry(Sequence)
def _seq2xywha_border(border: Sequence):
    xywhaN = np.array(border)
    xyxyN = xywhaN2xyxyN(xywhaN)
    return XYWHABorder(xywhaN, size=tuple(xyxyN[2:4].astype(np.int32)))


@XYWHABorder.REGISTER_COVERT.registry(XYXYBorder)
def _xyxy_border2xywha_border(border: XYXYBorder):
    return XYWHABorder(xyxyN2xywhaN(border._xyxyN), size=border.size)


@XYWHABorder.REGISTER_COVERT.registry(XYWHBorder)
def _xywh_border2xywha_border(border: XYWHBorder):
    return XYWHABorder(xywhN2xywhaN(border._xywhN), size=border.size)


@XYPBorder.REGISTER_COVERT.registry(XYXYBorder)
def _xyxy_border2xyp_border(border: XYXYBorder):
    return XYPBorder(xyxyN2xypN(border._xyxyN), size=border.size)


@XYPBorder.REGISTER_COVERT.registry(XYWHABorder)
def _xywha_border2xyp_border(border: XYWHABorder):
    return XYPBorder(xywhaN2xypN(border._xywhaN), size=border.size)


@XYPBorder.REGISTER_COVERT.registry(Sequence)
def _seq2xyp_border(border: Sequence):
    xypN = np.array(border)
    size = tuple(np.max(xypN, axis=0).astype(np.int32))
    return XYPBorder(xypN, size=size)


@XYPBorder.REGISTER_COVERT.registry(XYWHBorder)
def _xywh_border2xyp_border(border: XYWHBorder):
    return XYPBorder(xywhN2xypN(border._xywhN), size=border.size)


@AbsBoolRegion.REGISTER_COVERT.registry(HasMaskNbAbs)
def _bool_region2abs_bool_region(region: HasMaskNbAbs):
    return AbsBoolRegion(maskNb_abs=region.maskNb)


@AbsValRegion.REGISTER_COVERT.registry(HasMaskNbAbs)
def _bool_region2abs_val_region(region: HasMaskNbAbs):
    return AbsValRegion(maskN_abs=region.maskNb, conf_thres=0.5)


@RefValRegion.REGISTER_COVERT.registry(AbsBoolRegion)
def _abs_bool_region2ref_val_region(region: AbsBoolRegion):
    xyxy = RefValRegion._maskNb2xyxyN(region.maskNb)
    maskNb_ref = region._maskNb_abs[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxy[:2], size=region.size, conf_thres=0.5)


@RefValRegion.REGISTER_COVERT.registry(AbsValRegion)
def _abs_val_region2ref_val_region(region: AbsValRegion):
    xyxy = RefValRegion._maskNb2xyxyN(region.maskNb)
    maskNb_ref = region._maskN_abs[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxy[:2], size=region.size,
                        conf_thres=region._conf_thres)


@RefValRegion.REGISTER_COVERT.registry(XYXYBorder)
def _xyxy_border2ref_val_region(region: XYXYBorder):
    xyxyN = region._xyxyN.astype(np.int32)
    size_ref = xyxyN[2:4] - xyxyN[:2]
    maskNb_ref = np.full(shape=(size_ref[1], size_ref[0]), fill_value=1)
    return RefValRegion(maskN_ref=maskNb_ref, xyN=xyxyN[:2], size=region.size, conf_thres=0.5)


@RefValRegion.REGISTER_COVERT.registry(XYXYBorder, XYWHBorder, XYWHABorder, XYPBorder)
def _border2ref_val_region(region: Union[XYXYBorder, XYWHBorder, XYWHABorder, XYPBorder]):
    xypN = XYPBorder.convert(region)._xypN
    xyxyN = xysN2xyxyN(xypN).astype(np.int32)
    xyxyN = xyxyN_clip(xyxyN, xyxyN_rgn=np.array([0, 0, region.size[0], region.size[1]]))
    maskNb_ref = xypN2maskNb(xypN - xyxyN[:2], size=tuple(xyxyN[2:4] - xyxyN[:2]))
    return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxyN[:2], size=region.size, conf_thres=0.5)


@RefValRegion.REGISTER_COVERT.registry(XYSColoredSurface)
def _xysurf2ref_val_region(region: XYSColoredSurface):
    xyxyN = region.xyxyN.astype(np.int32)
    xysN_ref = region.xysN - xyxyN[:2]
    size_ref = xyxyN[2:4] - xyxyN[:2]
    maskNb_ref = np.full(shape=(size_ref[1], size_ref[0]), fill_value=0, dtype=np.uint8)
    verts = np.round(xysN_ref[region.surfsN]).astype(np.int32)
    # maskNb_ref = cv2.fillPoly(maskNb_ref, list(verts), color=1)
    for v in verts:
        maskNb_ref = cv2.fillPoly(maskNb_ref, [v], color=1)
    return RefValRegion(maskN_ref=maskNb_ref.astype(np.float32), xyN=xyxyN[:2], size=region.size, conf_thres=0.5)

# </editor-fold>
