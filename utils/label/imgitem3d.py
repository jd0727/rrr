from .element3d import *
from .imgitem import *


class StereoItem(dict, Movable3D, Projectable, MeasurableProjected, ClipableProjected):
    __slots__ = []

    def __init__(self, *seq, **kwargs):
        dict.__init__(self, *seq, **kwargs)


class StereoBoxItem(StereoItem, Convertable, ):
    __slots__ = ('border', 'category',)
    REGISTER_COVERT = Register()

    def __init__(self, border, category, *seq, **kwargs):
        dict.__init__(self, *seq, **kwargs)
        # print(border)
        self.border = border
        self.category = category

    def measure_projected(self, camera: MCamera) -> float:
        return self.border.measure_projected(camera)

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        self.border.clip3d_(xyxyN_rgn, camera)
        return self

    def project(self, camera: MCamera):
        return BoxItem(border=self.border.project(camera), category=self.category, **self)

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        self.border.transform_(rotation=rotation, translation=translation)
        return self

    def __eq__(self, other):
        return (isinstance(other, StereoBoxItem)
                and self.category == other.category
                and self.border == other.border)

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + super(StereoBoxItem, self).__repr__()


class StereoMixItem(StereoBoxItem, Movable, Clipable, HasXYXYN):

    @property
    def xyxyN(self) -> np.ndarray:
        return self.border_mv.xyxyN

    @property
    def size(self) -> Tuple[int, int]:
        return self.border_mv.size

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border_mv.clip_(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.border_mv.linear_(size, biasN=biasN, scaleN=scaleN, **kwargs)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.border_mv.perspective_(size, homographyN=homographyN, **kwargs)
        return self

    __slots__ = ('border', 'border_mv', 'category',)
    REGISTER_COVERT = Register()

    def __init__(self, border, mvborder, category, *seq, **kwargs):
        StereoBoxItem.__init__(self, border, category, *seq, **kwargs)
        self.border_mv = mvborder

    def project(self, camera: MCamera):
        return DualBoxItem(border=self.border_mv, border2=self.border.project(camera),
                           category=self.category, **self)

    def __eq__(self, other):
        return (isinstance(other, StereoMixItem)
                and self.category == other.category
                and self.border == other.border
                and self.border_mv == other.border_mv)

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + self.border_mv.__repr__() + StereoItem.__repr__(self)


# <editor-fold desc='注册json变换'>
REGISTRY_JSON_ENCDEC_BY_INIT(StereoBoxItem)

# </editor-fold>
