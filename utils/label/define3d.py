from typing import NoReturn, Dict

from .element import *

ROTATION_IDENTITY = np.eye(3)
TRANSLATION_IDENTITY = np.zeros(3)


def camera_lift_simple(xysN: np.ndarray, zsN: np.ndarray, focalN: np.ndarray, biasN: np.ndarray) -> np.ndarray:
    xysN_lift = zsN / focalN * (xysN - biasN)
    xyzsN_lift = np.concatenate([xysN_lift, zsN[..., None]], axis=-1)
    return xyzsN_lift


def camera_proj_simple(xyzsN: np.ndarray, focalN: np.ndarray, biasN: np.ndarray) -> np.ndarray:
    xysN, zsN = xyzsN[..., :2], xyzsN[..., 2:]
    xysN_proj = focalN * xysN / zsN + biasN
    return xysN_proj


def camera_proj(xyzsN: np.ndarray, projection: np.ndarray) -> np.ndarray:
    xyzsN_proj = xyzsN @ projection.T
    z = np.where(xyzsN_proj[..., 2:] > 0, xyzsN_proj[..., 2:], np.nan)
    return xyzsN_proj[..., :2] / z


def camera_lift(xysN: np.ndarray, zsN: np.ndarray, projection: np.ndarray) -> np.ndarray:
    projection_inv = np.linalg.inv(projection)
    projection_inv = projection_inv / projection_inv[2, 2]
    ext_shape = list(xysN.shape)[:-1] + [1]
    xys_ext = np.concatenate([xysN, np.ones(ext_shape)], axis=-1)
    xys_ext = xys_ext @ projection_inv.T
    xyzs = xys_ext / xys_ext[..., 2:] * zsN[..., None]
    return xyzs


def scaleN_biasN2homographyN(scaleN: np.ndarray, biasN: np.ndarray) -> np.ndarray:
    homography = np.array([[scaleN[0], 0, biasN[0]], [0, scaleN[1], biasN[1]], [0, 0, 1]])
    return homography


def focalN_biasN2intrinsicN(focalN: np.ndarray, biasN: np.ndarray) -> np.ndarray:
    intrinsic = np.array([[focalN[0], 0, biasN[0]], [0, focalN[1], biasN[1]], [0, 0, 1]])
    return intrinsic


class MCamera(Movable):
    __slots__ = ('_intrinsicN', '_size')

    def __init__(self, intrinsicN: np.ndarray, size: tuple):
        self._intrinsicN = np.array(intrinsicN)
        self._size = size

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self._intrinsicN = homographyN @ self._intrinsicN
        self._size = size
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY,
                scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self._intrinsicN = scaleN_biasN2homographyN(scaleN, biasN) @ self._intrinsicN
        self._size = size
        return self

    @staticmethod
    def UNIT(size: tuple):
        return MCamera(np.eye(3), size=size)

    @staticmethod
    def from_focal_bias(focalN: np.ndarray, biasN: np.ndarray, size: tuple):
        intrinsic = focalN_biasN2intrinsicN(focalN, biasN)
        return MCamera(intrinsic, size)

    @property
    def intrinsicN(self) -> np.ndarray:
        return self._intrinsicN

    def calibrate2rotationN(self, intrinsicN: np.ndarray):
        trans = np.linalg.inv(intrinsicN) @ self._intrinsicN
        trans = trans / np.sign(trans[2, 2])
        return trans

    def calibrate2rotationN_(self, intrinsicN: np.ndarray, size: tuple) -> np.ndarray:
        trans = np.linalg.inv(intrinsicN) @ self._intrinsicN
        trans = trans / np.sign(trans[2, 2])
        self._intrinsicN = intrinsicN
        self._size = size
        return trans

    @property
    def size(self) -> tuple:
        return self._size

    @size.setter
    def size(self, size):
        self._size = tuple(size)

    def project_xyzsN(self, xyzsN: np.ndarray):
        return camera_proj(xyzsN, self._intrinsicN)

    def lift_xysN_zsN(self, xysN: np.ndarray, zsN: np.ndarray):
        return camera_lift(xysN, zsN, self._intrinsicN)

    def __repr__(self):
        return 'CAMERA' + str(self._size)
        # return 'CAMERA{focal' + str(self.focalN) + ' bias' + str(self.biasN) + '}'


class HasCalibration(metaclass=ABCMeta):
    @abstractmethod
    def calibrate_(self, intrinsicN: np.ndarray, size: tuple):
        pass

    def calibrate_as(self, camera: MCamera) -> np.ndarray:
        return copy.deepcopy(self).calibrate_(camera.intrinsicN, camera.size)

    def calibrate_as_(self, camera: MCamera) -> np.ndarray:
        return self.calibrate_(camera.intrinsicN, camera.size)


class HasXYZSN(metaclass=ABCMeta):
    @property
    @abstractmethod
    def xyzsN(self) -> np.ndarray:
        pass


class HasXYZXYZSN(metaclass=ABCMeta):
    @property
    @abstractmethod
    def xyzxyzsN(self) -> np.ndarray:
        pass


class HasXYZXYZN(metaclass=ABCMeta):
    @property
    @abstractmethod
    def xyzxyzN(self) -> np.ndarray:
        pass


class HasXYZXYZNFromHasXYZSN(HasXYZSN, HasXYZXYZN):
    @property
    def xyzxyzN(self) -> np.ndarray:
        return xyzsN2xyzxyzN(self.xyzsN)


class HasVertexGraph3D(HasXYZSN):
    @property
    @abstractmethod
    def edgesN(self) -> np.ndarray:
        pass


class HasCamera(metaclass=ABCMeta):
    @property
    @abstractmethod
    def camera(self) -> MCamera:
        pass


class Movable3D(metaclass=ABCMeta):
    @abstractmethod
    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        pass

    def transform(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        other = copy.deepcopy(self)
        return other.transform_(rotation=rotation, translation=translation)


class Point3DExtractable(metaclass=ABCMeta):
    @abstractmethod
    def extract_xyzsN(self) -> np.ndarray:
        pass

    @abstractmethod
    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        pass

    @property
    @abstractmethod
    def num_xyzsN(self) -> int:
        pass


class Projectable(metaclass=ABCMeta):
    @abstractmethod
    def project(self, camera: MCamera):
        pass


class ProjectedFromProjectableHasCamera(HasCamera, Projectable):
    @property
    def projected(self):
        return self.project(self.camera)


class ClipableProjected(metaclass=ABCMeta):
    @abstractmethod
    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        pass

    def clip3d(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        return copy.deepcopy(self).clip3d_(xyxyN_rgn, camera, **kwargs)


class MeasurableProjected(metaclass=ABCMeta):

    @abstractmethod
    def measure_projected(self, camera: MCamera) -> float:
        pass


class HasVolume(metaclass=ABCMeta):
    @property
    @abstractmethod
    def volume(self) -> float:
        pass


class HasXYXYNProjected(metaclass=ABCMeta):

    @abstractmethod
    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        pass


class MeasurableProjectedFromHasXYZSN(HasXYZSN, MeasurableProjected):

    def measure_projected(self, camera: MCamera) -> float:
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[:2]))


class MovableFromMovable3DCamera(Movable3D, Movable, HasCamera, HasCalibration):
    def calibrate_(self, intrinsicN: np.ndarray, size: tuple) -> np.ndarray:
        trans = self.camera.calibrate2rotationN_(intrinsicN, size)
        self.transform_(rotation=trans)
        return trans

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.camera.linear_(size, biasN, scaleN)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.camera.perspective_(size, homographyN)
        return self

    @property
    def size(self) -> Tuple[int, int]:
        return self.camera.size

    @size.setter
    def size(self, size) -> NoReturn:
        self.camera.size = size
