try:
    from pytorch3d.renderer import PerspectiveCameras, TexturesVertex, MeshRenderer, MeshRasterizer, \
        RasterizationSettings, HardPhongShader, DirectionalLights
    from pytorch3d.structures import Meshes, join_meshes_as_batch
except Exception as e:
    Meshes = None
    PerspectiveCameras = None
    TexturesVertex = None

from .define3d import *
from ..typings import NV_Flt, SN_Flt
from ..iotools import load_txt, ensure_extend, os, load_img_cv2


# <editor-fold desc='几何元素'>
def _cvsN2vsN_ivsN(cvsN: np.ndarray) -> (np.ndarray, np.ndarray):
    shape, ndim = cvsN.shape[:-1], cvsN.shape[-1]
    vsN, ivsN = np.unique(cvsN.reshape(-1, ndim), return_inverse=True, axis=0)
    ivsN = ivsN.reshape(shape)
    return vsN, ivsN


class XYZSPoint(Convertable, Point3DExtractable, Movable3D, Projectable, HasXYZXYZNFromHasXYZSN, MeasurableProjected,
                ClipableProjected, HasXYXYNProjected):

    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        if self.xyzsN.shape[0] == 0:
            return np.array([0, 0, 0, 0])
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return xyxy

    def measure_projected(self, camera: MCamera) -> float:
        if self.xyzsN.shape[0] == 0:
            return 0.0
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        xyxy = xyxyN_clip(xyxy, np.array(camera.size))
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[0:2]))

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        xys = camera.project_xyzsN(self._xyzsN)
        fltr_in = np.all(~np.isnan(xys), axis=1) * np.all(xys > xyxyN_rgn[:2], axis=1) \
                  * np.all(xys < xyxyN_rgn[2:4], axis=1)
        self._xyzsN = self._xyzsN[fltr_in]
        # if np.all(~fltr_in):
        #     self._xyzsN = np.zeros(shape=(0, 2))
        return self

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self._xyzsN)
        return XYSPoint(xysN=xys, size=camera.size)

    REGISTER_COVERT = Register()
    __slots__ = ('_xyzsN',)

    def __init__(self, xyzsN: np.ndarray):
        self._xyzsN = np.array(xyzsN).astype(np.float32)

    @property
    def num_xyzsN(self) -> int:
        return self._xyzsN.shape[0]

    @property
    def xyzsN(self) -> np.ndarray:
        return self._xyzsN

    def extract_xyzsN(self) -> np.ndarray:
        return self._xyzsN

    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        self._xyzsN = xyzsN

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        self._xyzsN = self._xyzsN @ rotation.T + translation
        return self


class XYZSSurface(Convertable, Point3DExtractable, Movable3D, HasVertexGraph3D, HasXYZXYZNFromHasXYZSN, Projectable,
                  MeasurableProjected, ClipableProjected, HasXYXYNProjected):
    @staticmethod
    def form_surf_xyzsN(surf_xyzsN: np.ndarray):
        xyzsN, surfsN = _cvsN2vsN_ivsN(surf_xyzsN)
        return XYZSSurface(xyzsN, surfsN)

    @staticmethod
    def CUBE(lside: float = 1):
        _verts = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
                           [1, 1, -1], [1, 1, 1], [1, -1, 1], [1, -1, -1]])
        _surfs = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 5], [3, 5, 4],
                           [4, 5, 6], [4, 6, 7], [6, 1, 0], [6, 0, 7],
                           [0, 3, 4], [0, 4, 7], [1, 6, 5], [1, 5, 2]])
        return XYZSSurface(_verts * lside / 2, _surfs)

    REGISTER_COVERT = Register()
    __slots__ = ('_xyzsN', '_surfsN')

    def __init__(self, xyzsN: np.ndarray, surfsN: np.ndarray):
        self._xyzsN = np.array(xyzsN).astype(np.float32)
        self._surfsN = np.array(surfsN).astype(np.int32)

    def resamp(self, thres: float):
        surf_comp = self._xyzsN[self._surfsN]
        surf_comp = surf_xyzsN_subdiv(surf_comp, thres=thres)
        self._xyzsN, self._surfsN = _cvsN2vsN_ivsN(surf_comp)
        return self

    @property
    def vnormsN(self) -> np.ndarray:
        fnorms = self.fnormsN
        surfs = self.surfsN
        vnorms = np.zeros(shape=self.xyzsN.shape)
        nums = np.zeros(shape=(self.xyzsN.shape[0],))
        surfs_flt = surfs.reshape(-1)
        for i in range(3):
            fnorms_flt = np.broadcast_to(fnorms[:, i:i + 1], surfs.shape).reshape(-1)
            np.add.at(vnorms[:, i], surfs_flt, fnorms_flt)
        np.add.at(nums, surfs_flt, 1)
        vnorms = vnorms / nums[:, None]
        vnorms = vnorms / np.linalg.norm(vnorms, axis=1, keepdims=True)
        return vnorms

    @property
    def fnormsN(self) -> np.ndarray:
        xyzs = self.xyzsN
        surfs = self.surfsN
        ed1 = xyzs[surfs[:, 0]] - xyzs[surfs[:, 1]]
        ed2 = xyzs[surfs[:, 1]] - xyzs[surfs[:, 2]]
        fnorms = np.cross(ed1, ed2)
        fnorms = fnorms / np.linalg.norm(fnorms, axis=1, keepdims=True)
        return fnorms

    @property
    def edgesN(self) -> np.ndarray:
        surfsN_nxt = np.roll(self._surfsN, axis=1, shift=1)
        edges = np.stack([self._surfsN, surfsN_nxt], axis=-1).reshape(-1, 2)
        edges = edges[edges[:, 0] < edges[:, 1]]
        edges = np.unique(edges, axis=0)
        return edges

    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return xyxy

    def measure_projected(self, camera: MCamera) -> float:
        if self._xyzsN.shape[0] < 3:
            return 0
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        xyxy = xyxyN_clip(xyxy, np.array(camera.size))
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[:2]))

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        xys = camera.project_xyzsN(self._xyzsN)
        fltr_in = np.all(~np.isnan(xys), axis=1) * np.all(xys > xyxyN_rgn[:2], axis=1) \
                  * np.all(xys < xyxyN_rgn[2:4], axis=1)
        if np.all(~fltr_in):
            self._xyzsN = np.zeros(shape=(0, 3), dtype=np.float32)
            self._surfsN = np.zeros(shape=(0, 3), dtype=np.int32)
        return self

    @property
    def surf_xyzsN(self) -> np.ndarray:
        return self._xyzsN[self._surfsN]

    @property
    def xyzsN(self) -> np.ndarray:
        return self._xyzsN

    @property
    def surfsN(self) -> np.ndarray:
        return self._surfsN

    @property
    def num_xyzsN(self) -> int:
        return self._xyzsN.shape[0]

    def extract_xyzsN(self) -> np.ndarray:
        return self._xyzsN

    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        self._xyzsN = xyzsN

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        self._xyzsN = self._xyzsN @ rotation.T + translation
        return self

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self._xyzsN)
        depth = np.mean(self._xyzsN[self._surfsN, 2], axis=-1)
        order = np.argsort(-depth)
        surfsN = self._surfsN[order]
        return XYSSurface(xysN=xys, surfsN=surfsN, size=camera.size)


class XYZSColoredSurface(ColoredVertex, XYZSSurface):
    REGISTER_COVERT = Register()
    __slots__ = ('_xyzsN', '_surfsN', '_vcolorsN')

    def __init__(self, xyzsN: np.ndarray, surfsN: np.ndarray, vcolorsN: Optional[np.ndarray] = None):
        XYZSSurface.__init__(self, xyzsN=xyzsN, surfsN=surfsN)
        ColoredVertex.__init__(self, vcolorsN=vcolorsN)

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self._xyzsN)
        depth = np.mean(self._xyzsN[self._surfsN, 2], axis=-1)
        order = np.argsort(-depth)
        surfsN = self._surfsN[order]
        return XYSColoredSurface(xysN=xys, surfsN=surfsN, vcolorsN=self.vcolorsN, size=camera.size)


class XYZSGraph(Convertable, Point3DExtractable, Movable3D, HasVertexGraph3D, HasXYZXYZNFromHasXYZSN, Projectable,
                MeasurableProjected, ClipableProjected, HasXYXYNProjected):
    REGISTER_COVERT = Register()
    __slots__ = ('_xyzsN', '_edgesN')

    def __init__(self, xyzsN: np.ndarray, edgesN: np.ndarray):
        self._xyzsN = np.array(xyzsN).astype(np.float32)
        self._edgesN = np.array(edgesN).astype(np.int32)

    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return xyxy

    def measure_projected(self, camera: MCamera) -> float:
        if self._xyzsN.shape[0] < 3:
            return 0
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        xyxy = xyxyN_clip(xyxy, np.array(camera.size))
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[:2]))

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        xys = camera.project_xyzsN(self._xyzsN)
        fltr_in = np.all(~np.isnan(xys), axis=1) * np.all(xys > xyxyN_rgn[:2], axis=1) \
                  * np.all(xys < xyxyN_rgn[2:4], axis=1)
        if np.all(~fltr_in):
            self._xyzsN = np.zeros(shape=(0, 3), dtype=np.float32)
            self._edgesN = np.zeros(shape=(0, 2), dtype=np.int32)
        return self

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self._xyzsN)
        return XYSGraph(xysN=xys, edgesN=self._edgesN, size=camera.size)

    @property
    def xyzsN(self) -> np.ndarray:
        return self._xyzsN

    @property
    def edgesN(self) -> np.ndarray:
        return self._edgesN

    @property
    def num_xyzsN(self) -> int:
        return self._xyzsN.shape[0]

    def extract_xyzsN(self) -> np.ndarray:
        return self._xyzsN

    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        self._xyzsN = xyzsN

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        self._xyzsN = self._xyzsN @ rotation.T + translation
        return self


class XYZWHLBorder(Convertable, Point3DExtractable, Movable3D, HasVertexGraph3D, HasXYZXYZNFromHasXYZSN,
                   Projectable, MeasurableProjected, ClipableProjected, HasXYXYNProjected, HasVolume):
    REGISTER_COVERT = Register()
    __slots__ = ('_xyzwhlN',)

    def __init__(self, xyzwhlN: np.ndarray):
        self._xyzwhlN = np.array(xyzwhlN).astype(np.float32)

    @property
    def volume(self) -> float:
        return np.prod(self._xyzwhlN[3:6])

    @property
    def xyzsN(self) -> np.ndarray:
        return xyzwhlN2xyzsN_vert(self._xyzwhlN)

    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return xyxy

    def measure_projected(self, camera: MCamera) -> float:
        if self.volume == 0:
            return 0
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        xyxy = xyxyN_clip(xyxy, np.array(camera.size))
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[:2]))

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        if np.any(np.isnan(xys)) or np.any(xyxy[:2] > xyxyN_rgn[2:4]) or np.any(xyxy[2:4] < xyxyN_rgn[:2]):
            self._xyzwhlN = np.zeros(6)
        return self

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self.xyzsN)
        return XYSGraph(xysN=xys, edgesN=self.edgesN, size=camera.size)

    @property
    def num_xyzsN(self) -> int:
        return NUM_WHL2XYZS_SAMP

    def extract_xyzsN(self) -> np.ndarray:
        return xyzwhlN2xyzsN_samp(self._xyzwhlN)

    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        vol_last = self.volume
        self._xyzwhlN = xyzsN2xyzwhlN(xyzsN)
        vol_new = self.volume
        self._xyzwhlN *= np.power(vol_last / vol_new, 1 / 3)

    @property
    def edgesN(self) -> np.ndarray:
        return EDGES_CUBE

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        xyzsN = self.extract_xyzsN()
        xyzsN = xyzsN @ rotation.T + translation
        self.refrom_xyzsN(xyzsN)
        return self


class XYZXYZBorder(Convertable, Point3DExtractable, Movable3D, HasVertexGraph3D, HasXYZXYZNFromHasXYZSN,
                   Projectable, MeasurableProjected, ClipableProjected, HasXYXYNProjected, HasVolume):
    REGISTER_COVERT = Register()
    __slots__ = ('_xyzxyzN',)

    def __init__(self, xyzxyzN: np.ndarray):
        self._xyzxyzN = np.array(xyzxyzN).astype(np.float32)

    @property
    def volume(self) -> float:
        return float(xyzxyzN2volumeN(self._xyzxyzN))

    @property
    def xyzsN(self) -> np.ndarray:
        return xyzxyzN2xyzsN_vert(self._xyzxyzN)

    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return xyxy

    def measure_projected(self, camera: MCamera) -> float:
        if self.volume == 0:
            return 0
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        xyxy = xyxyN_clip(xyxy, np.array(camera.size))
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[:2]))

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        if np.any(np.isnan(xys)) or np.any(xyxy[:2] > xyxyN_rgn[2:4]) or np.any(xyxy[2:4] < xyxyN_rgn[:2]):
            self._xyzxyzN = np.zeros(6)
        return self

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self.xyzsN)
        return XYSGraph(xysN=xys, edgesN=self.edgesN, size=camera.size)

    @property
    def num_xyzsN(self) -> int:
        return NUM_WHL2XYZS_SAMP

    def extract_xyzsN(self) -> np.ndarray:
        return xyzxyzN2xyzsN_samp(self._xyzxyzN)

    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        vol_last = self.volume
        self._xyzxyzN = xyzsN2xyzxyzN(xyzsN)
        vol_new = self.volume
        self._xyzxyzN *= np.power(vol_last / vol_new, 1 / 3)

    @property
    def edgesN(self) -> np.ndarray:
        return EDGES_CUBE

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        xyzsN = self.extract_xyzsN()
        xyzsN = xyzsN @ rotation.T + translation
        self.refrom_xyzsN(xyzsN)
        return self


class XYZWHLQBorder(Convertable, Point3DExtractable, Movable3D, HasVertexGraph3D, HasXYZXYZNFromHasXYZSN,
                    Projectable, MeasurableProjected, ClipableProjected, HasXYXYNProjected, HasVolume):
    REGISTER_COVERT = Register()
    __slots__ = ('_xyzwhlqN',)

    def __init__(self, xyzwhlqN: np.ndarray):
        self._xyzwhlqN = np.array(xyzwhlqN).astype(np.float32)

    @property
    def volume(self) -> float:
        return np.prod(self._xyzwhlqN[3:6])

    @property
    def xyzsN(self) -> np.ndarray:
        return xyzwhlqN2xyzsN_vert(self._xyzwhlqN)

    def xyxyN_projected(self, camera: MCamera) -> np.ndarray:
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        return xyxy

    def measure_projected(self, camera: MCamera) -> float:
        if self.volume == 0:
            return 0
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        xyxy = xyxyN_clip(xyxy, np.array(camera.size))
        return np.sqrt(np.prod(xyxy[2:4] - xyxy[:2]))

    def clip3d_(self, xyxyN_rgn: np.ndarray, camera: MCamera, **kwargs):
        xys = camera.project_xyzsN(self.xyzsN)
        xyxy = xysN2xyxyN(xys)
        if np.any(np.isnan(xys)) or np.any(xyxy[:2] > xyxyN_rgn[2:4]) or np.any(xyxy[2:4] < xyxyN_rgn[:2]):
            self._xyzwhlqN = np.zeros(10)
        return self

    def project(self, camera: MCamera):
        xys = camera.project_xyzsN(self.xyzsN)
        return XYSGraph(xysN=xys, edgesN=self.edgesN, size=camera.size)

    @property
    def num_xyzsN(self) -> int:
        return NUM_WHL2XYZS_SAMP

    def extract_xyzsN(self) -> np.ndarray:
        return xyzwhlqN2xyzsN_samp(self._xyzwhlqN)

    def refrom_xyzsN(self, xyzsN: np.ndarray, **kwargs):
        vol_last = self.volume
        self._xyzwhlqN = xyzsN2xyzwhlqN(xyzsN)
        vol_new = self.volume
        self._xyzwhlqN[:6] *= np.power(vol_last / vol_new, 1 / 3)

    @property
    def edgesN(self) -> np.ndarray:
        return EDGES_CUBE

    def transform_(self, rotation: np.ndarray = ROTATION_IDENTITY, translation: np.ndarray = TRANSLATION_IDENTITY):
        xyzsN = self.extract_xyzsN()
        xyzsN = xyzsN @ rotation.T + translation
        self.refrom_xyzsN(xyzsN)
        return self


# </editor-fold>

# <editor-fold desc='注册json变换'>
REGISTRY_JSON_ENCDEC_BY_INIT(MCamera)
REGISTRY_JSON_ENCDEC_BY_INIT(XYZSPoint)
REGISTRY_JSON_ENCDEC_BY_INIT(XYZSSurface)
REGISTRY_JSON_ENCDEC_BY_INIT(XYSSurface)
REGISTRY_JSON_ENCDEC_BY_INIT(XYZSColoredSurface)
REGISTRY_JSON_ENCDEC_BY_INIT(XYZWHLBorder)
REGISTRY_JSON_ENCDEC_BY_INIT(XYZXYZBorder)
REGISTRY_JSON_ENCDEC_BY_INIT(XYZWHLQBorder)


# </editor-fold>

# <editor-fold desc='物体采样'>
def xyzsN_surfsN2distsN(xyzsN: np.ndarray, surfsN: np.ndarray) -> np.ndarray:
    surf_xyzsN = xyzsN[surfsN]
    dists = np.linalg.norm(surf_xyzsN - np.roll(surf_xyzsN, axis=-2, shift=1), axis=-1)
    return dists


def xyzsN_surfsN2areasN(xyzsN: np.ndarray, surfsN: np.ndarray) -> np.ndarray:
    surf_xyzsN = xyzsN[surfsN]
    v1 = surf_xyzsN[..., 1, :] - surf_xyzsN[..., 0, :]
    v2 = surf_xyzsN[..., 2, :] - surf_xyzsN[..., 0, :]
    areas = np.cross(v1, v2) / 2
    areas = np.linalg.norm(areas, axis=-1)
    return areas


def surf_xyzsN_collapse(xyzsN: np.ndarray, surfsN: np.ndarray, num_vert: Optional[int] = None,
                        thres_dist_min: Optional[float] = None, thres_dist_aver: Optional[float] = None) \
        -> (np.ndarray, np.ndarray):
    xyzsN_cur = copy.deepcopy(xyzsN)
    surfsN_cur = copy.deepcopy(surfsN)
    dists = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur)
    num_vert_cur = xyzsN.shape[0]
    while True:
        if len(dists) == 0:
            break
        _id = np.argmin(dists.reshape(-1))
        idxf, idx1 = divmod(_id, dists.shape[-1])
        dist_min = dists[idxf, idx1]
        if num_vert is not None:
            if num_vert >= num_vert_cur:
                break
        elif thres_dist_min is not None:
            if dist_min >= thres_dist_min:
                break
        elif thres_dist_aver is not None:
            if np.mean(dists) >= thres_dist_aver:
                break
        elif dist_min >= np.mean(dists) / 2:
            break
        idx2 = (idx1 - 1) % dists.shape[-1]
        idxv1 = surfsN_cur[idxf, idx1]
        idxv2 = surfsN_cur[idxf, idx2]
        has_v1 = (surfsN_cur == idxv1)
        has_v2 = (surfsN_cur == idxv2)
        ftrf_presv = ~(np.any(has_v1, axis=1) * np.any(has_v2, axis=1))
        ftrf_chgd = np.any(has_v1, axis=1) ^ np.any(has_v2, axis=1)
        # 部分更新
        xyz_new = (xyzsN_cur[idxv1] + xyzsN_cur[idxv2]) / 2
        xyzsN_cur[idxv1] = xyz_new
        surfsN_cur[has_v2] = idxv1
        dists[ftrf_chgd] = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur[ftrf_chgd])
        num_vert_cur = num_vert_cur - 1
        # 部分删除
        surfsN_cur = surfsN_cur[ftrf_presv]
        dists = dists[ftrf_presv]

    surf_xyzsN = xyzsN_cur[surfsN_cur]
    xyzsN_cur, surfsN_cur = _cvsN2vsN_ivsN(surf_xyzsN)
    if xyzsN_cur.shape[0] < num_vert_cur:
        detla = num_vert_cur - xyzsN_cur.shape[0]
        xyzsN_cur = np.concatenate([xyzsN_cur, np.broadcast_to(np.mean(xyzsN_cur, axis=0), (detla, 3))], axis=0)
    return xyzsN_cur, surfsN_cur


def surf_xyzsN_collapse2(xyzsN: np.ndarray, surfsN: np.ndarray, num_vert: Optional[int] = None,
                         thres_dist_min: Optional[float] = None, thres_dist_aver: Optional[float] = None) \
        -> (np.ndarray, np.ndarray):
    xyzsN_cur = copy.deepcopy(xyzsN)
    surfsN_cur = copy.deepcopy(surfsN)
    dists = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur)
    num_vert_cur = xyzsN.shape[0]
    while True:
        if len(dists) == 0:
            break
        fnorms, vnorms = xyzsN_surfsN2fnormsN_vnormsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur)
        dist_scales = 1 + xyzsN_surfsN2distsN(xyzsN=vnorms, surfsN=surfsN_cur)
        dists_eq = dists * (dist_scales) ** 0.5
        _id = np.argmin(dists_eq.reshape(-1))
        idxf, idx1 = divmod(_id, dists.shape[-1])
        dist_min = dists[idxf, idx1]
        dist_eq_min = dists_eq[idxf, idx1]
        if num_vert is not None:
            if num_vert >= num_vert_cur:
                break
        elif thres_dist_min is not None:
            if dist_eq_min >= thres_dist_min:
                break
        elif thres_dist_aver is not None:
            if np.mean(dists) >= thres_dist_aver:
                break
        elif dist_eq_min >= np.mean(dists) / 2:
            break
        idx2 = (idx1 - 1) % dists.shape[-1]
        idxv1 = surfsN_cur[idxf, idx1]
        idxv2 = surfsN_cur[idxf, idx2]
        has_v1 = (surfsN_cur == idxv1)
        has_v2 = (surfsN_cur == idxv2)
        ftrf_presv = ~(np.any(has_v1, axis=1) * np.any(has_v2, axis=1))
        ftrf_chgd = np.any(has_v1, axis=1) ^ np.any(has_v2, axis=1)
        # 部分更新
        xyz_new = (xyzsN_cur[idxv1] + xyzsN_cur[idxv2]) / 2
        xyzsN_cur[idxv1] = xyz_new
        surfsN_cur[has_v2] = idxv1
        dists[ftrf_chgd] = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur[ftrf_chgd])
        num_vert_cur = num_vert_cur - 1
        # 部分删除
        surfsN_cur = surfsN_cur[ftrf_presv]
        dists = dists[ftrf_presv]

    surf_xyzsN = xyzsN_cur[surfsN_cur]
    xyzsN_cur, surfsN_cur = _cvsN2vsN_ivsN(surf_xyzsN)
    if xyzsN_cur.shape[0] < num_vert_cur:
        detla = num_vert_cur - xyzsN_cur.shape[0]
        xyzsN_cur = np.concatenate([xyzsN_cur, np.broadcast_to(np.mean(xyzsN_cur, axis=0), (detla, 3))], axis=0)
    return xyzsN_cur, surfsN_cur


# 按照面积裁剪
def surf_xyzsN_subdiv(xyzsN: np.ndarray, surfsN: np.ndarray, num_vert: Optional[int] = None,
                      thres_area_max: Optional[float] = None, thres_area_aver: Optional[float] = None) \
        -> (np.ndarray, np.ndarray):
    xyzsN_cur = copy.deepcopy(xyzsN)
    surfsN_cur = copy.deepcopy(surfsN)
    areas = xyzsN_surfsN2areasN(xyzsN=xyzsN_cur, surfsN=surfsN_cur)
    num_vert_cur = xyzsN.shape[0]
    while True:
        if len(areas) == 0:
            break
        idxf = np.argmax(areas)
        area_max = areas[idxf]
        idx1 = np.argmax(xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur[idxf]))

        if num_vert is not None:
            if num_vert >= num_vert_cur:
                break
        elif thres_area_max is not None:
            if area_max <= thres_area_max:
                break
        elif thres_area_aver is not None:
            if np.mean(areas) <= thres_area_aver:
                break
        elif area_max <= np.mean(areas) * 2:
            break

        idx2 = (idx1 - 1) % surfsN_cur.shape[-1]
        idxv1 = surfsN_cur[idxf, idx1]
        idxv2 = surfsN_cur[idxf, idx2]
        has_v1 = (surfsN_cur == idxv1)
        has_v2 = (surfsN_cur == idxv2)
        ftrf_chgd = np.any(has_v1, axis=1) * np.any(has_v2, axis=1)
        # 添加节点
        xyz_new = (xyzsN_cur[idxv1] + xyzsN_cur[idxv2]) / 2
        xyzsN_cur = np.concatenate([xyzsN_cur, xyz_new[None]], axis=0)
        # 添加面
        surfs_chgd = surfsN_cur[ftrf_chgd]
        surfs_new = copy.deepcopy(surfs_chgd)
        surfs_new[surfs_new == idxv2] = num_vert_cur
        surfs_chgd[surfs_new == idxv1] = num_vert_cur
        surfsN_cur[ftrf_chgd] = surfs_chgd
        surfsN_cur = np.concatenate([surfsN_cur, surfs_new], axis=0)
        # 添加距离
        areas_new = xyzsN_surfsN2areasN(xyzsN=xyzsN_cur, surfsN=surfs_new)
        areas[ftrf_chgd] = xyzsN_surfsN2areasN(xyzsN=xyzsN_cur, surfsN=surfs_chgd)
        areas = np.concatenate([areas, areas_new], axis=0)
        # 更新
        num_vert_cur = num_vert_cur + 1
    return xyzsN_cur, surfsN_cur


# 按照边长裁剪
def surf_xyzsN_subdiv2(xyzsN: np.ndarray, surfsN: np.ndarray, num_vert: Optional[int] = None,
                       thres_dist_max: Optional[float] = None, thres_dist_aver: Optional[float] = None) \
        -> (np.ndarray, np.ndarray):
    xyzsN_cur = copy.deepcopy(xyzsN)
    surfsN_cur = copy.deepcopy(surfsN)
    dists = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfsN_cur)
    num_vert_cur = xyzsN.shape[0]
    while True:
        if len(dists) == 0:
            break
        _id = np.argmax(dists.reshape(-1))
        idxf, idx1 = divmod(_id, dists.shape[-1])
        dist_max = dists[idxf, idx1]

        if num_vert is not None:
            if num_vert <= num_vert_cur:
                break
        elif thres_dist_max is not None:
            if dist_max <= thres_dist_max:
                break
        elif thres_dist_aver is not None:
            if np.mean(dists) <= thres_dist_aver:
                break
        elif dist_max <= np.mean(dists) * 2:
            break

        idx2 = (idx1 - 1) % surfsN_cur.shape[-1]
        idxv1 = surfsN_cur[idxf, idx1]
        idxv2 = surfsN_cur[idxf, idx2]
        has_v1 = (surfsN_cur == idxv1)
        has_v2 = (surfsN_cur == idxv2)
        ftrf_chgd = np.any(has_v1, axis=1) * np.any(has_v2, axis=1)
        # 添加节点
        xyz_new = (xyzsN_cur[idxv1] + xyzsN_cur[idxv2]) / 2
        xyzsN_cur = np.concatenate([xyzsN_cur, xyz_new[None]], axis=0)
        # 添加面
        surfs_chgd = surfsN_cur[ftrf_chgd]
        surfs_new = copy.deepcopy(surfs_chgd)
        surfs_new[surfs_new == idxv2] = num_vert_cur
        surfs_chgd[surfs_new == idxv1] = num_vert_cur
        surfsN_cur[ftrf_chgd] = surfs_chgd
        surfsN_cur = np.concatenate([surfsN_cur, surfs_new], axis=0)
        # 添加距离
        dists_new = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfs_new)
        dists[ftrf_chgd] = xyzsN_surfsN2distsN(xyzsN=xyzsN_cur, surfsN=surfs_chgd)
        dists = np.concatenate([dists, dists_new], axis=0)
        # 更新
        num_vert_cur = num_vert_cur + 1

    return xyzsN_cur, surfsN_cur


# </editor-fold>

# <editor-fold desc='物体加载'>

class MTLTexture():
    def __init__(self, nsN: NV_Flt = 250.0, niN: NV_Flt = 1.45, dN: NV_Flt = 1.0, illumN: NV_Flt = 2,
                 kaN: SN_Flt = (1.0, 1.0, 1.0), ksN: SN_Flt = (0.5, 0.5, 0.5), kdN: SN_Flt = (0.8, 0.8, 0.8),
                 keN: SN_Flt = (0, 0, 0), uvmapN: Optional[np.ndarray] = None, name: str = 'tex'):
        self._nsN = np.array(nsN)
        self._kaN = np.array(kaN)
        self._ksN = np.array(ksN)
        self._kdN = np.array(kdN)
        self._keN = np.array(keN)
        self._niN = np.array(niN)
        self._dN = np.array(dN)
        self._illumN = np.array(illumN)
        self._uvmapN = np.array(uvmapN) if uvmapN is not None else None
        self._name = name

    @property
    def uvmapN(self) -> np.ndarray:
        return self._uvmapN

    @property
    def name(self) -> str:
        return self._name

    def samp(self, iuvsN: np.ndarray) -> np.ndarray:
        if self._uvmapN is None:
            return np.full(shape=list(iuvsN.shape)[:-1] + [3], fill_value=128)
        iuvsN = iuvsN * np.array([self._uvmapN.shape[0], self._uvmapN.shape[1]])
        iuvsN[..., 1] = self._uvmapN.shape[1] - iuvsN[..., 1]
        iuvsN = iuvsN.astype(np.int32)
        cols = self._uvmapN[iuvsN[..., 1], iuvsN[..., 0]]
        return cols


def load_mtl(mtl_pth: str, extend: str = 'mtl') -> Dict[str, MTLTexture]:
    mtl_pth = ensure_extend(mtl_pth, extend=extend, overwrite=False)
    lines = load_txt(mtl_pth)
    tex_dct = {}
    kwargs = {}
    for line in lines:
        line = line.split('#')[0].strip()
        if len(line) == 0:
            continue
        pieces = line.split(' ')
        if pieces[0] == 'newmtl':
            if len(kwargs) > 0:
                tex = MTLTexture(**kwargs)
                tex_dct[tex.name] = tex
                kwargs = {}
            kwargs['name'] = pieces[1]
        elif pieces[0] == 'map_Kd':
            map_pth = os.path.join(os.path.dirname(mtl_pth), pieces[1])
            kwargs['uvmapN'] = load_img_cv2(map_pth) if os.path.exists(map_pth) else None
        else:
            kwargs[pieces[0].lower() + 'N'] = np.array([float(p) for p in pieces[1:] if len(p) > 0])
    if len(kwargs) > 0:
        tex = MTLTexture(**kwargs)
        tex_dct[tex.name] = tex
    return tex_dct


def _load_wfobj_arr(wobj_pth: str, extend: str = 'obj', with_mtl: bool = True):
    wobj_pth = ensure_extend(wobj_pth, extend=extend, overwrite=False)
    lines = load_txt(wobj_pth)
    xyzsN = []
    uvsN = []
    normsN = []
    surfsN = []
    surfsN_uv = []
    surfsN_nm = []
    itexsN = []
    tex_dct = None
    tex_names = []
    tex_index = -1
    for line in lines:
        line = line.split('#')[0].strip()
        if len(line) == 0:
            continue
        pieces = line.split(' ')
        if with_mtl and pieces[0] == 'mtllib':
            mtl_pth = os.path.join(os.path.dirname(wobj_pth), pieces[1])
            if os.path.exists(mtl_pth):
                tex_dct = load_mtl(mtl_pth)
        elif pieces[0] == 'usemtl':
            tex_name = pieces[1]
            tex_names.append(tex_name)
            tex_index = tex_index + 1
        elif pieces[0] == 'v':
            xyzsN.append([float(x) for x in pieces[1:4]])
        elif pieces[0] == 'vt':
            uvsN.append([float(x) for x in pieces[1:3]])
        elif pieces[0] == 'vn':
            normsN.append([float(x) for x in pieces[1:4]])
        elif pieces[0] == 'f':
            ifN = []
            iuvN = []
            inN = []
            for iv_it_in in pieces[1:4]:
                iv_it_in = iv_it_in.split('/')
                if len(iv_it_in) > 1:
                    ifN.append(int(iv_it_in[0]))
                if len(iv_it_in) > 2 and len(iv_it_in[1]) > 0:
                    iuvN.append(int(iv_it_in[1]))
                if len(iv_it_in) > 3 and len(iv_it_in[2]) > 0:
                    inN.append(int(iv_it_in[2]))
            surfsN.append(ifN)
            surfsN_uv.append(iuvN)
            surfsN_nm.append(inN)
            if tex_index >= 0:
                itexsN.append(tex_index)
    xyzsN = np.array(xyzsN)
    normsN = np.array(normsN)
    uvsN = np.array(uvsN) if len(uvsN) > 0 else None
    itexsN = np.array(itexsN).astype(np.int32)
    surfsN = np.array(surfsN).astype(np.int32) - 1
    surfsN_uv = np.array(surfsN_uv).astype(np.int32) - 1
    surfsN_nm = np.array(surfsN_nm).astype(np.int32) - 1
    return xyzsN, normsN, uvsN, itexsN, tex_names, tex_dct, surfsN, surfsN_uv, surfsN_nm


def _build_ivtexsN(num_vert: int, itexsN: np.ndarray, surfsN: np.ndarray, ):
    ivtexsN = np.full(shape=(num_vert,), fill_value=0, dtype=np.int32)
    for itex in range(np.max(itexsN) + 1):
        fltr_itex = (itex == itexsN)
        ivtexsN[surfsN[fltr_itex].reshape(-1)] = itex
    return ivtexsN


def _build_vcolorsN(num_vert: int, itexsN: np.ndarray, texs: Optional[Sequence[MTLTexture]],
                    surfsN: Optional[np.ndarray], surfsN_uv: Optional[np.ndarray]) -> np.ndarray:
    if texs is None:
        vcolorsN = np.full(shape=(num_vert, 3), fill_value=230)
    elif np.prod(surfsN_uv.shape) > 0:
        vcolorsN = np.full(shape=(num_vert, 3), fill_value=0.0)
        for itex, tex in enumerate(texs):
            fltr_itex = (itex == itexsN)
            col_tex = tex.samp(surfsN_uv[fltr_itex])
            col_tex = col_tex.reshape(-1, 3)
            vcolorsN[surfsN[fltr_itex].reshape(-1)] = col_tex
    else:
        vcolorsN = np.full(shape=(num_vert, 3), fill_value=0.0)
        counter = np.full(shape=(num_vert,), fill_value=0.0)
        for itex, tex in enumerate(texs):
            fltr_itex = (itex == itexsN)
            idx_vert = surfsN[fltr_itex].reshape(-1)
            idx_rgb = np.tile(np.arange(3), idx_vert.shape[0])
            rgb = np.tile(tex._kdN * 255, idx_vert.shape[0])
            np.add.at(counter, idx_vert, 1.0)
            np.add.at(vcolorsN, (np.repeat(idx_vert, 3), idx_rgb), rgb)
        vcolorsN = vcolorsN / np.clip(counter[:, None], a_min=1, a_max=None)
        vcolorsN = np.clip(vcolorsN, a_min=0, a_max=255)  # 此处有点问题
    return vcolorsN


def load_wfobj(obj_pth: str, extend: str = 'obj', with_mtl: bool = True) -> Union[XYZSColoredSurface, XYZSSurface]:
    xyzsN, normsN, uvsN, itexsN, tex_names, tex_dct, surfsN, surfsN_uv, surfsN_nm \
        = _load_wfobj_arr(wobj_pth=obj_pth, extend=extend, with_mtl=with_mtl)

    if with_mtl and tex_dct is not None:
        texs = [tex_dct[name] for name in tex_names]
        vcolorsN = _build_vcolorsN(num_vert=xyzsN.shape[0], itexsN=itexsN, texs=texs, surfsN=surfsN,
                                   surfsN_uv=surfsN_uv)
        return XYZSColoredSurface(xyzsN=xyzsN, surfsN=surfsN, vcolorsN=vcolorsN)
    else:
        return XYZSSurface(xyzsN=xyzsN, surfsN=surfsN)


# </editor-fold>


def xyzsN_surfsN2fnormsN(xyzsN: np.ndarray, surfsN: np.ndarray) -> np.ndarray:
    vertices_faces = xyzsN[surfsN]
    fnorms = np.cross(
        vertices_faces[..., 2, :] - vertices_faces[..., 1, :],
        vertices_faces[..., 0, :] - vertices_faces[..., 1, :],
        axis=-1,
    )
    fnorms = fnorms / np.clip(np.linalg.norm(fnorms, axis=-1, keepdims=True), a_min=1e-7, a_max=None)
    return fnorms


def xyzsN_surfsN2fnormsN_vnormsN(xyzsN: np.ndarray, surfsN: np.ndarray) -> (np.ndarray, np.ndarray):
    vertices_faces = xyzsN[surfsN]
    fnorms = np.cross(
        vertices_faces[..., 2, :] - vertices_faces[..., 1, :],
        vertices_faces[..., 0, :] - vertices_faces[..., 1, :],
        axis=-1,
    )
    fnorms = fnorms / np.clip(np.linalg.norm(fnorms, axis=-1, keepdims=True), a_min=1e-7, a_max=None)
    num_vert = xyzsN.shape[0]
    num_surf = surfsN.shape[0]

    counter = np.full(shape=(num_vert,), fill_value=0.0)
    np.add.at(counter, surfsN.reshape(-1), 1.0)

    fshape = (num_surf, 3, 3)
    idx_nm = np.broadcast_to(np.arange(3)[None, :, None], fshape).reshape(-1)
    idx_vert = np.broadcast_to(surfsN[:, :, None], fshape).reshape(-1)
    nm = np.broadcast_to(fnorms[:, None, :], fshape).reshape(-1)
    vnorms = np.full(shape=(num_vert, 3), fill_value=0.0)
    np.add.at(vnorms, (idx_vert, idx_nm), nm)

    vnorms = vnorms / np.clip(np.linalg.norm(vnorms, axis=-1, keepdims=True), a_min=1e-7, a_max=None)
    return fnorms, vnorms


def xyzsT_surfsT2fnormsT(xyzsT: torch.Tensor, surfsT: torch.Tensor) -> torch.Tensor:
    xyz_size = list(xyzsT.size())[:-1] + [3, 3]
    surf_size = list(surfsT.size())[:-1] + [3, 3]
    xyzs_exp = xyzsT[..., None, :].expand(xyz_size)
    surfs_exp = surfsT[..., None].expand(surf_size)
    vertices_faces = torch.gather(xyzs_exp, dim=-3, index=surfs_exp)
    fnorms = torch.cross(
        vertices_faces[..., 2, :] - vertices_faces[..., 1, :],
        vertices_faces[..., 0, :] - vertices_faces[..., 1, :],
        dim=-1,
    )
    return fnorms


def xyzsT_surfsT2fcensT_fnormsT_vnormsT(xyzsT: torch.Tensor, surfsT: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    xyz_size = list(xyzsT.size())[:-1] + [3, 3]
    surf_size = list(surfsT.size())[:-1] + [3, 3]
    xyzs_exp = xyzsT[..., None, :].expand(xyz_size)
    surfs_exp = surfsT[..., None].expand(surf_size)
    surf_xyzs = torch.gather(xyzs_exp, dim=-3, index=surfs_exp)
    fcens = torch.mean(surf_xyzs, dim=-2)
    fnorms = torch.cross(
        surf_xyzs[..., 2, :] - surf_xyzs[..., 1, :],
        surf_xyzs[..., 0, :] - surf_xyzs[..., 1, :],
        dim=-1,
    )
    vnorms = torch.zeros_like(xyzsT)
    vnorms.scatter_add_(dim=-2, index=surfs_exp[..., 0, :], src=fnorms)
    vnorms.scatter_add_(dim=-2, index=surfs_exp[..., 1, :], src=fnorms)
    vnorms.scatter_add_(dim=-2, index=surfs_exp[..., 2, :], src=fnorms)
    vnorms = F.normalize(vnorms, dim=-1, eps=1e-6)
    return fcens, fnorms, vnorms


# <editor-fold desc='物体转换'>


@XYZSGraph.REGISTER_COVERT.registry(XYZWHLQBorder)
def _sbox3d2vgraph3d(sbox3d: XYZWHLQBorder):
    return XYZSGraph(sbox3d.xyzsN, edgesN=sbox3d.edgesN)


def csurfs2meshesT_parallel(csurfs: Union[XYZSColoredSurface, Sequence[XYZSColoredSurface]],
                            device: torch.device = DEVICE) -> Meshes:
    if not isinstance(csurfs, Sequence):
        csurfs = [csurfs]
    meshes = []
    for csurf in csurfs:
        xyz = arrsN2arrsT(csurf.xyzsN, device=device)
        surf = arrsN2arrsT(csurf.surfsN, device=device)
        color = arrsN2arrsT(csurf.vcolorsN, device=device) / 255
        texture = TexturesVertex(color[None])
        mesh = Meshes(verts=xyz[None], faces=surf[None], textures=texture)
        meshes.append(mesh)
    meshes = join_meshes_as_batch(meshes)
    return meshes


def csurfs2meshesT_concat(csurfs: Union[XYZSColoredSurface, Sequence[XYZSColoredSurface]],
                          device: torch.device = DEVICE) -> Meshes:
    if not isinstance(csurfs, Sequence):
        csurfs = [csurfs]
    xyzs = []
    surfs = []
    colors = []
    offset = 0
    for csurf in csurfs:
        xyz = arrsN2arrsT(csurf.xyzsN, device=device)
        surf = arrsN2arrsT(csurf.surfsN, device=device)
        color = arrsN2arrsT(csurf.vcolorsN, device=device) / 255
        xyzs.append(xyz)
        surfs.append(surf + offset)
        colors.append(color)
        offset = offset + xyz.size(0)
    xyzs = torch.cat(xyzs, dim=0)
    surfs = torch.cat(surfs, dim=0)
    colors = torch.cat(colors, dim=0)
    texture = TexturesVertex(colors[None])
    mesh = Meshes(verts=xyzs[None], faces=surfs[None], textures=texture)
    return mesh


# </editor-fold>

# <editor-fold desc='相机转换'>

_CAMT_R = torch.Tensor([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
])
_CAM_T = torch.Tensor([0, 0, 0])


def cameras2camerasT(cameras: Union[MCamera, Sequence[MCamera]],
                     device: torch.device = DEVICE) -> PerspectiveCameras:
    if not isinstance(cameras, Sequence):
        cameras = [cameras]
    intrs = []
    img_sizes_inv = []
    for i, camera in enumerate(cameras):
        # intr = copy.deepcopy(camera.intrinsicN)
        intrs.append(camera.intrinsicN)
        img_sizes_inv.append((camera.size[1], camera.size[0]))
    intrs = torch.from_numpy(np.stack(intrs, axis=0, )).to(device)
    K = torch.zeros(size=(len(cameras), 4, 4), device=device)
    K[:, 3, 2] = 1
    K[:, :3, :3] = intrs
    R = _CAMT_R[None].to(device).expand(len(cameras), 3, 3)
    T = _CAM_T[None].to(device).expand(len(cameras), 3)
    return PerspectiveCameras(R=R, T=T, K=K, image_size=img_sizes_inv, in_ndc=False)


def focalsT_biassT_sizesT2camerasT(Nb: int, focalsT: torch.Tensor, biassT: torch.Tensor, sizesT: torch.Tensor) \
        -> PerspectiveCameras:
    focalsT = focalsT.expand(Nb, 2)
    biassT = biassT.expand(Nb, 2)
    sizesT = sizesT.expand(Nb, 2)
    R = _CAMT_R[None].to(focalsT.device)
    T = _CAM_T[None].to(focalsT.device)
    focalsT_m = focalsT[..., None] * torch.eye(2, device=focalsT.device)
    K1 = torch.cat([focalsT_m, torch.zeros(size=(Nb, 2, 2), device=focalsT.device)], dim=-2)
    K2 = torch.cat([biassT, torch.ones(size=(Nb, 2), device=focalsT.device)], dim=-1)[..., None]
    K3 = torch.zeros(size=(Nb, 4, 1), device=focalsT.device)
    K = torch.cat([K1, K2, K3], dim=-1)
    sizesRT = torch.flip(sizesT, dims=(-1,))
    return PerspectiveCameras(R=R, T=T, K=K, image_size=sizesRT, in_ndc=False)


def rand_lightsT(Nb: int = 1, device=DEVICE):
    direction = torch.rand(Nb, 3, device=device) * torch.Tensor([2, 2, 1]).to(device) \
                + torch.Tensor([-1, -1, -1]).to(device)
    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
    ligs_amb_gn = torch.rand(Nb, 1, device=device) * 0.3 + 0.3
    # ligs_amb_gn = ligs_amb_gn.clip(min=0, max=1)
    ligs_dif_gn = torch.rand(Nb, 1, device=device) * 2.0 + 1.0
    # ligs_dif_gn = ligs_dif_gn.clip(min=0, max=1)
    ligs_dif_gn = ligs_dif_gn.expand(Nb, 3)
    ligs_amb_gn = ligs_amb_gn.expand(Nb, 3)
    lights = DirectionalLights(direction=direction, diffuse_color=ligs_dif_gn, ambient_color=ligs_amb_gn)
    return lights
# </editor-fold>
