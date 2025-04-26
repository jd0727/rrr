import numpy as np
import torch

try:
    import pytorch3d.transforms as py3dtrans
except Exception as e:
    pass
from .borders import *

# <editor-fold desc='numpy 3d边界'>
##################################################################################
#  ^ z
#   \
#    1----------2
#    |\       / |
#    | 0------3-+----> y
#    | |      | |
#    | 7------4 |
#    |/|      \ |
#    6-+--------5
#      |
#      v x
##################################################################################
_SIGN_WHL2XYZS_VERT_N = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
                                  [1, 1, -1], [1, 1, 1], [1, -1, 1], [1, -1, -1]])
_SIGN_WHL2XYZS_SURFCEN_N = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                                     [0, 0, 1], [0, 0, -1]])
_SIGN_WHL2XYZS_SAMP_N = np.concatenate([_SIGN_WHL2XYZS_SURFCEN_N, _SIGN_WHL2XYZS_VERT_N / np.sqrt(3)], axis=0)
NUM_WHL2XYZS_SAMP = _SIGN_WHL2XYZS_SAMP_N.shape[0]
EDGES_CUBE = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                       [0, 7], [1, 6], [2, 5], [3, 4],
                       [4, 5], [5, 6], [6, 7], [7, 4]])

_IDX_XYZXYZ2XL = np.array((0, 0, 0, 0, 3, 3, 3, 3))
_IDX_XYZXYZ2YL = np.array((1, 1, 4, 4, 4, 4, 1, 1))
_IDX_XYZXYZ2ZL = np.array((2, 5, 5, 2, 2, 5, 5, 2))
_IDX_CUBE_AJCENT = np.array([[7, 3, 1], [0, 2, 6], [5, 1, 3], [2, 0, 4],
                             [7, 5, 3], [4, 0, 6], [5, 7, 1], [4, 0, 6]])

_IDX_CUBE_WDIR = np.array([[1, 6], [2, 5], [3, 4], [0, 7]])
_IDX_CUBE_HDIR = np.array([[0, 3], [7, 4], [1, 2], [6, 5]])
_IDX_CUBE_LDIR = np.array([[0, 1], [3, 2], [4, 5], [7, 6]])
_IDX_CUBE_WHLDIR = np.stack([_IDX_CUBE_WDIR, _IDX_CUBE_HDIR, _IDX_CUBE_LDIR], axis=0)

_IDX_CUBE_SURF = np.array([[0, 1, 2, 3], [3, 2, 5, 4],
                           [4, 5, 6, 7], [6, 1, 0, 7],
                           [0, 3, 4, 7], [1, 6, 5, 2]])

_IDX_CUBE_EDGE = np.concatenate([_IDX_CUBE_WDIR, _IDX_CUBE_HDIR, _IDX_CUBE_LDIR], axis=0)

##################################################################################


xyzdwhlN2xyzxyzN = cordsizeN2cordcordN
xyzdwhlN2xyzwhlN = cordsizeN2ccrdsizeN
xyzwhlN2xyzxyzN = ccrdsizeN2cordcordN
xyzxyzN2xyzwhlN = cordcordN2ccrdsizeN
xyzxyzN2volumeN = cordcordN2cpctN
xyzwhlN2volumeN = ccrdsizeN2cpctN
xyzsN2xyzxyzN = cordsN2cordcordN
xyzsN2xyzwhlN = cordsN2ccrdsizeN
xypN_samp = cordsN_samp
xyzsN2rot3N = cordsN2rotN

xyzxyzN2linear = cordcordN2linear
xyzN_linear = cordN_linear
xyzN_perspective = cordN_perspective


def xyzwhlqN2xyzsN_vert(xyzwhlqN: np.ndarray) -> np.ndarray:
    xyz, whl, quater = np.split(xyzwhlqN, (3, 6), axis=-1)
    return xyz + (_SIGN_WHL2XYZS_VERT_N * whl / 2) @ np.swapaxes(quaterN2rot3N(quater), axis1=-1, axis2=-2)


def xyzwhlqN2xyzsN_samp(xyzwhlqN: np.ndarray) -> np.ndarray:
    xyz, whl, quater = np.split(xyzwhlqN, (3, 6), axis=-1)
    return xyz + (_SIGN_WHL2XYZS_SAMP_N * whl / 2) @ np.swapaxes(quaterN2rot3N(quater), axis1=-1, axis2=-2)


def xyzxyzN2xyzsN_vert(xyzxyzN: np.ndarray) -> np.ndarray:
    xyps = np.stack([xyzxyzN[..., _IDX_XYZXYZ2XL],
                     xyzxyzN[..., _IDX_XYZXYZ2YL], xyzxyzN[..., _IDX_XYZXYZ2ZL]], axis=-1)
    return xyps


def xyzxyzN2xyzsN_surfcen(xyzxyzN: np.ndarray) -> np.ndarray:
    return xyzwhlN2xyzsN_surfcen(xyzxyzN2xyzwhlN(xyzxyzN))


def xyzwhlN2xyzsN_surfcen(xyzwhlN: np.ndarray) -> np.ndarray:
    return xyzwhlN[..., None, :3] + _SIGN_WHL2XYZS_SURFCEN_N * xyzwhlN[..., None, 3:6] / 2


def whlN2xyzsN_surfcen(whlN: np.ndarray) -> np.ndarray:
    return _SIGN_WHL2XYZS_SURFCEN_N * whlN[..., None, :] / 2


def whlN2xyzsN_vert(whlN: np.ndarray) -> np.ndarray:
    return _SIGN_WHL2XYZS_VERT_N * whlN[..., None, :] / 2


def xyzxyzN2xyzsN_samp(xyzxyzN: np.ndarray) -> np.ndarray:
    return xyzwhlN2xyzsN_samp(xyzxyzN2xyzwhlN(xyzxyzN))


def xyzwhlN2xyzsN_samp(xyzwhlN: np.ndarray) -> np.ndarray:
    return xyzwhlN[..., None, :3] + _SIGN_WHL2XYZS_SAMP_N * xyzwhlN[..., None, 3:6] / 2


def whlN2xyzsN_samp(whlN: np.ndarray) -> np.ndarray:
    return _SIGN_WHL2XYZS_SAMP_N * whlN / 2


def xyzwhlN2xyzsN_vert(xyzwhlN: np.ndarray) -> np.ndarray:
    return xyzwhlN[..., None, :3] + _SIGN_WHL2XYZS_VERT_N * xyzwhlN[..., None, 3:6] / 2


def xyzsN2xyzwhlqN(xyzsN: np.ndarray) -> (np.ndarray):
    if xyzsN.shape[-2] <= 4:
        return np.zeros(shape=list(xyzsN.shape[:-2]) + [10])
    rot3 = xyzsN2rot3N(xyzsN)
    xyz_proj = xyzsN @ rot3
    xyzwhl_proj = xyzsN2xyzwhlN(xyz_proj)
    xyz_cen = np.sum(rot3 * xyzwhl_proj[..., None, :3], axis=-1)
    quaterN = rot3N2quaterN(rot3)
    return np.concatenate([xyz_cen, xyzwhl_proj[..., 3:6], quaterN], axis=-1)


def xyzsN_rot3N2xyzN_whlN(xyzsN: np.ndarray, rot3N: np.ndarray) -> (np.ndarray, np.ndarray):
    xyzs_proj = xyzsN @ rot3N
    xyzwhl_proj = xyzsN2xyzwhlN(xyzs_proj)
    xyz_cen = xyzwhl_proj[:3] @ rot3N.T
    return xyz_cen, xyzwhl_proj[3:6]


def xyzsN_quaterN2xyzN_whlN(xyzsN: np.ndarray, quaterN: np.ndarray) -> (np.ndarray, np.ndarray):
    if xyzsN.shape[0] <= 1:
        return np.zeros(shape=6)
    rot3N = quaterN2rot3N(quaterN)
    return xyzsN_rot3N2xyzN_whlN(xyzsN, rot3N)


# </editor-fold>

# <editor-fold desc='numpy 四元数'>
# def _quaterN2mat(quaterN: np.ndarray) -> np.ndarray:
#     return np.stack([
#         np.stack([quaterN[..., 0], -quaterN[..., 1], -quaterN[..., 2], -quaterN[..., 3]], axis=-1),
#         np.stack([quaterN[..., 1], quaterN[..., 0], -quaterN[..., 3], quaterN[..., 2]], axis=-1),
#         np.stack([quaterN[..., 2], quaterN[..., 3], quaterN[..., 0], -quaterN[..., 1]], axis=-1),
#         np.stack([quaterN[..., 3], -quaterN[..., 2], quaterN[..., 1], quaterN[..., 0]], axis=-1), ], axis=-2)
#
#
# def _quaterN2mat_bar(quaterN: np.ndarray) -> np.ndarray:
#     return np.stack([
#         np.stack([quaterN[..., 0], -quaterN[..., 1], -quaterN[..., 2], -quaterN[..., 3]], axis=-1),
#         np.stack([quaterN[..., 1], quaterN[..., 0], quaterN[..., 3], -quaterN[..., 2]], axis=-1),
#         np.stack([quaterN[..., 2], -quaterN[..., 3], quaterN[..., 0], quaterN[..., 1]], axis=-1),
#         np.stack([quaterN[..., 3], quaterN[..., 2], -quaterN[..., 1], quaterN[..., 0]], axis=-1), ], axis=-2)


def quaterN_conj(quaterN: np.ndarray) -> np.ndarray:
    return np.concatenate([quaterN[:1], -quaterN[1:]], axis=0)


def quaterN_inv(quaterN: np.ndarray) -> np.ndarray:
    return quaterN_conj(quaterN) / np.linalg.norm(quaterN)


def quaterN_mul(quaterN1: np.ndarray, quaterN2: np.ndarray) -> np.ndarray:
    q1_0, q1_1, q1_2, q1_3 = np.split(quaterN2, 4, axis=-1)
    q2_0, q2_1, q2_2, q2_3 = np.split(quaterN1, 4, axis=-1)  # 故意交换
    q_0 = q1_0 * q2_0 - q1_1 * q2_1 - q1_2 * q2_2 - q1_3 * q2_3
    q_1 = q1_1 * q2_0 + q1_0 * q2_1 + q1_3 * q2_2 - q1_2 * q2_3
    q_2 = q1_2 * q2_0 - q1_3 * q2_1 + q1_0 * q2_2 + q1_1 * q2_3
    q_3 = q1_3 * q2_0 + q1_2 * q2_1 - q1_1 * q2_2 + q1_0 * q2_3
    return np.concatenate([q_0, q_1, q_2, q_3], axis=-1)


def quaterN2rot3N(quaterN: np.ndarray) -> np.ndarray:
    outer = quaterN[..., None] * quaterN[..., None, :]
    mat = np.stack(
        [np.stack([2 * (outer[..., 0, 0] + outer[..., 1, 1]) - 1,
                   2 * (outer[..., 1, 2] - outer[..., 0, 3]),
                   2 * (outer[..., 1, 3] + outer[..., 0, 2])],
                  axis=-1),
         np.stack([2 * (outer[..., 1, 2] + outer[..., 0, 3]),
                   2 * (outer[..., 0, 0] + outer[..., 2, 2]) - 1,
                   2 * (outer[..., 2, 3] - outer[..., 0, 1])],
                  axis=-1),
         np.stack([2 * (outer[..., 1, 3] - outer[..., 0, 2]),
                   2 * (outer[..., 2, 3] + outer[..., 0, 1]),
                   2 * (outer[..., 0, 0] + outer[..., 3, 3]) - 1],
                  axis=-1)],
        axis=-2)
    return mat


_ROT3_FLAT2K3_FLAT = np.array([
    # 11 12 13 21 22 23 31 32 33
    [1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, -1, 0],
    [0, 0, -1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, -1, 0, 0, 0, 0, 0],

    # 11 12 13 21 22 23 31 32 33
    [0, 0, 0, 0, 0, 1, 0, -1, 0],
    [1, 0, 0, 0, -1, 0, 0, 0, -1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],

    # 11 12 13 21 22 23 31 32 33
    [0, 0, -1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 1, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],

    # 11 12 13 21 22 23 31 32 33
    [0, 1, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [-1, 0, 0, 0, -1, 0, 0, 0, 1],
]).T


def rot3N2quaterN(rot3N: np.ndarray) -> np.ndarray:
    rot3N = rot3N.swapaxes(-1, -2)
    rot3N_flat = rot3N.reshape(list(rot3N.shape[:-2]) + [9])
    K3_flat = rot3N_flat @ _ROT3_FLAT2K3_FLAT
    K3 = K3_flat.reshape(list(rot3N.shape[:-2]) + [4, 4])
    evals, evec = np.linalg.eigh(K3)
    # power = (evals[..., None, :] + 1) / 4
    power = (evals == np.max(evals, axis=-1, keepdims=True))[..., None, :]
    quats = np.sum(power * evec, axis=-1)
    return quats


#
# def rot3N2quaterN(rot3N: np.ndarray) -> np.ndarray:
#     """
#     This code uses a modification of the algorithm described in:
#     https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
#     which is itself based on the method described here:
#     http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
#
#     Altered to work with the column vector convention instead of row vectors
#     """
#     m = rot3N.conj().transpose()
#     if m[2, 2] < 0:
#         if m[0, 0] > m[1, 1]:
#             t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
#             q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
#         else:
#             t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
#             q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
#     else:
#         if m[0, 0] < -m[1, 1]:
#             t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
#             q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
#         else:
#             t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
#             q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]
#
#     q = np.array(q).astype('float64')
#     q *= 0.5 / np.sqrt(t)
#     return q


# </editor-fold>


# <editor-fold desc='torch 3d边界'>

xyzdwhlT2xyzxyzT = cordsizeT2cordcordT
xyzdwhlT2xyzwhlT = cordsizeT2ccrdsizeT
xyzwhlT2xyzxyzT = ccrdsizeT2cordcordT
xyzxyzT2xyzwhlT = cordcordT2ccrdsizeT
xyzxyzT2areaT = cordcordT2cpctT
xyzwhlT2areaT = ccrdsizeT2cpctT
xyzsT2xyzxyzT = cordsT2cordcordT
xypT_samp = cordsT_samp
xyzsT2rot2T = cordsT2rotT

xyzsT_linear = cordsT_linear
xyzsT_perspective = cordsT_perspective


def xyzxyzT2xyzpT(xyzxyzT: torch.Tensor) -> torch.Tensor:
    xyzp = torch.stack([xyzxyzT[..., _IDX_XYZXYZ2XL],
                        xyzxyzT[..., _IDX_XYZXYZ2YL], xyzxyzT[..., _IDX_XYZXYZ2ZL]], dim=-1)
    return xyzp


# </editor-fold>


# <editor-fold desc='torch 四元数'>


def quaterT_inv(quaterT: torch.Tensor) -> torch.Tensor:
    return py3dtrans.quaternion_invert(quaterT)


def quaterT_mul(quaterT1: torch.Tensor, quaterT2: torch.Tensor) -> torch.Tensor:
    return py3dtrans.quaternion_raw_multiply(quaterT1, quaterT2)


def quaterT2rot3T(quaterT: torch.Tensor) -> torch.Tensor:
    return py3dtrans.quaternion_to_matrix(quaterT)


def rot3T2quaterT(rot3T: torch.Tensor) -> torch.Tensor:
    return py3dtrans.matrix_to_quaternion(rot3T)

# </editor-fold>


# <editor-fold desc='其它'>


# </editor-fold>
