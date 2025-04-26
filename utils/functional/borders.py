from .cvting import *

try:
    from shapely.geometry import Polygon, GeometryCollection
except Exception as e:
    pass


# <editor-fold desc='numpy通用转换'>
def cordsizeN2cordcordN(cordsizeN: np.ndarray, axis: int = -1) -> np.ndarray:
    cord, size = np.split(cordsizeN, 2, axis=axis)
    return np.concatenate([cord, cord + size], axis=axis)


def cordsizeN2ccrdsizeN(cordsizeN: np.ndarray, axis: int = -1) -> np.ndarray:
    cord, size = np.split(cordsizeN, 2, axis=axis)
    return np.concatenate([cord + size / 2, size], axis=axis)


def ccrdsizeN2cordcordN(ccrdsizeN: np.ndarray, axis: int = -1) -> np.ndarray:
    ccrd, size = np.split(ccrdsizeN, 2, axis=axis)
    size_2 = size / 2
    return np.concatenate([ccrd - size_2, ccrd + size_2], axis=axis)


def ccrdsizeN2cordsizeN(ccrdsizeN: np.ndarray, axis: int = -1) -> np.ndarray:
    ccrd, size = np.split(ccrdsizeN, 2, axis=axis)
    return np.concatenate([ccrd - size / 2, size], axis=axis)


def cordcordN2ccrdsizeN(cordcordN: np.ndarray, axis: int = -1) -> np.ndarray:
    cord1, cord2 = np.split(cordcordN, 2, axis=axis)
    return np.concatenate([(cord1 + cord2) / 2, cord2 - cord1], axis=axis)


def cordcordN2cordsizeN(cordcordN: np.ndarray, axis: int = -1) -> np.ndarray:
    cord1, cord2 = np.split(cordcordN, 2, axis=axis)
    return np.concatenate([cord1, cord2 - cord1], axis=axis)


def cordcordN2cpctN(cordcordN: np.ndarray, axis=-1) -> np.ndarray:
    cord1, cord2 = np.split(cordcordN, 2, axis=axis)
    return np.prod(cord2 - cord1, axis=axis)


def ccrdsizeN2cpctN(ccrdsizeN: np.ndarray, axis=-1) -> np.ndarray:
    _, size = np.split(ccrdsizeN, 2, axis=axis)
    return np.prod(size, axis=axis)


def cordsizeN2cpctN(cordsizeN: np.ndarray, axis=-1) -> np.ndarray:
    _, size = np.split(cordsizeN, 2, axis=axis)
    return np.prod(size, axis=axis)


def cordsN2cordcordN(cordsN: np.ndarray, num_dim: int = 4) -> np.ndarray:
    if cordsN.shape[-2] == 0:
        return np.zeros(shape=list(cordsN.shape[:-2]) + [num_dim])
    else:
        cord1 = np.min(cordsN, axis=-2)
        cord2 = np.max(cordsN, axis=-2)
        return np.concatenate([cord1, cord2], axis=-1)


def cordsN2ccrdsizeN(cordsN: np.ndarray) -> np.ndarray:
    return cordcordN2ccrdsizeN(cordsN2cordcordN(cordsN))


def cordsN2ccrdN(cordsN: np.ndarray) -> np.ndarray:
    if cordsN.shape[-2] == 0:
        return np.zeros(shape=list(cordsN.shape[:-2]) + [0])
    else:
        cord1 = np.min(cordsN, axis=-2)
        cord2 = np.max(cordsN, axis=-2)
        return (cord1 + cord2) / 2


def cordsN2rotN(cordsN: np.ndarray) -> np.ndarray:
    if cordsN.shape[-2] < cordsN.shape[-1]:
        return np.eye(cordsN.shape[-1])
    cordsN_nmd = cordsN - np.mean(cordsN, axis=0)
    s, v, d = np.linalg.svd(cordsN_nmd)
    d = d * np.sign(np.linalg.det(d)[..., None, None])
    return np.swapaxes(d, axis2=-2, axis1=-1)


def cordsN_samp(cordsN: np.ndarray, num_samp: int = 1) -> np.ndarray:
    if num_samp == 1:
        return cordsN
    cordsN_rnd = np.concatenate([cordsN[..., 1:, :], cordsN[..., :1, :]], axis=-2)
    pows = np.linspace(start=0, stop=1, num=num_samp, endpoint=False)[..., None]
    cordsN_mix = cordsN_rnd[..., None, :] * pows + cordsN[..., None, :] * (1 - pows)
    new_shape = list(cordsN.shape[:-2]) + [cordsN.shape[-2] * num_samp] + [cordsN.shape[-1]]
    cordsN_mix = np.reshape(cordsN_mix, newshape=new_shape)
    return cordsN_mix


def cordN_perspective(cordN: np.ndarray, homographyN: np.ndarray) -> np.ndarray:
    cordsN_ext = np.concatenate([cordN, np.ones(shape=list(cordN.shape[:-1]) + [1])], axis=-1)
    cordsN_ext = cordsN_ext @ np.swapaxes(homographyN, axis1=-1, axis2=-2)
    return cordsN_ext[..., :cordN.shape[-1]] / cordsN_ext[..., cordN.shape[-1]:]


def cordN_linear(cordN: np.ndarray, scaleN: np.ndarray, biasN: np.ndarray) -> np.ndarray:
    return cordN * scaleN + biasN


def cordcordN_clip(cordcordN: np.ndarray, cordcordN_rgn: np.ndarray):
    cord_min, cord_max = np.split(cordcordN_rgn, 2, axis=-1)
    cord_min = np.concatenate([cord_min, cord_min], axis=-1)
    cord_max = np.concatenate([cord_max, cord_max], axis=-1)
    return np.minimum(np.maximum(cordcordN, cord_min), cord_max)


def cordN_clip(cordN: np.ndarray, cordcordN_rgn: np.ndarray):
    cord_min, cord_max = np.split(cordcordN_rgn, 2, axis=-1)
    return np.minimum(np.maximum(cordN, cord_min), cord_max)


def cordcordN2linear(cordcordN_src: np.ndarray, cordcordN_dst: np.ndarray) \
        -> (np.ndarray, np.ndarray):
    cord1_src, cord2_src = np.split(cordcordN_src, 2, axis=-1)
    cord1_dst, cord2_dst = np.split(cordcordN_dst, 2, axis=-1)
    scale = (cord2_dst - cord1_dst) / (cord2_src - cord1_src)
    bias = cord1_dst - cord1_src * scale
    return scale, bias


# </editor-fold>

# <editor-fold desc='numpy水平边界'>
##################################################################################
#      0---3----> x
#      |   |
#      1---2
#      |
#      v y
##################################################################################
_SIGN_WH2XYS_VERT_N = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])
_SIGN_WH2XYS_EDGEMID_N = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
_SIGN_WH2XYS_SAMP_N = np.concatenate([_SIGN_WH2XYS_EDGEMID_N, _SIGN_WH2XYS_VERT_N / np.sqrt(2)], axis=0)
NUM_WH2XYS_SAMP = _SIGN_WH2XYS_SAMP_N.shape[0]
_IDX_XYXY2XP_N = np.array((0, 0, 2, 2))
_IDX_XYXY2YP_N = np.array((1, 3, 3, 1))
_IDX_BOX_WDIR = np.array([[0, 3], [1, 2]])
_IDX_BOX_HDIR = np.array([[0, 1], [3, 2]])
_IDX_BOX_WHDIR = np.stack([_IDX_BOX_WDIR, _IDX_BOX_HDIR], axis=0)

_IDX_BOX_EDGE = np.array([[0, 1], [1, 2],
                          [2, 3], [3, 0]])

##################################################################################


xydwhN2xyxyN = cordsizeN2cordcordN
xydwhN2xywhN = cordsizeN2ccrdsizeN
xywhN2xyxyN = ccrdsizeN2cordcordN
xyxyN2xywhN = cordcordN2ccrdsizeN
xyxyN2areaN = cordcordN2cpctN
xywhN2areaN = ccrdsizeN2cpctN
xysN2xyxyN = cordsN2cordcordN
xypN_samp = cordsN_samp
xysN2rot2N = cordsN2rotN

xyxyN2linear = cordcordN2linear
xysN_linear = cordN_linear
xysN_perspective = cordN_perspective


def xysN2xywhN(xysN: np.ndarray) -> np.ndarray:
    return xyxyN2xywhN(xysN2xyxyN(xysN))


def xyxyN2xypN(xyxyN: np.ndarray) -> np.ndarray:
    return np.stack([xyxyN[..., _IDX_XYXY2XP_N], xyxyN[..., _IDX_XYXY2YP_N]], axis=-1)


def xywhN2xypN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[..., None, :2] + _SIGN_WH2XYS_VERT_N * xywhN[..., None, 2:4] / 2


def xywhN2xysN_edgemid(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[..., None, :2] + _SIGN_WH2XYS_EDGEMID_N * xywhN[..., None, 2:4] / 2


def xyxyN2xysN_edgemid(xyxyN: np.ndarray) -> np.ndarray:
    return xywhN2xysN_edgemid(xyxyN2xywhN(xyxyN))


def xywhN2xysN_samp(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[..., None, :2] + _SIGN_WH2XYS_SAMP_N * xywhN[..., None, 2:4] / 2


def xyxyN2xysN_samp(xyxyN: np.ndarray) -> np.ndarray:
    return xywhN2xysN_samp(xyxyN2xywhN(xyxyN))


def xyxyN_perspective2linear(xyxyN: np.ndarray, homographyN: np.ndarray) -> (np.ndarray, np.ndarray):
    xyp = xyxyN2xypN(xyxyN)
    xys_cen = np.mean(xyp[_IDX_BOX_EDGE], axis=1)
    xys_pjd = xysN_perspective(xys_cen, homographyN)
    xyxyN_pjd = xysN2xyxyN(xys_pjd)
    return xyxyN2linear(xyxyN, xyxyN_pjd)


def xysN2perspective(xysN_src: np.ndarray, xysN_dst: np.ndarray) -> np.ndarray:
    assert xysN_src.shape == xysN_dst.shape, 'len err'
    num_dim = xysN_src.shape[-1]
    xxp = xysN_src * xysN_dst[..., 0:1]
    yyp = xysN_src * xysN_dst[..., 1:2]
    Ax = np.concatenate([xysN_src, np.ones(list(xysN_src.shape[:-1]) + [1]),
                         np.zeros(list(xysN_src.shape[:-1]) + [3]), -xxp], axis=-1)
    Ay = np.concatenate([np.zeros(list(xysN_src.shape[:-1]) + [3]), xysN_src,
                         np.ones(list(xysN_src.shape[:-1]) + [1]), -yyp], axis=-1)
    A = np.concatenate([Ax, Ay], axis=-2)
    b = np.concatenate([xysN_dst[..., 0:1], xysN_dst[..., 1:2]], axis=-2)
    A_T = np.swapaxes(A, axis1=-1, axis2=-2)
    h = np.linalg.inv(A_T @ A) @ A_T @ b
    h = np.concatenate([h[..., 0], np.ones(list(h.shape[:-2]) + [1])], axis=-1)
    h = h.reshape(list(xysN_src.shape[:-2]) + [num_dim + 1, num_dim + 1])
    return h


# </editor-fold>

# <editor-fold desc='numpy旋转边界'>


def alphaN2rot2N(alphaN: np.ndarray) -> np.ndarray:
    cos = np.cos(alphaN)
    sin = np.sin(alphaN)
    mat = np.stack([np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)], axis=-2)
    return mat


def unit2N2rot2N(unit2N: np.ndarray) -> np.ndarray:
    mats = np.stack([unit2N, np.stack([-unit2N[..., 1], unit2N[..., 0]], axis=-1)], axis=-2)
    return mats


def unit2N2alphaN(unit2N: np.ndarray) -> np.ndarray:
    return np.arctan2(unit2N[..., 1], unit2N[..., 0])


def alphaN2unit2N(alphaN: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(alphaN), np.sin(alphaN)], axis=-1)


def xywhaN2xypN(xywhaN: np.ndarray) -> np.ndarray:
    rot2 = alphaN2rot2N(xywhaN[..., 4])
    xyp = xywhaN[..., None, :2] + (_SIGN_WH2XYS_VERT_N * xywhaN[..., None, 2:4] / 2) @ rot2
    return xyp


def xywhaN2xysN_edgemid(xywhaN: np.ndarray) -> np.ndarray:
    rot2 = alphaN2rot2N(xywhaN[..., 4])
    xyp = xywhaN[..., None, :2] + (_SIGN_WH2XYS_EDGEMID_N * xywhaN[..., None, 2:4] / 2) @ rot2
    return xyp


def xywhaN2xysN_samp(xywhaN: np.ndarray) -> np.ndarray:
    rot2 = alphaN2rot2N(xywhaN[..., 4])
    xyp = xywhaN[..., None, :2] + (_SIGN_WH2XYS_SAMP_N * xywhaN[..., None, 2:4] / 2) @ rot2
    return xyp


def xywhaN2xyxyN(xywhaN: np.ndarray) -> np.ndarray:
    return xysN2xyxyN(xywhaN2xypN(xywhaN))


def xypN2xywhN(xypN: np.ndarray) -> np.ndarray:
    return xyxyN2xywhN(xysN2xyxyN(xypN))


def xywhN2xywhaN(xywhN: np.ndarray, longer_width: bool = True) -> np.ndarray:
    alphas = np.zeros(shape=xywhN.shape[:-1])
    if longer_width:
        fltr = xywhN[..., 3] > xywhN[..., 2]
        alphas = np.where(fltr, alphas + np.pi / 2, alphas)
        ws = np.where(fltr, xywhN[..., 3], xywhN[..., 2])
        hs = np.where(fltr, xywhN[..., 2], xywhN[..., 3])

        return np.concatenate([xywhN[..., :2], ws[..., None], hs[..., None], alphas[..., None]], axis=-1)
    else:
        return np.concatenate([xywhN, alphas[..., None]], axis=-1)


def xyxyN2xywhaN(xyxyN: np.ndarray, longer_width: bool = True) -> np.ndarray:
    return xywhN2xywhaN(xyxyN2xywhN(xyxyN), longer_width=longer_width)


def xysN2diruN(xysN: np.ndarray) -> np.ndarray:
    return xysN2rot2N(xysN)[0]


def xysN2alphaN(xysN: np.ndarray) -> np.ndarray:
    return unit2N2alphaN(xysN2diruN(xysN))


def xysN_aN2xywhaN(xysN: np.ndarray, aN: np.ndarray) -> np.ndarray:
    rot2 = alphaN2rot2N(aN)
    xys_proj = xysN @ rot2.T
    xywh_proj = xysN2xywhN(xys_proj)
    xy_cen = xywh_proj[:2] @ rot2
    return np.concatenate([xy_cen, xywh_proj[2:4], [aN]], axis=0)


def xysN2xywhaN(xysN: np.ndarray) -> np.ndarray:
    rot2 = xysN2rot2N(xysN)
    xys_proj = xysN @ rot2
    xywh_proj = xysN2xywhN(xys_proj)
    xy_cen = np.sum(rot2 * xywh_proj[..., None, :2], axis=-1)
    aN = unit2N2alphaN(rot2[..., 0])
    return np.concatenate([xy_cen, xywh_proj[..., 2:4], aN[..., None]], axis=-1)


def xywhaN2xywhuvN(xywhaN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhaN[..., :4], alphaN2unit2N(xywhaN[..., 4])], axis=-1)


def xywhuvN2xywhaN(xywhuvN: np.ndarray) -> np.ndarray:
    return np.concatenate([xywhuvN[..., :4], unit2N2alphaN(xywhuvN[..., 4:6])[..., None]], axis=-1)


def xywhaN2xywhN(xywhaN: np.ndarray) -> np.ndarray:
    rot2 = alphaN2rot2N(xywhaN[..., 4])
    wh = np.sum(np.abs(xywhaN[..., 2:4, None] * rot2), axis=-1)
    return np.concatenate([xywhaN[..., :2], wh], axis=-1)

def xypN_xysN_isin(xypN: np.ndarray, xysN: np.ndarray):
    abcl = xypN2abclN(xypN)[..., None, :]
    r = np.sum(xysN * abcl[..., :2], axis=-1) + abcl[..., 2]
    msk = np.all(r >= 0, axis=-2) + np.all(r <= 0, axis=-2)
    return msk
# </editor-fold>

# <editor-fold desc='numpy极坐标变换'>
xypN2xyN = cordsN2ccrdN


def xyN2ialphaN(xyN: np.ndarray, num_div: int = 16) -> np.ndarray:
    alphas = np.arctan2(xyN[..., 1], xyN[..., 0])
    ainds = np.round(alphas / (np.pi * 2) * num_div).astype(np.int32) % num_div
    return ainds


def xypN2xyN_dpN(xypN: np.ndarray, num_div: int = 18) -> (np.ndarray, np.ndarray):
    cen = xypN2xyN(xypN)
    dpN = xyN_xypN2dpN(xyN=cen, xypN=xypN, num_div=num_div)
    return cen, dpN


def xyN_xypN_alphaN2dpN_interp(xyN: np.ndarray, xypN: np.ndarray, asN: np.ndarray) -> np.ndarray:
    xypN_ref = xypN - xyN
    dpN = np.linalg.norm(xypN_ref, axis=1)
    alphas_vert = unit2N2alphaN(xypN_ref / dpN[:, None]) % (np.pi * 2)
    order = np.argsort(alphas_vert)
    alphas_vert, dpN = alphas_vert[order], dpN[order]
    dpN_intp = np.interp(asN, alphas_vert, dpN, period=np.pi * 2)
    return dpN_intp


def xyN_xypN2dpN(xyN: np.ndarray, xypN: np.ndarray, num_div: int = 18) -> np.ndarray:
    abcls = xypN2abclN(xypN)
    thetas = divide_circleN(num_div)
    sins = np.sin(thetas)
    coss = np.cos(thetas)
    norms = -(abcls[..., 0:1] * coss + abcls[..., 1:2] * sins)
    rs = (xyN[..., 0:1] * abcls[..., 0] + xyN[..., 1:2] * abcls[..., 1] + abcls[..., 2])
    dls = rs[..., None] / norms
    detlas = xyN[..., None, :] - xypN
    ks = -(detlas[..., 0:1] * sins - detlas[..., 1:2] * coss) / norms
    fltr = (ks >= -1e-7) * (ks <= 1 + 1e-7) * (dls > 0)
    dls[~fltr] = np.inf
    dls = np.min(dls, axis=-2)
    dls[np.isinf(dls)] = 0
    return dls


def dpN_smth(dpN: np.ndarray, nei_power: float = 0.5, axis: int = -1) -> np.ndarray:
    dls_p = np.roll(dpN, shift=-1, axis=axis)
    dls_n = np.roll(dpN, shift=1, axis=axis)
    dls = (dpN + (dls_p + dls_n) * nei_power) / (1 + 2 * nei_power)
    return dls


def dpN_smth_fft(dpN: np.ndarray, num_spec: int = 5, axis: int = -1) -> np.ndarray:
    pows = np.fft.rfft(dpN, axis=axis)
    pows[..., num_spec:] = 0
    dpN_rec = np.fft.irfft(pows, axis=axis)
    return dpN_rec


def xypN_regularization(xypN: np.ndarray, num_div: int = 18) -> np.ndarray:
    censN, dpN = xypN2xyN_dpN(xypN, num_div=num_div)
    xypN = cenN_dpN2xypN(censN, dpN)
    return xypN


def divide_circleN(num_div: int = 18) -> np.ndarray:
    return np.linspace(start=0, stop=np.pi * 2, num=num_div, endpoint=False)


def create_unit2sN(num_div: int = 18, bias: float = 0) -> np.ndarray:
    alphas = divide_circleN(num_div) + bias
    return np.stack([np.cos(alphas), np.sin(alphas)], axis=1)


def cenN_dpN2xypN(xyN: np.ndarray, dpN: np.ndarray) -> np.ndarray:
    dxys = create_unit2sN(num_div=dpN.shape[-1])
    xypN = xyN[..., None, :] + dpN[..., None] * dxys
    return xypN


# </editor-fold>

# <editor-fold desc='numpy转换mask'>

def maskNb2xysNi(maskNb: np.ndarray) -> np.ndarray:
    iys, ixs = np.nonzero(maskNb)
    if len(iys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    else:
        return np.stack([ixs, iys], axis=1)


def xypN2maskNb(xypN: np.ndarray, size: tuple) -> np.ndarray:
    if len(xypN.shape) > 2:
        return np.stack([xypN2maskNb(xypN_sub, size) for xypN_sub in xypN], axis=0)
    elif len(xypN.shape) == 2:
        maskNb = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
        if np.prod(size) > 0 and xypN.shape[0] >= 3:
            cv2.fillPoly(maskNb, [xypN.astype(np.int32)], color=1.0)
        return maskNb.astype(bool)
    else:
        raise Exception('size err')


def xypNs2maskNb(xypNs: List[np.ndarray], size: tuple) -> np.ndarray:
    maskNb = np.zeros(shape=(size[1], size[0]), dtype=np.float32)
    xypNs = [xypN.astype(np.int32) for xypN in xypNs]
    cv2.fillPoly(maskNb, xypNs, color=1.0)
    return maskNb.astype(bool)


def arange2dN(height: int, width: int) -> (np.ndarray, np.ndarray):
    ys = np.broadcast_to(np.arange(height)[:, None], (height, width))
    xs = np.broadcast_to(np.arange(width)[None, :], (height, width))
    return ys, xs


def xyxyN2xysNi(xyxyN: np.ndarray, size: tuple) -> np.ndarray:
    xyxyN = xyxyN.astype(np.int32)
    xyxyN = np.clip(np.minimum(xyxyN, np.tile(size, 2)), a_min=0, a_max=None)
    iys, ixs = arange2dN(xyxyN[3] - xyxyN[1], xyxyN[2] - xyxyN[0])
    ixys = np.stack([ixs, iys], axis=2).reshape(-1, 2) + xyxyN[:2]
    return ixys


def xywhN2xysNi(xywhN: np.ndarray, size: tuple) -> np.ndarray:
    return xyxyN2xysNi(xywhN2xyxyN(xywhN), size=size)


def xypN2xysNi(xypN: np.ndarray, size: tuple) -> np.ndarray:
    xyxy = xysN2xyxyN(xypN).astype(np.int32)
    xyxy = np.clip(np.minimum(xyxy, np.tile(size, 2)), a_min=0, a_max=None)
    patch_size = xyxy[2:4] - xyxy[:2]
    maskNb = xypN2maskNb(xypN - xyxy[:2], patch_size)
    iys, ixs = np.nonzero(maskNb)
    ixys = np.stack([ixs, iys], axis=1) + xyxy[:2]
    return ixys


def xypN2abclN(xypN: np.ndarray) -> np.ndarray:
    xypN_rnd = np.roll(xypN, shift=-1, axis=-2)
    As = xypN[..., 1] - xypN_rnd[..., 1]
    Bs = xypN_rnd[..., 0] - xypN[..., 0]
    Cs = xypN_rnd[..., 1] * xypN[..., 0] - xypN[..., 1] * xypN_rnd[..., 0]
    return np.stack([As, Bs, Cs], axis=-1)


def xypN_clkwise(xypN: np.ndarray) -> np.ndarray:
    xypN_rnd = np.roll(xypN, shift=-1, axis=-2)
    areas = xypN_rnd[..., 0] * xypN[..., 1] - xypN[..., 0] * xypN_rnd[..., 1]
    fltr = np.sum(areas, axis=-1) < 0
    xypN = np.where(fltr[..., None, None], xypN[..., ::-1, :], xypN)
    return xypN


def xypN2areaN_sign(xypN: np.ndarray) -> np.ndarray:
    xypN_rnd = np.roll(xypN, shift=-1, axis=-2)
    areas = xypN_rnd[..., 0] * xypN[..., 1] - xypN[..., 0] * xypN_rnd[..., 1]
    areas = np.sum(areas, axis=-1) / 2
    return areas


def xypN2areaN(xypN: np.ndarray) -> np.ndarray:
    return np.abs(xypN2areaN_sign(xypN))


def xyxyN2maskNb(xyxyN: np.ndarray, size: tuple) -> np.ndarray:
    ys, xs = arange2dN(size[1], size[0])
    mesh = np.stack([xs, ys], axis=-1)
    xy_min, xy_max = np.split(xyxyN[..., None, None, :], 2, axis=-1)
    maskNb = np.all((mesh < xy_max) * (mesh > xy_min), axis=-1)
    return maskNb


def xywhN2maskNb(xywhN: np.ndarray, size: tuple) -> np.ndarray:
    xyxyN = xywhN2xyxyN(xywhN)
    return xyxyN2maskNb(xyxyN, size)


def xywhaN2maskNb(xywhaN: np.ndarray, size: tuple) -> np.ndarray:
    xypN = xywhaN2xypN(xywhaN)
    return xypN2maskNb(xypN, size)


# </editor-fold>

# <editor-fold desc='numpy多边形'>
def xyxyN_perspective(xyxyN: np.ndarray, homographyN: np.ndarray) -> np.ndarray:
    xypN = xyxyN2xysN_edgemid(xyxyN)
    xypN = xysN_perspective(xypN, homographyN=homographyN)
    return xysN2xyxyN(xypN)


def xywhN_perspective(xywhN: np.ndarray, homographyN: np.ndarray) -> np.ndarray:
    xypN = xywhN2xysN_edgemid(xywhN)
    xypN = xysN_perspective(xypN, homographyN=homographyN)
    return xysN2xywhN(xypN)


def xywhaN_perspective(xywhaN: np.ndarray, homographyN: np.ndarray) -> np.ndarray:
    xypN = xywhaN2xysN_edgemid(xywhaN)
    xypN = xysN_perspective(xypN, homographyN=homographyN)
    return xysN2xywhaN(xypN)


def xypN_intersect(xyp1N: np.ndarray, xyp2N: np.ndarray) -> Union[np.ndarray, List]:
    if len(xyp1N.shape) > len(xyp2N.shape):
        xyp2N = np.broadcast_to(xyp2N, list(xyp1N.shape[:-2]) + list(xyp2N.shape[-2:]))
    elif len(xyp1N.shape) < len(xyp2N.shape):
        xyp1N = np.broadcast_to(xyp1N, list(xyp2N.shape[:-2]) + list(xyp1N.shape[-2:]))
    if len(xyp1N.shape) > 2:
        return [xypN_intersect(xyp1N_sub, xyp2N_sub) for xyp1N_sub, xyp2N_sub in zip(xyp1N, xyp2N)]
    elif len(xyp1N.shape) == 2:
        if xyp1N.shape[0] < 3 or xyp2N.shape[0] < 3:
            return xyp1N
        p1 = Polygon(xyp1N)
        p2 = Polygon(xyp2N)
        pi = p1.intersection(p2)
        if isinstance(pi, GeometryCollection):
            pi = pi.envelope
        xyNi = np.stack(pi.exterior.coords.xy, axis=-1)
        return xyNi
    else:
        raise Exception('size err')


def xypN_xyxyN_intersect(xypN: np.ndarray, xyxyN: np.ndarray) -> Union[np.ndarray, List]:
    return xypN_intersect(xypN, xyxyN2xypN(xyxyN))


# </editor-fold>


# <editor-fold desc='torch通用转换'>
def cordsizeT2cordcordT(cordsizeT: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cord, size = torch.chunk(cordsizeT, 2, dim=dim)
    return torch.cat([cord, cord + size], dim=dim)


def cordsizeT2ccrdsizeT(cordsizeT: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cord, size = torch.chunk(cordsizeT, 2, dim=dim)
    return torch.cat([cord + size / 2, size], dim=dim)


def ccrdsizeT2cordcordT(ccrdsizeT: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ccrd, size = torch.chunk(ccrdsizeT, 2, dim=dim)
    size_2 = size / 2
    return torch.cat([ccrd - size_2, ccrd + size_2], dim=dim)


def ccrdsizeT2cordsizeT(ccrdsizeN: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ccrd, size = torch.chunk(ccrdsizeN, 2, dim=dim)
    return torch.cat([ccrd - size / 2, size], dim=dim)


def cordcordT2ccrdsizeT(cordcordN: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cord1, cord2 = torch.chunk(cordcordN, 2, dim=dim)
    return torch.cat([(cord1 + cord2) / 2, cord2 - cord1], dim=dim)


def cordcordT2cordsizeT(cordcordN: torch.Tensor, dim: int = -1) -> torch.Tensor:
    cord1, cord2 = torch.chunk(cordcordN, 2, dim=dim)
    return torch.cat([cord1, cord2 - cord1], dim=dim)


def cordcordT2cpctT(cordcordN: torch.Tensor, dim=-1) -> torch.Tensor:
    cord1, cord2 = torch.chunk(cordcordN, 2, dim=dim)
    return torch.prod(cord2 - cord1, dim=dim)


def ccrdsizeT2cpctT(ccrdsizeN: torch.Tensor, dim=-1) -> torch.Tensor:
    _, size = torch.chunk(ccrdsizeN, 2, dim=dim)
    return torch.prod(size, dim=dim)


def cordsizeT2cpctT(cordsizeT: torch.Tensor, dim=-1) -> torch.Tensor:
    _, size = torch.chunk(cordsizeT, 2, dim=dim)
    return torch.prod(size, dim=dim)


def cordsT2cordcordT(cordsT: torch.Tensor) -> torch.Tensor:
    if cordsT.shape[-2] == 0:
        return torch.zeros(size=list(cordsT.size())[:-2] + [0], device=cordsT.device)
    else:
        cord1 = torch.amin(cordsT, dim=-2)
        cord2 = torch.amax(cordsT, dim=-2)
        return torch.cat([cord1, cord2], dim=-1)


def cordsT2ccrdT(cordsT: torch.Tensor) -> torch.Tensor:
    if cordsT.shape[-2] == 0:
        return torch.zeros(size=list(cordsT.size())[:-2] + [0], device=cordsT.device)
    else:
        cord1 = torch.amin(cordsT, dim=-2)
        cord2 = torch.amax(cordsT, dim=-2)
        return (cord1 + cord2) / 2


def cordsT2rotT(cordsT: torch.Tensor) -> torch.Tensor:
    if cordsT.shape[-2] < cordsT.shape[-1]:
        return torch.eye(cordsT.shape[-1])
    cordsT_nmd = cordsT - torch.mean(cordsT, dim=0)
    s, v, d = torch.linalg.svd(cordsT_nmd)
    if torch.linalg.det(d) < 0:
        d = -d
    return torch.transpose(d, dim0=-2, dim1=-1)


def cordsT_samp(cordsT: torch.Tensor, num_samp: int = 1) -> torch.Tensor:
    if num_samp == 1:
        return cordsT
    cordsT_rnd = torch.cat([cordsT[..., 1:, :], cordsT[..., :1, :]], dim=-2)
    pows = torch.linspace(start=0, end=1, steps=num_samp, device=cordsT.device)[..., None]
    cordsT_mix = cordsT_rnd[..., None, :] * pows + cordsT[..., None, :] * (1 - pows)
    new_shape = list(cordsT.shape[:-2]) + [cordsT.shape[-2] * num_samp] + [cordsT.shape[-1]]
    cordsT_mix = cordsT_mix.reshape(new_shape)
    return cordsT_mix


def cordsT_perspective(cordsT: torch.Tensor, homographyT: torch.Tensor) -> torch.Tensor:
    cordsT_ext = torch.cat([cordsT, torch.ones(size=list(cordsT.size())[:-1] + [1], device=cordsT.device)], dim=-1)
    cordsT_ext = cordsT_ext @ homographyT.transpose(-1, -2)
    return cordsT_ext[..., :cordsT.shape[-1]] / cordsT_ext[..., cordsT.shape[-1]:]


def cordsT_linear(cordsT: torch.Tensor, scaleT: torch.Tensor, biasT: torch.Tensor) -> torch.Tensor:
    return cordsT * scaleT + biasT


# </editor-fold>

# <editor-fold desc='torch水平边界'>

_SIGN_WH2DXY_T = torch.Tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]])
_IDX_XYXY2XL_T = torch.Tensor((0, 0, 2, 2))
_IDX_XYXY2YL_T = torch.Tensor((1, 3, 3, 1))

xydwhT2xyxyT = cordsizeT2cordcordT
xydwhT2xywhT = cordsizeT2ccrdsizeT
xywhT2xyxyT = ccrdsizeT2cordcordT
xyxyT2xywhT = cordcordT2ccrdsizeT
xyxyT2areaT = cordcordT2cpctT
xywhT2areaT = ccrdsizeT2cpctT
xysT2xyxyT = cordsT2cordcordT
xypT_samp = cordsT_samp
xysT2rot2T = cordsT2rotT

xysT_linear = cordsT_linear
xysT_perspective = cordsT_perspective


def xysT2xywhT(xysT: torch.Tensor) -> torch.Tensor:
    return xyxyT2xywhT(xysT2xyxyT(xysT))


def xyxyT2xywhaT(xyxyT: torch.Tensor, longer_width: bool = True) -> torch.Tensor:
    return xywhT2xywhaT(xyxyT2xywhT(xyxyT), longer_width=longer_width)


def xyxyT2xypT(xyxyT: torch.Tensor) -> torch.Tensor:
    xypT = torch.stack([xyxyT[..., _IDX_XYXY2XL_T], xyxyT[..., _IDX_XYXY2YL_T]], dim=-1)
    return xypT


def xypT2xywhT(xypT: torch.Tensor) -> torch.Tensor:
    return xyxyT2xywhT(xysT2xyxyT(xypT))


def xywhaT2xyxyT(xywhaT: torch.Tensor) -> torch.Tensor:
    return xysT2xyxyT(xywhaT2xypT(xywhaT))


# </editor-fold>

# <editor-fold desc='torch旋转边界'>


def unit2T2rot2T(unit2T: torch.Tensor) -> torch.Tensor:
    rot2 = torch.stack([unit2T, torch.stack([-unit2T[..., 1], unit2T[..., 0]], dim=-1)], dim=-2)
    return rot2


def alphaT2rot2T(alphaT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(alphaT), torch.sin(alphaT)
    mat = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=-2)
    return mat


def unit2T2alphaT(unit2T: torch.Tensor) -> torch.Tensor:
    return torch.atan2(unit2T[..., 1], unit2T[..., 0])


def alphaT2unit2T(alphaT: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.cos(alphaT), torch.sin(alphaT)], dim=-1)


def xywhT2xywhaT(xywhT: torch.Tensor, longer_width: bool = True) -> torch.Tensor:
    alphas = torch.zeros(size=xywhT.size()[:-1])
    if longer_width:
        fltr = xywhT[..., 3] > xywhT[..., 2]
        alphas = torch.where(fltr, alphas + np.pi / 2, alphas)
        ws = torch.where(fltr, xywhT[..., 3], xywhT[..., 2])
        hs = torch.where(fltr, xywhT[..., 2], xywhT[..., 3])
        return torch.cat([xywhT[..., :2], ws[..., None], hs[..., None], alphas[..., None]], dim=-1)
    else:
        return torch.cat([xywhT, alphas[..., None]], dim=-1)


def xywhaT2xypT(xywhaT: torch.Tensor) -> torch.Tensor:
    mat = alphaT2rot2T(xywhaT[..., 4])
    xyp = xywhaT[..., None, :2] + (_SIGN_WH2DXY_T.to(xywhaT.device) * xywhaT[..., None, 2:4] / 2) @ mat
    return xyp


def xypT2areaT_sign(xypT: torch.Tensor) -> torch.Tensor:
    xypT_rnd = torch.roll(xypT, shifts=-1, dims=-2)
    areas = xypT_rnd[..., 0] * xypT[..., 1] - xypT[..., 0] * xypT_rnd[..., 1]
    areas = torch.abs(torch.sum(areas, dim=-1)) / 2
    return areas


def xypT2areaT(xypT: torch.Tensor) -> torch.Tensor:
    return torch.abs(xypT2areaT_sign(xypT))


def xypT2abclT(xypT: torch.Tensor) -> torch.Tensor:
    xypT_rnd = torch.roll(xypT, shifts=-1, dims=-2)
    As = xypT[..., 1] - xypT_rnd[..., 1]
    Bs = xypT_rnd[..., 0] - xypT[..., 0]
    Cs = xypT_rnd[..., 1] * xypT[..., 0] - xypT[..., 1] * xypT_rnd[..., 0]
    return torch.stack([As, Bs, Cs], dim=-1)


def xysT2unit2T(xysT: torch.Tensor) -> torch.Tensor:
    return xysT2rot2T(xysT)[..., 0, :]


def xysT_aT2xywhaT(xysT: torch.Tensor, alphaT: torch.Tensor) -> torch.Tensor:
    rot2 = alphaT2rot2T(alphaT)
    xys_proj = xysT @ rot2.T
    xywh_proj = xysT2xywhT(xys_proj)
    xy_c = xywh_proj[:2] @ rot2
    return torch.cat([xy_c, xywh_proj[2:4], [alphaT]], dim=0)


def xysT2xywhaT(xysT: torch.Tensor) -> torch.Tensor:
    rot2 = xysT2rot2T(xysT)
    xys_proj = xysT @ rot2
    xywh_proj = xysT2xywhT(xys_proj)
    xy_c = torch.sum(rot2 * xywh_proj[..., None, :2], dim=-1)
    aT = unit2T2alphaT(rot2[..., 0])
    return torch.cat([xy_c, xywh_proj[..., 2:4], aT[..., None]], dim=-1)


# </editor-fold>

# <editor-fold desc='torch极坐标变换'>
xypT2xyT = cordsT2ccrdT


def dpT_samp(dpT: torch.Tensor, num_samp: int = 1) -> torch.Tensor:
    if num_samp <= 1:
        return dpT
    num_div = dpT.size(-1)
    dpT_ext = dpT[..., None].expand(list(dpT.size()) + [num_samp])
    dpT_rnd = torch.roll(dpT_ext, shifts=-1, dims=-2)
    pows = torch.arange(num_samp, device=dpT.device) / num_samp
    dpT_smp = dpT_ext * (1 - pows) + dpT_rnd * pows
    dpT_smp = dpT_smp.reshape(list(dpT.size())[:-1] + [num_samp * num_div])
    return dpT_smp


def dpT_smth(dpT: torch.Tensor, nei_power: float = 0.5, dim: int = -1) -> torch.Tensor:
    dls_p = torch.roll(dpT, shifts=-1, dims=dim)
    dls_n = torch.roll(dpT, shifts=1, dims=dim)
    dls = (dpT + (dls_p + dls_n) * nei_power) / (1 + 2 * nei_power)
    return dls


def dpT_smth_fft(dpT: torch.Tensor, num_spec: int = 5, dim: int = -1) -> torch.Tensor:
    pows = torch.fft.fft(dpT, dim=dim)
    if not (dim + 1) % len(dpT.size()) == 0:
        pows = pows.transpose(dim, -1)
    pows[..., num_spec:] = 0
    if not (dim + 1) % len(dpT.size()) == 0:
        pows = pows.transpose(dim, -1)
    dpT_rec = torch.fft.ifft(pows, dim=dim).float()
    return dpT_rec


def dpT_dstr2dpT(dpT_dstr: torch.Tensor) -> torch.Tensor:
    num_dstr = dpT_dstr.size(-1)
    dpT_dstr_sft = torch.softmax(dpT_dstr, dim=-1) * torch.arange(num_dstr, device=dpT_dstr.device)
    dpT = torch.sum(dpT_dstr_sft, dim=-1)
    return dpT


def xypT_regularization(xypT: torch.Tensor, num_div: int = 18) -> torch.Tensor:
    censT, dpT = xypT2xyT_dpT(xypT, num_div=num_div)
    xypT = xyT_dpT2xypT(censT, dpT)
    return xypT


def dpT2areaT(dpT: torch.Tensor) -> torch.Tensor:
    num_div = dpT.size(-1)
    theta = np.pi * 2 / num_div
    areas = torch.sum(dpT ** 2, dim=-1) * theta / 2
    return areas


def xysT2iasT(xysT: torch.Tensor, num_div: int = 16) -> torch.Tensor:
    alphas = torch.atan2(xysT[..., 1], xysT[..., 0])
    ainds = torch.round(alphas / (np.pi * 2) * num_div).long() % num_div
    return ainds


def divide_circleT(num_div: int = 18, device: torch.device = DEVICE) -> torch.Tensor:
    return torch.arange(0, 2 * np.pi, step=2 * np.pi / num_div, device=device)


def create_unit2sT(num_div: int = 18, bias: float = 0, device: torch.device = DEVICE) -> torch.Tensor:
    alphas = divide_circleT(num_div, device) + bias
    return torch.stack([torch.cos(alphas), torch.sin(alphas)], dim=1)


def xyT_dpT2xypT(xyT: torch.Tensor, dpT: torch.Tensor) -> torch.Tensor:
    dxys = create_unit2sT(num_div=dpT.size(-1), device=xyT.device)
    xyp = xyT[..., None, :] + dpT[..., None] * dxys
    return xyp


def xyT_xypT2dpT(xyT: torch.Tensor, xypT: torch.Tensor, num_div: int = 18) -> torch.Tensor:
    abcls = xypT2abclT(xypT)
    thetas = divide_circleT(num_div).to(xyT.device)
    sins = torch.sin(thetas)
    coss = torch.cos(thetas)
    norms = -(abcls[..., 0:1] * coss + abcls[..., 1:2] * sins)
    rs = (xyT[..., 0:1] * abcls[..., 0] + xyT[..., 1:2] * abcls[..., 1] + abcls[..., 2])
    dls = rs[..., None] / norms
    detlas = xyT[..., None, :] - xypT
    ks = -(detlas[..., 0:1] * sins - detlas[..., 1:2] * coss) / norms
    fltr = (ks >= -1e-7) * (ks <= 1 + 1e-7) * (dls > 0)

    dls[~fltr] = np.inf
    dls = torch.min(dls, dim=-2)[0]
    dls[torch.isinf(dls)] = 0
    return dls


def xypT2xyT_dpT(xypT: torch.Tensor, num_div: int = 18) -> (torch.Tensor, torch.Tensor):
    cens = xypT2xyT(xypT)
    dpT = xyT_xypT2dpT(xyT=cens, xypT=xypT, num_div=num_div)
    return cens, dpT


# </editor-fold>

# <editor-fold desc='torch转换mask'>

def arange2dT(height: int, width: int, device: torch.device = DEVICE) -> (torch.Tensor, torch.Tensor):
    ys = torch.arange(height, device=device)[:, None].expand(height, width)
    xs = torch.arange(width, device=device)[None, :].expand(height, width)
    return ys, xs
# </editor-fold>
