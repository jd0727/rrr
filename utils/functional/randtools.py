from typing import Optional

import cv2
from .borders import *
from ..typings import ps_flt2_repeat, TV_Flt2, TV_Int2


def xyxyN_samp_by_area(xyxyN: np.ndarray, aspect: Optional[float] = None, area_ratio: TV_Flt2 = (0.5, 1)) -> np.ndarray:
    xyxyN = np.array(xyxyN)
    wh = xyxyN[2:4] - xyxyN[0:2]
    area_ratio = ps_flt2_repeat(area_ratio)
    if aspect is not None:
        area_samp = np.random.uniform(low=area_ratio[0], high=area_ratio[1]) * np.prod(wh)
        wh_patch = np.sqrt([area_samp * aspect, area_samp / aspect])
    else:
        len_range = np.sqrt(area_ratio)
        wh_patch = np.random.uniform(low=len_range[0], high=len_range[1], size=2) * wh

    wh_patch = np.minimum(wh_patch, wh)
    x1 = np.random.uniform(low=xyxyN[0], high=xyxyN[2] - wh_patch[0] + 1)
    y1 = np.random.uniform(low=xyxyN[1], high=xyxyN[3] - wh_patch[1] + 1)
    xyxy_patch = np.array([x1, y1, x1 + wh_patch[0], y1 + wh_patch[1]])
    return xyxy_patch


def xyxyN_samp_by_size(xyxyN: np.ndarray, size: TV_Flt2 = (10.0, 5.0)) -> np.ndarray:
    xyxyN = np.array(xyxyN)
    wh = xyxyN[2:4] - xyxyN[0:2]
    size = np.array(ps_flt2_repeat(size))
    size = np.minimum(size, wh)
    x1 = np.random.uniform(low=xyxyN[0], high=xyxyN[2] - size[0] + 1)
    y1 = np.random.uniform(low=xyxyN[1], high=xyxyN[3] - size[1] + 1)
    xyxy_patch = np.array([x1, y1, x1 + size[0], y1 + size[1]])
    return xyxy_patch


def maskNb_samp_ptchNb(maskNb: np.ndarray, ptchNb: np.ndarray) -> np.ndarray:
    h, w = maskNb.shape
    ph, pw = ptchNb.shape
    pw = min(pw, w)
    ph = min(ph, h)
    maskNb_part = maskNb[ph // 2:h - ph // 2 - ph % 2, pw // 2:w - pw // 2 - pw % 2]
    if not np.any(maskNb_part):
        return np.zeros(shape=4)
    maskNb_valid = cv2.erode(maskNb_part.astype(np.uint8), kernel=ptchNb.astype(np.uint8))
    ys, xs = np.nonzero(maskNb_valid)
    if len(ys) == 0:
        return np.zeros(shape=4)
    index = np.random.choice(a=len(ys))
    xc, yc = xs[index], ys[index]
    return np.array([xc, yc])


def maskNb_samp_size(maskNb: np.ndarray, size: TV_Int2 = (20, 20)) -> np.ndarray:
    pw, ph = ps_flt2_repeat(size)
    xc, yc = maskNb_samp_ptchNb(maskNb, ptchNb=np.ones((ph, pw)))
    return np.array([xc, yc, xc + pw, yc + ph])
