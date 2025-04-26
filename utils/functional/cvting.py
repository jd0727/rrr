import warnings
from typing import Iterable, Union, Optional, List, Sequence, Tuple

import PIL
import cv2
import numpy as np
import torch
from PIL import Image

from ..iotools import DEVICE
from ..typings import TV_Int3, ps_int3_repeat, TV_Int2, ps_int2_repeat

warnings.filterwarnings("ignore")
import torch.nn.functional as F
import copy


# <editor-fold desc='分类格式转换'>
def cindT2chotT(cindT: torch.Tensor, num_cls: int) -> torch.Tensor:
    return F.one_hot(cindT, num_classes=num_cls).float()


def cindN2chotN(cindN: np.ndarray, num_cls: int) -> np.ndarray:
    one_hot = np.identity(num_cls)[cindN]
    return one_hot


# </editor-fold>

# <editor-fold desc='numpy和torch转化'>
def arrsN2arrsT(arrsN: Union[np.ndarray, torch.Tensor, Iterable], device: torch.device = DEVICE) \
        -> Union[torch.Tensor, Iterable]:
    if isinstance(arrsN, torch.Tensor):
        return arrsN.to(device)
    elif isinstance(arrsN, np.ndarray):
        arrsN = torch.as_tensor(arrsN).to(device)
        if arrsN.dtype == torch.float64:
            arrsN = arrsN.float()
        if arrsN.dtype == torch.int32:
            arrsN = arrsN.long()
        return arrsN
    elif isinstance(arrsN, dict):
        for key in arrsN.keys():
            arrsN[key] = arrsN2arrsT(arrsN[key], device=device)
        return arrsN
    elif isinstance(arrsN, list) or isinstance(arrsN, tuple):
        arrsN = list(arrsN)
        for i in range(len(arrsN)):
            arrsN[i] = arrsN2arrsT(arrsN[i], device=device)
        return arrsN
    else:
        raise Exception('err')


def arrsT2arrsN(arrsT: Union[np.ndarray, torch.Tensor, Iterable]) -> Union[np.ndarray, Iterable]:
    if isinstance(arrsT, np.ndarray):
        return arrsT
    elif isinstance(arrsT, torch.Tensor):
        return arrsT.detach().cpu().numpy()
    elif isinstance(arrsT, dict):
        for key in arrsT.keys():
            arrsT[key] = arrsT2arrsN(arrsT[key])
        return arrsT
    elif isinstance(arrsT, Iterable):
        arrsT = list(arrsT)
        for i in range(len(arrsT)):
            arrsT[i] = arrsT2arrsN(arrsT[i])
        return arrsT
    else:
        raise Exception('err')


def is_arrsN(arrsN: Union[np.ndarray, torch.Tensor, Iterable]) -> bool:
    if isinstance(arrsN, np.ndarray):
        return True
    elif arrsN.__class__.__name__ in ['dict']:
        return all([is_arrsN(value) for value in arrsN.values()])
    elif arrsN.__class__.__name__ in ['list', 'tuple']:
        return all([is_arrsN(value) for value in arrsN])
    else:
        return False


def arrNs_align(arrNs: Sequence[np.ndarray], pad_val=0, axis: int = 0,
                align_len: Optional[int] = None) -> List[np.ndarray]:
    if align_len is None:
        align_len = max([arr.shape[axis] for arr in arrNs])
    arrs_align = []
    for arr in arrNs:
        shape = list(arr.shape)
        shape[axis] = align_len - shape[axis]
        padding = np.full(shape=shape, fill_value=pad_val, dtype=arr.dtype)
        arr = np.concatenate([arr, padding], axis=axis)
        arrs_align.append(arr)
    return arrs_align


def arrTs_align(arrTs: Sequence[torch.Tensor], pad_val=0, dim: int = 0,
                align_len: Optional[int] = None) -> List[torch.Tensor]:
    if align_len is None:
        align_len = max([arr.shape[dim] for arr in arrTs])
    arrs_align = []
    for arr in arrTs:
        shape = list(arr.shape)
        shape[dim] = align_len - shape[dim]
        padding = torch.full(size=shape, fill_value=pad_val, dtype=arr.dtype, device=arr.device)
        arr = torch.cat([arr, padding], dim=dim)
        arrs_align.append(arr)
    return arrs_align


# </editor-fold>

# <editor-fold desc='list边界格式转换'>

def xyxyS2xywhS(xyxyS: Sequence) -> Tuple:
    x1, y1, x2, y2 = xyxyS
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def xywhS2xyxyS(xywhS: Sequence) -> Tuple:
    xc, yc, w, h = xywhS
    w_2 = w / 2
    h_2 = h / 2
    return (xc - w_2, yc - h_2, xc + w_2, yc + h_2)


# </editor-fold>

# <editor-fold desc='图像格式转换'>
_IMG_GEN = Union[PIL.Image.Image, np.ndarray, torch.Tensor]
_IMGS_GEN = Union[np.ndarray, torch.Tensor, Sequence[_IMG_GEN]]


def imgP2imgN(imgP: PIL.Image.Image) -> np.ndarray:
    if imgP.size[0] == 0 or imgP.size[1] == 0:
        if imgP.mode == 'L':
            return np.zeros(shape=(imgP.size[1], imgP.size[0]))
        elif imgP.mode == 'RGB':
            return np.zeros(shape=(imgP.size[1], imgP.size[0], 3))
        elif imgP.mode == 'RGBA':
            return np.zeros(shape=(imgP.size[1], imgP.size[0], 4))
        else:
            raise Exception('err num ' + str(imgP.mode))
    imgN = np.array(imgP)
    return imgN


def imgN2imgP(imgN: np.ndarray) -> PIL.Image.Image:
    if len(imgN.shape) == 2:
        imgP_tp = 'L'
    elif len(imgN.shape) == 3 and imgN.shape[2] == 1:
        imgP_tp = 'L'
        imgN = imgN.squeeze(axis=2)
    elif imgN.shape[2] == 3:
        imgP_tp = 'RGB'
    elif imgN.shape[2] == 4:
        imgP_tp = 'RGBA'
    else:
        raise Exception('err num ' + str(imgN.shape))
    imgN = Image.fromarray(imgN.astype(np.uint8), mode=imgP_tp)
    return imgN


def imgN2imgT(imgN: np.ndarray, device=DEVICE) -> torch.Tensor:
    if len(imgN.shape) == 2:
        imgN = imgN[..., None]
    imgT = torch.from_numpy(np.ascontiguousarray(imgN)).float()
    imgT = imgT.permute((2, 0, 1)) / 255
    imgT = imgT[None, :]
    return imgT.to(device)


def imgT2imgN(imgT: torch.Tensor) -> np.ndarray:
    imgT = imgT * 255
    imgN = imgT.detach().cpu().numpy()
    if len(imgN.shape) == 4 and imgN.shape[0] == 1:
        imgN = imgN.squeeze(axis=0)
    imgN = np.transpose(imgN, (1, 2, 0))  # CHW转为HWC
    return imgN


def imgP2imgT(imgP: PIL.Image.Image, device=DEVICE) -> torch.Tensor:
    imgT = torch.from_numpy(np.array(imgP)).float()
    imgT = imgT.permute((2, 0, 1)) / 255
    imgT = imgT[None, :]
    return imgT.to(device)


def imgT2imgP(imgT: torch.Tensor) -> PIL.Image.Image:
    imgN = imgT2imgN(imgT)
    imgP = imgN2imgP(imgN)
    return imgP


def imgNs2imgPs(imgs: Sequence[np.ndarray]) -> List[PIL.Image.Image]:
    return [imgN2imgP(imgs[i]) for i in range(len(imgs))]


def imgPs2imgNs(imgs: Sequence[PIL.Image.Image]) -> List[np.ndarray]:
    return [imgP2imgN(imgs[i]) for i in range(len(imgs))]


def imgs2imgNs(imgs: Sequence[_IMG_GEN]) -> List[np.ndarray]:
    return [img2imgN(imgs[i]) for i in range(len(imgs))]


def imgs2imgPs(imgs: Sequence[_IMG_GEN]) -> List[PIL.Image.Image]:
    return [img2imgP(imgs[i]) for i in range(len(imgs))]


def img2imgT(img: _IMG_GEN, device: torch.device = DEVICE) -> torch.Tensor:
    if isinstance(img, np.ndarray):
        return imgN2imgT(img, device)
    elif isinstance(img, PIL.Image.Image):
        return imgP2imgT(img, device)
    elif isinstance(img, torch.Tensor):
        if len(img.size()) == 3:
            return img[None].to(device)
        elif len(img.size()) == 4:
            return img.to(device)
        else:
            raise Exception('err size ' + str(img.size()))
    else:
        raise Exception('err type ' + img.__class__.__name__)


def img2imgP(img: _IMG_GEN) -> PIL.Image.Image:
    if isinstance(img, np.ndarray):
        return imgN2imgP(img)
    elif isinstance(img, PIL.Image.Image):
        return img
    elif isinstance(img, torch.Tensor):
        return imgT2imgP(img)
    else:
        raise Exception('err type ' + img.__class__.__name__)


def img2imgN(img: _IMG_GEN) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, PIL.Image.Image):
        return imgP2imgN(img)
    elif isinstance(img, torch.Tensor):
        return imgT2imgN(img)
    else:
        raise Exception('err type ' + img.__class__.__name__)


def imgsT_normalize(imgsT: torch.Tensor, mean: tuple = (0.485, 0.456, 0.406),
                    std: tuple = (0.229, 0.224, 0.225)) -> torch.Tensor:
    mean = torch.as_tensor(mean).to(imgsT.device)
    std = torch.as_tensor(std).to(imgsT.device)
    imgsT = (imgsT - mean[..., None, None]) / std[..., None, None]
    return imgsT


# </editor-fold>


# <editor-fold desc='图像批次处理'>
def _size_limt_scale(size: tuple, max_size: tuple, only_smaller: bool = False, only_larger: bool = False) \
        -> (np.ndarray, Tuple):
    scale = min(np.array(max_size) / np.array(size))
    if (scale > 1.0 and only_smaller) or (scale < 1.0 and only_larger):
        return np.ones(shape=2), size
    else:
        final_size = tuple((np.array(size) * scale).astype(np.int32))
        return np.array([scale, scale]), final_size


def imgP_lmtsize(imgP: PIL.Image.Image, max_size: tuple, resample=Image.BILINEAR,
                 only_smaller: bool = False, only_larger: bool = False) -> (PIL.Image.Image, np.ndarray):
    scale, final_size = _size_limt_scale(imgP.size, max_size, only_smaller=only_smaller, only_larger=only_larger)
    if np.all(scale == 1.0):
        return imgP, scale
    else:
        imgP = imgP.resize(size=final_size, resample=resample)
        return imgP, scale


def imgN_lmtsize(imgN: np.ndarray, max_size: tuple, resample=cv2.INTER_CUBIC,
                 only_smaller: bool = False, only_larger: bool = False) -> (np.ndarray, np.ndarray):
    size = (imgN.shape[1], imgN.shape[0])
    scale, final_size = _size_limt_scale(size, max_size, only_smaller=only_smaller, only_larger=only_larger)
    if np.all(scale == 1.0):
        return imgN, scale
    else:
        imgN = cv2.resize(imgN, final_size, interpolation=resample)
        return imgN, scale


def imgN_pad_aspect(imgN: np.ndarray, aspect: float, pad_val: int = 127) -> (np.ndarray, np.ndarray):
    h, w, _ = imgN.shape
    if w / h > aspect:
        h_dt = int(round(w / aspect) - h)
        pad_width = np.array(((h_dt // 2, h_dt - h_dt // 2), (0, 0), (0, 0)))
        imgN = np.pad(imgN, pad_width=pad_width, constant_values=pad_val)
        bias = np.array([0, h_dt // 2])
    elif w / h < aspect:
        w_dt = int(round(h * aspect) - w)
        pad_width = np.array(((0, 0), (w_dt // 2, w_dt - w_dt // 2), (0, 0)))
        imgN = np.pad(imgN, pad_width=pad_width, constant_values=pad_val)
        bias = np.array([w_dt // 2, 0])
    else:
        bias = np.zeros(shape=2)
    return imgN, bias


def imgP_pad_aspect(imgP: PIL.Image.Image, aspect: float, pad_val: TV_Int3 = (127, 127, 127)) \
        -> (PIL.Image.Image, np.ndarray):
    w, h = imgP.size
    pad_val = ps_int3_repeat(pad_val)
    if w / h > aspect:
        h_dt = int(round(w / aspect) - h)
        imgP_new = Image.new(mode=imgP.mode, size=(w, round(w / aspect)), color=pad_val)
        imgP_new.paste(imgP, (0, h_dt // 2))
        bias = np.array([0, h_dt // 2])
    elif w / h < aspect:
        w_dt = int(round(h * aspect) - w)
        imgP_new = Image.new(mode=imgP.mode, size=(round(h * aspect), h), color=pad_val)
        imgP_new.paste(imgP, (w_dt // 2, 0))
        bias = np.array([w_dt // 2, 0])
    else:
        imgP_new = imgP
        bias = np.zeros(shape=2)
    return imgP_new, bias


def imgP_pad_lmtsize(imgP: PIL.Image.Image, max_size: tuple, pad_val: int = 127, resample=Image.BILINEAR) \
        -> (PIL.Image.Image, np.ndarray, np.ndarray):
    if imgP.size == max_size:
        return imgP, np.ones(shape=2), np.zeros(shape=2)
    aspect = max_size[0] / max_size[1]
    imgP, bias = imgP_pad_aspect(imgP, aspect=aspect, pad_val=pad_val)
    scale = min(max_size[0] / imgP.size[0], max_size[1] / imgP.size[1])
    scale = np.array([scale, scale])
    imgP = imgP.resize(max_size, resample=resample)
    return imgP, scale, bias * scale


def imgN_pad_lmtsize(imgN: np.ndarray, max_size: tuple, pad_val: int = 127, resample=cv2.INTER_CUBIC) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    if img2size(imgN) == max_size:
        return imgN, np.ones(shape=2), np.zeros(shape=2)
    aspect = max_size[0] / max_size[1]
    imgN, bias = imgN_pad_aspect(imgN, aspect=aspect, pad_val=pad_val)
    scale = min(max_size[0] / imgN.shape[1], max_size[1] / imgN.shape[0])
    scale = np.array([scale, scale])
    imgN = cv2.resize(imgN, max_size, interpolation=resample)
    return imgN, scale, bias * scale


def imgT_pad(imgT: torch.Tensor, target_size: TV_Int2, pad_val: float = 0.5) \
        -> (torch.Tensor, np.ndarray, np.ndarray):
    cur_size = np.array((imgT.size(3), imgT.size(2)))
    target_size = np.array(ps_int2_repeat(target_size))
    if np.all(cur_size == target_size):
        return imgT, np.ones(shape=2), np.zeros(shape=2)
    detla_size = target_size - cur_size
    pad_before = detla_size // 2
    pad_after = detla_size - pad_before
    imgT = F.pad(imgT, pad=(pad_before[0], pad_after[0], pad_before[1], pad_after[1]), value=pad_val)
    scale = np.ones(shape=2)
    return imgT, scale, pad_before


def imgT_pad_aspect(imgT: torch.Tensor, aspect: float, pad_val: float = 0.5) \
        -> (torch.Tensor, np.ndarray):
    _, _, h, w = imgT.size()
    if w / h > aspect:
        target_size = (w, round(w / aspect))
    elif w / h < aspect:
        target_size = (round(h * aspect), h)
    else:
        target_size = (w, h)
    return imgT_pad(imgT, target_size=target_size, pad_val=pad_val)


def imgT_pad_lmtsize(imgT: torch.Tensor, max_size: TV_Int2, pad_val: float = 0.5) \
        -> (torch.Tensor, np.ndarray, np.ndarray):
    cur_size = np.array((imgT.size(3), imgT.size(2)))
    max_size = np.array(ps_int2_repeat(max_size))
    if np.all(cur_size == max_size):
        return imgT, np.ones(shape=2), np.zeros(shape=2)
    scale = np.min(max_size / cur_size)
    imgT = F.interpolate(imgT, size=(int(imgT.size(2) * scale), int(imgT.size(3) * scale)))
    imgT, s0, bias = imgT_pad(imgT, target_size=tuple(max_size), pad_val=pad_val)
    return imgT, scale * s0, bias


def imgT_pad_divide(imgT: torch.Tensor, divisor: TV_Int2, pad_val: float = 0.5) \
        -> (torch.Tensor, np.ndarray, np.ndarray):
    cur_size = np.array((imgT.size(3), imgT.size(2)))
    divisor = np.array(ps_int2_repeat(divisor))
    target_size = ((cur_size + divisor - 1) // divisor) * divisor
    return imgT_pad(imgT, target_size=target_size, pad_val=pad_val)


def imgs2imgsT_pad_lmtsize(imgs: _IMGS_GEN, max_size: TV_Int2, pad_val: float = 0.5, device: torch.device = DEVICE) \
        -> (torch.Tensor, np.ndarray, np.ndarray):
    if isinstance(imgs, np.ndarray):
        return imgs2imgsT_pad_lmtsize(arrsN2arrsT(imgs, device=device), max_size=max_size, pad_val=pad_val,
                                      device=device)
    elif isinstance(imgs, torch.Tensor):
        imgs = imgs.to(device)
        imgsT, scale, bias = imgT_pad_lmtsize(imgT=imgs, max_size=max_size, pad_val=pad_val)
        scale = np.repeat(scale[None], axis=0, repeats=imgs.size(0))
        bias = np.repeat(bias[None], axis=0, repeats=imgs.size(0))
        return imgsT, scale, bias
    else:
        assert len(imgs) > 0, 'no img'
        imgsT = []
        scales = []
        biass = []
        for img in imgs:
            imgT = img2imgT(img, device=device)
            imgT, scale, bias = imgT_pad_lmtsize(imgT=imgT, max_size=max_size, pad_val=pad_val)
            imgsT.append(imgT)
            scales.append(scale)
            biass.append(bias)
        imgsT = torch.cat(imgsT, dim=0)
        scales = np.stack(scales, axis=0)
        biass = np.stack(biass, axis=0)
        return imgsT, scales, biass


def imgs2imgsT_pad_divide(imgs: _IMGS_GEN, divisor: TV_Int2, pad_val: float = 0.5, device: torch.device = DEVICE) \
        -> (torch.Tensor, np.ndarray, np.ndarray):
    if isinstance(imgs, np.ndarray):
        return imgs2imgsT_pad_divide(arrsN2arrsT(imgs, device=device), divisor=divisor, pad_val=pad_val, device=device)
    elif isinstance(imgs, torch.Tensor):
        imgs = imgs.to(device)
        imgsT, scale, bias = imgT_pad_divide(imgT=imgs, divisor=divisor, pad_val=pad_val)
        return imgsT, np.repeat(scale[None], repeats=len(imgs)), np.repeat(bias[None], repeats=len(imgs))
    else:
        assert len(imgs) > 0, 'no img'
        img_sizes = np.array([img2size(img) for img in imgs])
        max_size = np.max(img_sizes, axis=0)
        target_size = ((max_size + divisor - 1) // divisor) * divisor
        imgsT = []
        scales = []
        biass = []
        for img in imgs:
            imgT = img2imgT(img, device=device)
            imgT, scale, bias = imgT_pad(imgT=imgT, target_size=target_size, pad_val=pad_val)
            imgsT.append(imgT)
            scales.append(scale)
            biass.append(bias)
        imgsT = torch.cat(imgsT, dim=0)
        scales = np.stack(scales, axis=0)
        biass = np.stack(biass, axis=0)
        return imgsT, scales, biass


def linear_reverse(scale: np.ndarray, bias: np.ndarray) -> (np.ndarray, np.ndarray):
    return 1 / scale, -bias / scale


def is_linear_equal(scale: np.ndarray, bias: np.ndarray) -> bool:
    return np.all(scale == 1) and np.all(bias == 0)


def img2size(img: _IMG_GEN) -> Tuple:
    if isinstance(img, PIL.Image.Image):
        return img.size
    elif isinstance(img, np.ndarray):
        return (img.shape[1], img.shape[0])
    elif isinstance(img, torch.Tensor):
        return (img.size(-1), img.size(-2))
    else:
        raise Exception('err type ' + img.__class__.__name__)


def imgs2img_sizes(imgs) -> list:
    return [img2size(img) for img in imgs]


BIAS_IDENTITY = np.zeros(shape=2)
SCALE_IDENTIIY = np.ones(shape=2)
HOMOGRAPHY_IDENTITY = np.eye(3)
ROTATION_IDENTITY = np.eye(3)
TRANSLATION_IDENTITY = np.zeros(shape=3)


def imgN_perspective(imgN: np.ndarray, size: tuple, homography: np.ndarray = HOMOGRAPHY_IDENTITY,
                     resample=cv2.INTER_CUBIC, **kwargs):
    imgN = cv2.warpPerspective(imgN.astype(np.uint8), homography, size,
                               flags=resample)
    return imgN


def imgN_linear(imgN: np.ndarray, size: Tuple, bias: np.ndarray = BIAS_IDENTITY, scale: np.ndarray = SCALE_IDENTIIY,
                resample=cv2.INTER_CUBIC, **kwargs):
    affine = np.array([[scale[0], 0, bias[0]], [0, scale[1], bias[1]]])
    imgN = cv2.warpAffine(imgN.astype(np.uint8), affine, size,
                          flags=resample)
    return imgN


def imgN_move(imgN: np.ndarray, size: Tuple, bias: np.ndarray = BIAS_IDENTITY, pad_val: int = 127):
    size_cur = np.array([imgN.shape[1], imgN.shape[0]])
    size = np.array(size)
    bias = np.round(bias).astype(np.int32)
    starts = np.minimum(np.clip(-bias, a_min=0, a_max=None), size_cur)
    ends = np.minimum(np.clip(size - bias, a_min=0, a_max=None), size_cur)
    inter = imgN[starts[1]:ends[1], starts[0]:ends[0]]
    starts_pad = np.clip(bias - starts, a_min=0, a_max=None)
    ends_pad = np.clip(size - bias - ends, a_min=0, a_max=None)
    if np.any(starts_pad) or np.any(ends_pad):
        pad_width = np.array(((starts_pad[1], ends_pad[1]), (starts_pad[0], ends_pad[0]), (0, 0)))
        inter = np.pad(inter, pad_width=pad_width, constant_values=pad_val)
    return inter


def imgs_linear(imgs: Sequence, img_sizes: List, scales: np.ndarray,
                biass: np.ndarray, reverse: bool = False, ) -> Sequence:
    assert len(imgs) == len(scales) and len(imgs) == len(img_sizes), 'len err'
    if np.all(scales == 1) and np.all(biass == 0):
        return imgs
    imgs_scaled = []
    if reverse:
        scales, biass = linear_reverse(scales, biass)
    for img, img_size, scale, bias in zip(imgs, img_sizes, scales, biass):
        imgN = img2imgN(img)
        if np.all(scale == 1):
            imgN = imgN_move(imgN, size=img_size, bias=bias)
        else:
            imgN = imgN_linear(imgN, size=img_size, scale=scale, bias=bias)
        imgs_scaled.append(imgN)
    return imgs_scaled

# </editor-fold>
