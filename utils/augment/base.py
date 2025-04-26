from .define import *

# <editor-fold desc='基本功能'>
# 标准tnsor输出
from ..functional.cvting import _size_limt_scale


class ToTensor(LabelUnrelatedDataTransform):

    def __init__(self, concat: bool = True):
        self.concat = concat

    def trans_img(self, img):
        return img2imgT(img)

    def trans_imgs(self, imgs):
        imgs = [img2imgT(img) for img in imgs]
        if self.concat and len(imgs) > 0:
            imgs = torch.cat(imgs, dim=0)
        return imgs


class ItemsFiltMeasure(ImgUnrelatedDataTransform, SingleLabelTransform):

    def __init__(self, thres: float = 1.0, with_clip: bool = False):
        self.thres = thres
        self.with_clip = with_clip

    def trans_label(self, label):
        if self.with_clip:
            label.clip_(xyxyN_rgn=np.array([0, 0, label.img_size[0], label.img_size[1]]))
        if isinstance(label, HasFiltMeasureList):
            label.filt_measure_(thres=self.thres)
        return label


class ItemsFiltCind(ImgUnrelatedDataTransform, SingleLabelTransform):

    def __init__(self, cinds: Iterable = None):
        self.cinds = cinds

    def trans_label(self, label):
        if isinstance(label, HasFiltList):
            label.filt_(lambda item: item.category in self.cinds)
        return label


class ItemsFilt(SingleDataTransform):

    def __init__(self, fltr: Callable = None):
        self.fltr = fltr

    def trans_data(self, img, label):
        if isinstance(label, ImageItemsLabel):
            label.filt_(fltr=self.fltr)
        return img, label


class ConvertItemsToCategoryByMain(SingleLabelTransform, ImgUnrelatedDataTransform):

    def trans_label(self, label):
        cate = IndexCategory(num_cls=1, cindN=0)
        for item in label:
            if item.get('main', False):
                cate = item.category
                break
        label_c = CategoryLabel(category=cate, img_size=label.img_size, meta=label.meta, **label.kwargs)
        label_c.ctx_from(label)
        return label_c


def _get_img_size(img, label):
    assert img is not None or label is not None
    if img is not None:
        return img2size(img)
    else:
        return label.img_size


# 缩放最大边
class LargestMaxSize(SizedTransform, SingleDataTransform):
    def __init__(self, max_size: TV_Int2 = (256, 256), resample=cv2.INTER_CUBIC, thres: float = 10,
                 only_smaller: bool = False, only_larger: bool = False):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres
        self.only_smaller = only_smaller
        self.only_larger = only_larger

    @property
    def img_size(self):
        return self.max_size

    @img_size.setter
    def img_size(self, img_size):
        self.max_size = img_size

    def trans_img(self, img):
        imgN = img2imgN(img)
        imgN_scld, _ = imgN_lmtsize(imgN, max_size=self.max_size, resample=self.resample,
                                    only_smaller=self.only_smaller, only_larger=self.only_larger)
        return imgN_scld

    def trans_data(self, img, label):
        img_size = _get_img_size(img, label)
        scale, final_size = _size_limt_scale(
            size=img_size, max_size=self.max_size,
            only_smaller=self.only_smaller, only_larger=self.only_larger)
        if np.all(scale == 1):
            return img, label
        if img is not None:
            imgN = img2imgN(img)
            img, scale = imgN_lmtsize(
                imgN, max_size=self.max_size, resample=self.resample,
                only_smaller=self.only_smaller, only_larger=self.only_larger)
        if label is not None:
            label.linear_(scaleN=scale, size=final_size)
            if isinstance(label, HasFiltMeasureList):
                label.filt_measure_(thres=self.thres)
        return img, label


class LargestMaxSizeWithPadding(SingleDataTransform):
    def __init__(self, max_size: TV_Int2 = (256, 256), resample=cv2.INTER_CUBIC, thres: float = 10):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres

    def trans_data(self, img, label):
        imgN, label = img2imgN(img), label
        imgN_scld, scale, bias = imgN_pad_lmtsize(imgN, max_size=self.max_size, pad_val=PAD_CVAL,
                                                  resample=self.resample)
        if not is_linear_equal(scale, bias):
            label.linear_(scaleN=scale, biasN=bias, size=img2size(imgN_scld))
        if isinstance(label, HasFiltMeasureList):
            label.filt_measure_(thres=self.thres)
        return imgN_scld, label


class LargestMaxSizeWithPaddingPIL(SingleDataTransform):
    def __init__(self, max_size: TV_Int2 = (256, 256),
                 resample=Image.BILINEAR, thres: float = 10):
        self.max_size = max_size
        self.resample = resample
        self.thres = thres

    def trans_data(self, img, label):
        imgP = img2imgP(img)
        imgP_scld, scale, bias = imgP_pad_lmtsize(imgP, max_size=self.max_size, pad_val=PAD_CVAL,
                                                  resample=self.resample)
        if not is_linear_equal(scale, bias):
            label.linear_(scaleN=scale, biasN=bias, size=img2size(imgP_scld))
        if isinstance(label, HasFiltMeasureList):
            label.filt_measure_(thres=self.thres)
        return imgP_scld, label


# 缩放最大边
class CenterRescale(SingleDataTransform):

    def __init__(self, size: TV_Int2 = (256, 256), expand_ratio: float = 1.0,
                 resample=cv2.INTER_CUBIC, thres: float = 10):
        self.size = size
        self.resample = resample
        self.thres = thres
        self.expand_ratio = expand_ratio

    def trans_data(self, img, label):
        imgN = img2imgN(img)
        img_size = np.array((imgN.shape[1], imgN.shape[0]))
        size = np.array(self.size)
        ratio = min(size / img_size) * self.expand_ratio
        bias = size[0] / 2 - ratio * img_size / 2
        A = np.array([[ratio, 0, bias[0]], [0, ratio, bias[1]]]).astype(np.float32)
        imgN = cv2.warpAffine(imgN.astype(np.float32), A, size, flags=self.resample)
        imgN = np.clip(imgN, a_min=0, a_max=255).astype(np.uint8)
        label.linear_(scaleN=[ratio, ratio], biasN=bias, size=tuple(size))
        if isinstance(label, HasFiltMeasureList):
            label.filt_measure_(thres=self.thres)
        return imgN, label


class ConvertBorderType(SingleLabelTransform, ImgUnrelatedDataTransform):

    def __init__(self, border_type=XYWHABorder):
        self.border_type = border_type

    def trans_label(self, label):
        assert isinstance(label, ImageItemsLabel), 'fmt err ' + label.__class__.__name__
        for j, item in enumerate(label):
            if isinstance(item, BoxItem) or isinstance(item, InstItem):
                item.border = self.border_type.convert(item.border)
        return label


# </editor-fold>


#  <editor-fold desc='cap扩展'>
def _cutsN2intervalsN(cutsN: np.ndarray, low: float = 0.0, high: float = np.inf) -> np.ndarray:
    cuts_min = np.concatenate([[low], cutsN], axis=0)
    cuts_max = np.concatenate([cutsN, [high]], axis=0)
    return np.stack([cuts_min, cuts_max], axis=1)


def _xyxyN_wsplit(xyxyN: np.ndarray, cutsN: np.ndarray) -> np.ndarray:
    ints = _cutsN2intervalsN(cutsN)
    xyxysN_cliped = np.repeat(xyxyN[None], axis=0, repeats=ints.shape[0])
    xyxysN_cliped[:, 0] = np.maximum(xyxysN_cliped[:, 0], ints[:, 0])
    xyxysN_cliped[:, 2] = np.minimum(xyxysN_cliped[:, 2], ints[:, 1])
    return xyxysN_cliped


def _xyxyN_hsplit(xyxyN: np.ndarray, cutsN: np.ndarray) -> np.ndarray:
    ints = _cutsN2intervalsN(cutsN)
    xyxysN_cliped = np.repeat(xyxyN[None], axis=0, repeats=ints.shape[0])
    xyxysN_cliped[:, 1] = np.maximum(xyxysN_cliped[:, 1], ints[:, 0])
    xyxysN_cliped[:, 3] = np.minimum(xyxysN_cliped[:, 3], ints[:, 1])
    return xyxysN_cliped

# </editor-fold>
