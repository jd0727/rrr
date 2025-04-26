from itertools import chain

from ..typings import ps_int_multiply
from ..interface import SettableImageSize
from ..label import *

PAD_CVAL = 127
PAD_CVALS = (PAD_CVAL, PAD_CVAL, PAD_CVAL)


# <editor-fold desc='基本定义'>
class ImgTransform():

    @abstractmethod
    def trans_imgs(self, imgs):
        pass

    @abstractmethod
    def trans_img(self, img):
        pass


class ImgUnrelatedTransform(ImgTransform):

    def trans_imgs(self, imgs):
        return imgs

    def trans_img(self, img):
        return img


class BatchedImgTransform(ImgTransform):

    def trans_img(self, img):
        imgs = self.trans_imgs([img])
        return imgs[0]


class SingleImgTransform(ImgTransform):

    def trans_imgs(self, imgs):
        imgs_aug = [self.trans_img(img) for img in imgs]
        return imgs_aug


class LabelTransform():

    @abstractmethod
    def trans_labels(self, labels):
        pass

    @abstractmethod
    def trans_label(self, label):
        pass


class LabelUnrelatedTransform(LabelTransform):

    def trans_labels(self, labels):
        return labels

    def trans_label(self, label):
        return label


class BatchedLabelTransform(LabelTransform):

    def trans_label(self, img):
        imgs = self.trans_labels([img])
        return imgs[0]


class SingleLabelTransform(LabelTransform):

    def trans_labels(self, imgs):
        imgs_aug = [self.trans_label(img) for img in imgs]
        return imgs_aug


class DataTransform():

    @abstractmethod
    def trans_datas(self, imgs, labels):
        pass

    @abstractmethod
    def trans_data(self, img, label):
        pass


class BatchedDataTransform(DataTransform):

    def trans_data(self, img, label):
        imgs, labels = self.trans_datas([img], [label])
        img, label = imgs[0], labels[0]
        return img, label


class SingleDataTransform(DataTransform):

    def trans_datas(self, imgs, labels):
        imgs_aug, labels_aug = [], []
        for img, label in zip(imgs, labels):
            img_aug, label_aug = self.trans_data(img, label)
            imgs_aug.append(img_aug)
            labels_aug.append(label_aug)
        return imgs_aug, labels_aug


class LabelUnrelatedDataTransform(LabelUnrelatedTransform, DataTransform, ImgTransform):

    def trans_datas(self, imgs, labels):
        return self.trans_imgs(imgs), labels

    def trans_data(self, img, label):
        return self.trans_img(img), label


class ImgUnrelatedDataTransform(ImgUnrelatedTransform, DataTransform, LabelTransform):

    def trans_datas(self, imgs, labels):
        return imgs, self.trans_labels(labels)

    def trans_data(self, img, label):
        return img, self.trans_label(label)


class SizedTransform(ImgTransform,SettableImageSize):
    pass


class ReInitSizedTransform(SizedTransform):

    def __init__(self, img_size, **kwargs):
        self.kwargs = kwargs
        self._img_size = img_size
        self.transform = self._build_transform(img_size, **self.kwargs)

    def extract_dct(self):
        return dict(kwargs=self.kwargs, img_size=self.img_size, transform=self.transform.extract_dct())

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        if not self._img_size == img_size:
            self._img_size = img_size
            self.transform = self._build_transform(img_size, **self.kwargs)

    def trans_datas(self, imgs, labels):
        imgs, labels = self.transform(imgs, labels)
        return imgs, labels

    @abstractmethod
    def _build_transform(self, img_size, **kwargs):
        pass


# 组合增广
class BatchedDataCompose(list, BatchedDataTransform):

    def __init__(self, *item):
        super().__init__(item)

    def trans_datas(self, imgs, labels):
        for seq in self:
            if isinstance(seq, LabelUnrelatedDataTransform):
                imgs = seq.trans_imgs(imgs)
            elif isinstance(seq, ImgUnrelatedDataTransform):
                labels = seq.trans_labels(labels)
            elif isinstance(seq, DataTransform):
                imgs, labels = seq.trans_datas(imgs, labels)
        return imgs, labels


class SingleDataCompose(list, SingleDataTransform):

    def __init__(self, *item):
        super().__init__(item)

    def trans_data(self, img, label):
        for seq in self:
            if isinstance(seq, LabelUnrelatedDataTransform):
                img = seq.trans_img(img)
            elif isinstance(seq, ImgUnrelatedDataTransform):
                label = seq.trans_label(label)
            elif isinstance(seq, DataTransform):
                img, label = seq.trans_data(img, label)
        return img, label


class SizedCompose(list, SizedTransform):

    @property
    def img_size(self):
        for seq in self:
            if isinstance(seq, SizedTransform):
                return seq.img_size
        return None

    @img_size.setter
    def img_size(self, img_size):
        for seq in self:
            if isinstance(seq, SizedTransform):
                seq.img_size = img_size


# </editor-fold>


# <editor-fold desc='多图像混合'>


class BLEND_TYPE:
    APPEND = 'appendx'
    REPLACE = 'replace'
    COVER = 'cover'
    COVER_SRC_ORD = 'cover_ord'
    COVER_SRC_RAND = 'cover_rand'


def _batched_sampling(total: int, num_repeat: int, num_input: int, different: bool = True):
    num_require = num_repeat * num_input
    if different and num_require <= total:
        inds = np.random.choice(a=total, replace=False, size=num_require)
        indss = np.reshape(inds, (num_repeat, num_input))
    elif different and num_input <= total:
        indss = np.stack([np.random.choice(a=total, replace=False, size=num_input)
                          for _ in range(num_repeat)], axis=0)
    else:
        indss = np.random.choice(a=total, replace=True, size=(num_repeat, num_input))
    return indss


def _batched_blend_scheme(total: int, indss_src: np.ndarray, num_output: int, different: bool = True,
                          blend_type=BLEND_TYPE.APPEND):
    num_batch, num_input = indss_src.shape
    if blend_type == BLEND_TYPE.REPLACE:

        return None
    elif blend_type == BLEND_TYPE.COVER:
        indss_tar = _batched_sampling(total, num_repeat=num_batch, num_input=num_output,
                                      different=different)
        return indss_tar
    elif blend_type == BLEND_TYPE.COVER_SRC_ORD:
        indss_tar = indss_src[:, :num_output]

        return indss_tar
    elif blend_type == BLEND_TYPE.COVER_SRC_RAND:
        indss_tar = [np.random.choice(a=inds_src, replace=num_output > num_input, size=num_output)
                     for inds_src in indss_src]
        indss_tar = np.stack(indss_tar, axis=0)
        return indss_tar
    else:
        raise Exception('err add type')


def _execute_blend(objs_ori: list, objss: list, indss_tar: np.ndarray, blend_type=BLEND_TYPE.APPEND):
    if blend_type == BLEND_TYPE.REPLACE:
        return list(chain(*objss))
    else:
        for objs, inds_tar in zip(objss, indss_tar):
            for obj, ind_tar in zip(objs, inds_tar):
                objs_ori[ind_tar] = obj
        return objs_ori


class MutiMixTransform():

    def __init__(self, num_input: int, num_output: int, num_repeat: Union[float, int] = 3.0,
                 blend_type=BLEND_TYPE.APPEND, different_input: bool = True, different_output: bool = True):
        self.num_input = num_input
        self.num_output = num_output
        self.num_repeat = num_repeat
        self.blend_type = blend_type
        self.different_input = different_input
        self.different_output = different_output

    def generate_scheme(self, total):
        num_repeat = ps_int_multiply(self.num_repeat, reference=total)
        indss_src = _batched_sampling(total, num_repeat=num_repeat, num_input=self.num_input,
                                      different=self.different_input)
        indss_tar = _batched_blend_scheme(total, indss_src=indss_src, num_output=self.num_output,
                                          different=self.different_output, blend_type=self.blend_type)
        return indss_src, indss_tar


class MutiMixDataTransform(MutiMixTransform, BatchedDataTransform):

    @abstractmethod
    def mix_datas(self, imgs, labels):
        pass

    def trans_datas(self, imgs, labels):
        total = len(imgs)
        if total < self.num_input:
            return imgs, labels
        indss_src, indss_tar = self.generate_scheme(total)
        imgss_p = []
        labelss_p = []
        for n in range(len(indss_src)):
            imgs_c = [copy.deepcopy(imgs[int(ind)]) for ind in indss_src[n]]
            labels_c = [copy.deepcopy(labels[int(ind)]) for ind in indss_src[n]]
            imgs_p, labels_p = self.mix_datas(imgs_c, labels_c)
            imgss_p.append(imgs_p)
            labelss_p.append(labels_p)
        labels = _execute_blend(labels, labelss_p, indss_tar, blend_type=self.blend_type)
        imgs = _execute_blend(imgs, imgss_p, indss_tar, blend_type=self.blend_type)
        return imgs, labels


class MutiMixImgTransform(MutiMixTransform, BatchedImgTransform):

    @abstractmethod
    def mix_imgs(self, imgs):
        pass

    def trans_imgs(self, imgs):
        total = len(imgs)
        if total < self.num_input:
            return imgs
        indss_src, indss_tar = self.generate_scheme(total)
        imgss_p = []
        for n in range(len(indss_src)):
            imgs_c = [copy.deepcopy(imgs[int(ind)]) for ind in indss_src[n]]
            imgs_p = self.mix_imgs(imgs_c)
            imgss_p.append(imgs_p)
        imgs = _execute_blend(imgs, imgss_p, indss_tar, blend_type=self.blend_type)
        return imgs

# </editor-fold>
