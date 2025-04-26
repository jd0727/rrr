from .define import *

# <editor-fold desc='基础标签变换'>

# 混合类别标签
from ..typings import ps_int2_repeat


def _blend_cates(cate1: CategoryLabel, cate2: CategoryLabel, mix_rate: float = 0.5) -> CategoryLabel:
    oc0 = OneHotCategory.convert(cate1.category)
    oc2 = OneHotCategory.convert(cate2.category)
    chot = (1 - mix_rate) * oc0._chotN + mix_rate * oc2._chotN
    cate = copy.deepcopy(cate1)
    cate.category = OneHotCategory(chotN=chot)
    return cate


# 混合检测类标签
def _blend_cates_label(cates_label1: ImageItemsLabel, cates_label2: ImageItemsLabel,
                       mix_rate: float = 0.5) -> ImageItemsLabel:
    cates_label1 = copy.deepcopy(cates_label1)
    cates_label2 = copy.deepcopy(cates_label2)
    for cate_cont in cates_label1:
        cate_cont.category.conf_scale_(1 - mix_rate)
    for cate_cont in cates_label2:
        cate_cont.category.conf_scale_(mix_rate)
        cates_label1.append(cate_cont)
    return cates_label1


def imgP_affine(imgP: Image.Image, scale: float = 1.0, angle: float = 0.0, shear: float = 0.0,
                resample=Image.BICUBIC) -> Image.Image:
    img_size = np.array(imgP.size)
    img_size_scled = (img_size * scale).astype(np.int32)
    A = np.array([[np.cos(angle + shear), np.sin(angle + shear)],
                  [-np.sin(angle - shear), np.cos(angle - shear)]]) * scale
    Ai = np.linalg.inv(A)
    bi = img_size / 2 - Ai @ img_size_scled / 2
    data = [Ai[0, 0], Ai[0, 1], bi[0], Ai[1, 0], Ai[1, 1], bi[1]]
    imgP = imgP.transform(size=tuple(img_size_scled), data=data,
                          method=Image.AFFINE, resample=resample, )
    return imgP


# </editor-fold>
# 按透明度混合
class MixAlpha(MutiMixDataTransform):
    def __init__(self, num_repeat: Union[float, int] = 0.2, mix_rate: float = 0.5, blend_type=BLEND_TYPE.COVER,
                 different_input: bool = True, different_output: bool = True):
        MutiMixDataTransform.__init__(self, num_input=2, num_output=1, num_repeat=num_repeat, blend_type=blend_type,
                                      different_input=different_input, different_output=different_output)
        self.mix_rate = mix_rate

    def mix_datas(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        img = (1 - self.mix_rate) * imgs[0] + self.mix_rate * imgs[1]
        if isinstance(labels[0], CategoryLabel):
            label = _blend_cates(labels[0], labels[1], mix_rate=self.mix_rate)
        elif isinstance(labels[0], ImageItemsLabel):
            label = _blend_cates_label(labels[0], labels[1], mix_rate=self.mix_rate)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return [img], [label]


def _rand_uniform_log(low=0.0, high=1.0, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))


# 马赛克增广
class Mosaic(MutiMixDataTransform):
    def __init__(self, num_repeat: Union[float, int] = 0.5, img_size: Tuple = (416, 416),
                 blend_type=BLEND_TYPE.COVER, pad_val=(127, 127, 127),
                 different_input: bool = True, different_output: bool = True,
                 scale_aspect: Sequence[float] = (0.7, 1.3),
                 scale_base: Sequence[float] = (0.35, 1.4),
                 resample=cv2.INTER_CUBIC,
                 center_ratio: Sequence[float] = (0.5, 0.5)):
        MutiMixDataTransform.__init__(self, num_input=4, num_output=1, num_repeat=num_repeat, blend_type=blend_type,
                                      different_input=different_input, different_output=different_output)
        self.img_size = img_size
        self.pad_val = pad_val
        self.scale_aspect = scale_aspect
        self.scale_base = scale_base
        self.resample = resample
        self.center_ratio = center_ratio

    def mix_datas(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        w, h = self.img_size
        # 图像缩放
        whs = np.array([[img.shape[1], img.shape[0]] for img in imgs], dtype=np.float32)
        scales = np.repeat(np.sqrt(w * h / np.prod(whs, axis=-1, keepdims=True)), axis=-1, repeats=2)
        # scales = np.ones_like(whs)
        if self.scale_base is not None:
            scales = scales * _rand_uniform_log(self.scale_base[0], self.scale_base[1], size=whs.shape[0])[:, None]
        if self.scale_aspect is not None:
            scales = scales * _rand_uniform_log(self.scale_aspect[0], self.scale_aspect[1], size=whs.shape[1])[None, :]
        whs = (whs * scales).astype(np.int32)
        # 中心点确定
        l, r = np.max(whs[[0, 1], 0]), w - np.max(whs[[2, 3], 0])
        t, d = np.max(whs[[0, 2], 1]), h - np.max(whs[[1, 3], 1])
        xyxy_outer = np.array([min(l, r), min(t, d), max(l, r), max(t, d)])
        wh_outer = xyxy_outer[2:] - xyxy_outer[:2]
        wp_intvl = xyxy_outer[0] + np.array(self.center_ratio) * wh_outer[0]
        hp_intvl = xyxy_outer[1] + np.array(self.center_ratio) * wh_outer[1]
        wp = int(np.random.uniform(low=wp_intvl[0], high=wp_intvl[1]))
        hp = int(np.random.uniform(low=hp_intvl[0], high=hp_intvl[1]))
        # 定义偏移量
        xyxys_rgn = np.array([
            [wp - whs[0, 0], hp - whs[0, 1], wp, hp],
            [wp - whs[1, 0], hp, wp, hp + whs[1, 1]],
            [wp, hp - whs[2, 1], wp + whs[2, 0], hp],
            [wp, hp, wp + whs[3, 0], hp + whs[3, 1]]]).astype(np.int32)
        xyxys_rgn = xyxyN_clip(xyxys_rgn, np.array([0, 0, w, h]))
        whs_r = xyxys_rgn[:, 2:4] - xyxys_rgn[:, :2]
        xyxys_src = np.array([
            [max(whs_r[0, 0] - wp, 0), max(whs_r[0, 1] - hp, 0), whs_r[0, 0], whs_r[0, 1]],
            [max(whs_r[1, 0] - wp, 0), 0, whs_r[1, 0], min(h - hp, whs_r[1, 1])],
            [0, max(whs_r[2, 1] - hp, 0), min(w - wp, whs_r[2, 0]), whs_r[2, 1]],
            [0, 0, min(w - wp, whs_r[3, 0]), min(h - hp, whs_r[3, 1])]]).astype(np.int32)
        # 整合
        img_sum = np.zeros(shape=(self.img_size[1], self.img_size[0], 3)) + np.array(self.pad_val)
        label_sum = labels[0].empty()
        for i, (img, label, scale, xyxy_src, xyxy_rgn, wh) in enumerate(
                zip(imgs, labels, scales, xyxys_src, xyxys_rgn, whs)):
            if np.any(xyxy_rgn[2:4] - xyxy_rgn[:2] <= 0):
                continue
            label.linear_(scaleN=scale, biasN=xyxy_rgn[2:4] - xyxy_src[2:4], size=(w, h))
            if isinstance(label, StereoItemsLabel):
                label.calibrate_as_(label_sum.camera)
            label.clip_(xyxyN_rgn=xyxy_rgn)
            label_sum.extend(label)
            if not np.all(scale == 1):
                img = cv2.resize(img, dsize=wh.astype(np.int32), interpolation=self.resample)
            img_sum[xyxy_rgn[1]:xyxy_rgn[3], xyxy_rgn[0]:xyxy_rgn[2]] = \
                img[xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
        label_sum.ctx_size = self.img_size
        label_sum.filt_measure_(thres=1)
        return [img_sum.astype(np.uint8)], [label_sum]


def _samp_pair_scale(scale=None, keep_aspect=False, ):
    if scale is None:
        return np.ones(shape=2)
    scale = ps_int2_repeat(scale)
    if keep_aspect:
        scale_smpd = np.random.uniform(low=scale[0], high=scale[1], size=1)
        scale_smpd = np.repeat(scale_smpd, repeats=2)
    else:
        scale_smpd = np.random.uniform(low=scale[0], high=scale[1], size=2)
    return scale_smpd


# 目标区域的裁剪混合
class CutMix(MutiMixDataTransform):
    def __init__(self, num_repeat=0.5, num_patch=2.0, scale=(0.5, 1.5), keep_aspect=False, with_frgd=True,
                 thres_irate=0.2, blend_type=BLEND_TYPE.COVER, different_input=True, different_output=True,
                 resample=cv2.INTER_CUBIC):
        MutiMixDataTransform.__init__(self, num_input=2, num_output=1, num_repeat=num_repeat, blend_type=blend_type,
                                      different_input=different_input, different_output=different_output)
        self.num_patch = num_patch
        self.thres_irate = thres_irate
        self.scale = scale
        self.keep_aspect = keep_aspect
        self.resample = resample
        self.with_frgd = with_frgd

    def cutmix_cates(self, imgs, cates):
        imgs = [img2imgN(img) for img in imgs]
        xyxy_src = xyxyN_samp_by_area(np.array((0, 0, imgs[1].shape[1], imgs[1].shape[0])), aspect=None,
                                      area_ratio=(0.5, 1)).astype(np.int32)
        patch = imgs[1][xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
        aspect = patch.shape[1] / patch.shape[0] if self.keep_aspect else None
        xyxy_dst = xyxyN_samp_by_area(np.array((0, 0, imgs[1].shape[1], imgs[1].shape[0])), aspect=aspect,
                                      area_ratio=(0.5, 1)).astype(np.int32)
        patch = cv2.resize(patch, dsize=xyxy_dst[2:4] - xyxy_dst[:2], interpolation=self.resample)
        imgs[0][xyxy_dst[1]:xyxy_dst[3], xyxy_dst[0]:xyxy_dst[2]] = patch
        mix_rate = np.prod(patch.shape) / np.prod(imgs[0].shape)
        cate = _blend_cates(cates[0], cates[1], mix_rate=mix_rate)
        return imgs[0], cate

    def cutmix_items(self, imgs, labels):
        num_src = len(labels[1])
        num_patch = int(np.ceil(self.num_patch * num_src)) \
            if isinstance(self.num_patch, float) else self.num_patch
        if num_patch == 0:
            return imgs[0], labels[0]
        img_size = labels[0].img_size
        imgs = [img2imgN(img) for img in imgs]
        patches = []
        masks = []
        xyxys_patch = []
        items_patch = []
        inds = np.random.choice(size=num_patch, a=min(num_src, num_patch), replace=True)
        for ind in inds:
            item = copy.deepcopy(labels[1][ind])
            if isinstance(item, BoxItem):
                border = copy.deepcopy(item.border)
                xyxy_src = XYXYBorder.convert(border)._xyxyN.astype(np.int32)
                wh_src = xyxy_src[2:4] - xyxy_src[:2]
                if not self.with_frgd:
                    mask = np.full(shape=wh_src, fill_value=1.0)
                else:
                    border.linear_(biasN=-xyxy_src[:2], size=xyxy_src[2:4] - xyxy_src[:2])
                    mask = border.maskNb.astype(np.float32)
            elif isinstance(item, InstItem):
                rgn = RefValRegion.convert(item.rgn)
                xyxy_src = rgn._xyxyN.astype(np.int32)
                wh_src = xyxy_src[2:4] - xyxy_src[:2]
                if not self.with_frgd:
                    mask = np.full(shape=wh_src, fill_value=1.0)
                else:
                    mask = rgn.maskNb_ref.astype(np.float32)
            else:
                raise Exception('err item')
            patch = imgs[1][xyxy_src[1]:xyxy_src[3], xyxy_src[0]:xyxy_src[2]]
            scale_smpd = _samp_pair_scale(self.scale, self.keep_aspect)
            wh_dst = (wh_src * scale_smpd).astype(np.int32)
            xy_dst = (np.random.rand(2) * (np.array(img_size) - wh_dst)).astype(np.int32)
            bias = xy_dst - xyxy_src[:2] * scale_smpd
            item.linear_(biasN=bias, scaleN=scale_smpd, size=img_size)
            xyxy_dst = np.concatenate([xy_dst, xy_dst + wh_dst], axis=0)
            if not np.all(scale_smpd == 1):
                mask = cv2.resize(mask, wh_dst, interpolation=self.resample)
                patch = cv2.resize(patch, wh_dst, interpolation=self.resample)
            xyxys_patch.append(xyxy_dst)
            masks.append(mask)
            patches.append(patch)
            items_patch.append(item)

        xyxys_dist = labels[0].export_xyxysN()
        # 放置patch
        for i in range(num_patch):
            xyxy_patch = xyxys_patch[i]
            irate = xyxyN_ropr(xyxy_patch[None], xyxys_dist, opr_type=OPR_TYPE.RATEI2)
            if np.max(irate) > self.thres_irate:  # 防止新粘贴的图像影响原有目标
                continue
            imgs[0][xyxy_patch[1]:xyxy_patch[3], xyxy_patch[0]:xyxy_patch[2]] = \
                np.where(masks[i][..., None] > 0, patches[i],
                         imgs[0][xyxy_patch[1]:xyxy_patch[3], xyxy_patch[0]:xyxy_patch[2]])
            labels[0].append(items_patch[i])
            xyxys_dist = np.concatenate([xyxys_dist, xyxy_patch[None]], axis=0)
        return imgs[0], labels[0]

    def mix_datas(self, imgs, labels):
        if isinstance(labels[0], CategoryLabel):
            img, label = self.cutmix_cates(imgs, labels)
        elif isinstance(labels[0], ImageItemsLabel):
            img, label = self.cutmix_items(imgs, labels)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return [img], [label]


# 图片透明度混合
class Mixup(MutiMixDataTransform):
    def __init__(self, num_repeat=0.5, mix_rate=(0.4, 0.6), resample=cv2.INTER_CUBIC,
                 blend_type=BLEND_TYPE.COVER, different_input=True, different_output=True, ):
        MutiMixDataTransform.__init__(self, num_input=2, num_output=1, num_repeat=num_repeat, blend_type=blend_type,
                                      different_input=different_input, different_output=different_output)
        self.mix_rate = mix_rate
        self.resample = resample

    def mixup_cates(self, imgs, cates):
        imgs = [img2imgN(img) for img in imgs]
        mix_rate = np.random.uniform(low=self.mix_rate[0], high=self.mix_rate[1])
        if not imgs[1].shape == imgs[0].shape:
            imgs[1] = cv2.resize(imgs[1], dsize=(imgs[0].shape[1], imgs[0].shape[0]), interpolation=self.resample)
        img_mix = (mix_rate * imgs[0] + (1 - mix_rate) * imgs[1]).astype(np.uint8)
        cate = _blend_cates(cates[0], cates[1], mix_rate=mix_rate)
        return img_mix, cate

    def mixup_items(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        mix_rate = np.random.uniform(low=self.mix_rate[0], high=self.mix_rate[1])
        if not imgs[1].shape == imgs[0].shape:
            size0 = (imgs[0].shape[1], imgs[0].shape[0])
            scale = (imgs[0].shape[1] / imgs[1].shape[1], imgs[0].shape[0] / imgs[1].shape[0])
            imgs[1] = cv2.resize(imgs[1], dsize=size0, interpolation=self.resample)
            labels[1].linear_(scaleN=scale, size=size0)
        img_mix = (mix_rate * imgs[0] + (1 - mix_rate) * imgs[1]).astype(np.uint8)
        label_mix = labels[0].empty()
        for item in labels[0]:
            cate = OneHotCategory.convert(item.category)
            cate.conf_scale_(mix_rate)
            item.category = cate
            label_mix.append(item)

        for item in labels[1]:
            cate = OneHotCategory.convert(item.category)
            cate.conf_scale_(1 - mix_rate)
            item.category = cate
            label_mix.append(item)

        return img_mix, label_mix

    def mix_datas(self, imgs, labels):
        if isinstance(labels[0], CategoryLabel):
            img, label = self.mixup_cates(imgs, labels)
        elif isinstance(labels[0], ImageItemsLabel):
            img, label = self.mixup_items(imgs, labels)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return [img], [label]


# 分割图像前景交换
class ExchangeMix(MutiMixDataTransform):

    def __init__(self, num_repeat=0.3, blend_type=BLEND_TYPE.COVER_SRC_ORD, resample=cv2.INTER_CUBIC,
                 different_input=True, different_output=True):
        MutiMixDataTransform.__init__(self, num_input=2, num_output=2, num_repeat=num_repeat, blend_type=blend_type,
                                      different_input=different_input, different_output=different_output)
        self.resample = resample

    def mix_datas(self, imgs, labels):
        if isinstance(labels[0], ImageItemsLabel):
            imgs, labels = self.exchangemix_items(imgs, labels)
        else:
            raise Exception('err fmt ' + labels[0].__class__.__name__)
        return imgs, labels

    def exchangemix_items(self, imgs, labels):
        imgs = [img2imgN(img) for img in imgs]
        if not imgs[1].shape == imgs[0].shape:
            size0 = (imgs[0].shape[1], imgs[0].shape[0])
            scale = (imgs[0].shape[1] / imgs[1].shape[1], imgs[0].shape[0] / imgs[1].shape[0])
            imgs[1] = cv2.resize(imgs[1], dsize=size0, interpolation=self.resample)
            labels[1].linear_(scaleN=scale, size=size0)

        masks = []
        for label in labels:
            if isinstance(label, BoxesLabel):
                mask = label.export_border_masksN_enc(img_size=label.img_size, num_cls=-1)
            elif isinstance(label, InstsLabel):
                mask = label.export_masksN_enc(img_size=label.img_size, num_cls=-1)
            else:
                raise Exception('err fmt ' + label.__class__.__name__)
            masks.append(mask != -1)

        mask_join = (masks[0] + masks[1])[..., None]
        img0, img1 = imgs
        imgs[0] = np.where(mask_join, img0, img1)
        imgs[1] = np.where(mask_join, img1, img0)
        return imgs, labels

# </editor-fold>
