from .base import *
from .albu import *
from .mutimix import *


#  <editor-fold desc='增广序列'>


class AugNorm(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [LargestMaxSize(max_size=img_size),
                 A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS,
                               border_mode=pad_mode, always_apply=True),
                 # A.ToGray(p=1.0),
                 ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV1(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, scale=(0.5, 1.5),
                 **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(scale=scale, p=p, cval=PAD_CVALS, mode=pad_mode, keep_ratio=True),
                A_HorizontalFlip(p=0.5),
                A.ToGray(p=0.2),
            ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV1R(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        assert img_size[1] == img_size[0], 'A.RandomResizedCrop has bug if rect'
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.02, p=p),
                A.ToGray(p=0.2),
                A_HorizontalFlip(p=0.5),
                A.RandomRotate90(p=1.0),
            ]),
            ItemsFiltMeasure(thres=thres)
        ]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV2(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(scale=(0.8, 1.2), p=p, cval=PAD_CVALS, mode=pad_mode),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV3(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(shear=(-5, 5), scale=(0.5, 1.5), p=p,
                         translate_percent=(-0.2, 0.2), cval=PAD_CVALS, mode=pad_mode),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.02, p=p),
                A_HorizontalFlip(p=0.5), ]),

            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV3C(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, **kwargs):
        assert img_size[1] == img_size[0], 'A.RandomResizedCrop has bug if rect'
        trans = [
            A_Compose([
                A.RandomResizedCrop(height=img_size[1], width=img_size[0], always_apply=True, scale=(0.4, 1.0),
                                    ratio=(0.75, 1.33)),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.02, p=p),
                A.ToGray(p=0.2),
                # A_HorizontalFlip(p=0.5),
                A.RandomRotate90(p=1.0),
            ]),
            ItemsFiltMeasure(thres=thres)
        ]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV3R(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, scale=(0.3, 3.0),
                 rotate=(-180, 180), shear=(-5, 5), translate_percent=(-0.0, 0.0), **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A.Affine(shear=shear, scale=scale, p=p, rotate=rotate, keep_ratio=True,
                         translate_percent=translate_percent, cval=PAD_CVALS, mode=pad_mode),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5),
                A_VerticalFlip(p=0.5),
                A.GaussianBlur(p=0.2),
                A.MedianBlur(p=0.2),
                A.ToGray(p=0.2),
                A.GaussNoise(p=0.2, var_limit=(10.0, 400.0)),
                A.CLAHE(p=0.2),
            ]),
            ItemsFiltMeasure(thres=thres)
        ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV4(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size: Tuple, thres: float = 10, to_tensor: bool = True,
                 num_repeat: Union[float, int] = 0.2, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size, thres=thres),
            Mosaic(num_repeat=num_repeat, img_size=img_size, blend_type=BLEND_TYPE.REPLACE, pad_val=PAD_CVALS, ),
            A_Compose([
                A.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1, always_apply=True),
                A_HorizontalFlip(p=0.5), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV5(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True, num_repeat: Union[float, int] = 1.0,
                 blend_type=BLEND_TYPE.REPLACE, **kwargs):
        trans = [
            # LargestMaxSize(max_size=img_size),
            Mosaic(num_repeat=num_repeat, img_size=img_size, scale_aspect=(0.8, 1.2),
                   scale_base=(1 / 3, 3), blend_type=BLEND_TYPE.COVER, pad_val=PAD_CVALS, ),
            A_Compose([
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01), ]),
            ItemsFiltMeasure(thres=thres)]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugV5R(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True,
                 num_repeat: Union[float, int] = 1.0, cen_range=0.8, rotate=(-180, 180), scale=(0.5, 2),
                 pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            # LargestMaxSize(max_size=img_size),
            Mosaic(num_repeat=num_repeat, img_size=img_size, scale_aspect=(0.8, 1.2),
                   scale_base=(1 / 3, 3), blend_type=BLEND_TYPE.COVER, pad_val=PAD_CVALS, ),
            A_Compose([
                # A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                #               always_apply=True),
                A_Affine(rotate=rotate, shear=(-5, 5), cval=PAD_CVALS, mode=cv2.BORDER_CONSTANT,
                         interpolation=cv2.INTER_CUBIC, p=p),
                A.ColorJitter(brightness=0.4, contrast=0.1, saturation=0.7, hue=0.1, p=p),
                A_HorizontalFlip(p=0.5),
                A_VerticalFlip(p=0.5),
                A.GaussianBlur(p=0.2),
                A.MedianBlur(p=0.2),
                A.ToGray(p=0.2),
                A.GaussNoise(p=0.2, var_limit=(10.0, 400.0)),
                A.CLAHE(p=0.2),
            ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugAffine(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=1, to_tensor=True,
                 rotate=(-0, 0), scale=(0.5, 2), pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A_Affine(rotate=rotate, scale=scale, cval=PAD_CVALS, mode=cv2.BORDER_CONSTANT,
                         interpolation=cv2.INTER_CUBIC, p=p),
                A_HorizontalFlip(p=0.5),
            ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugRigid(SizedCompose, BatchedDataCompose):
    def __init__(self, img_size, thres=1, p=0.5, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT, **kwargs):
        trans = [
            LargestMaxSize(max_size=img_size),
            A_Compose([
                A_PadIfNeeded(min_height=img_size[1], min_width=img_size[0], value=PAD_CVALS, border_mode=pad_mode,
                              always_apply=True),
                A_RandomRotate90(p=p),
                A_HorizontalFlip(p=0.5),
                A_VerticalFlip(p=0.5),
            ]),
            ItemsFiltMeasure(thres=thres)]

        if to_tensor:
            trans.append(ToTensor(concat=True))
        BatchedDataCompose.__init__(self, *trans)


class AugINStd(SizedCompose):
    def __init__(self, img_size, thres=1, to_tensor=True, pad_mode=cv2.BORDER_CONSTANT,
                 interpolation=cv2.INTER_LINEAR, **kwargs):
        opt_trans = [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, always_apply=True, ),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, ),
            A.Posterize(num_bits=(2, 6), always_apply=True, ),
            A.Solarize(threshold=128, always_apply=True, ),
            A.InvertImg(always_apply=True, ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,
                                       always_apply=True, ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True, ),
            A.GaussianBlur(blur_limit=(3, 7), always_apply=True, ),
            A.Affine(scale=(0.4, 1.6), translate_percent=(-0.2, 0.2),
                     rotate=(-np.pi / 4, np.pi / 4), shear=(-0.1, 0.1), interpolation=1, always_apply=True, ),
            A_GridDistortion(num_steps=5, distort_limit=0.4, interpolation=1, border_mode=4, always_apply=True, ),
        ]
        W, H = img_size
        trans = [
            A_Compose([
                A.RandomResizedCrop(height=H, width=W, scale=(0.4, 1.2), ratio=(0.75, 1.33),
                                    interpolation=interpolation, always_apply=True),
                A_HorizontalFlip(p=0.5),
                A.OneOf(opt_trans, p=0.5)
            ]),
            # CutMix(repeat=1.0),
            # Mixup(repeat=1.0, mix_rate=(0.2, 0.2)),
        ]
        if to_tensor:
            trans.append(ToTensor(concat=True))
        SizedCompose.__init__(self, *trans)
# </editor-fold>
