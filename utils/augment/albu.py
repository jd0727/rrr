from .define import *

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations as A
import skimage


#  <editor-fold desc='Albumentations偏移修改'>
def _trans_data_with_albu(albu_trans, img, label):
    kwargs = dict(image=img2imgN(img))
    if isinstance(label, PointsExtractable):
        xys = label.extract_xysN()
        xys = np.concatenate([xys, np.ones(shape=(xys.shape[0], 2))], axis=1)
        kwargs['keypoints'] = xys
    masksN = []
    has_bool_chan = isinstance(label, BoolMaskExtractable) and label.num_bool_chan > 0
    has_val_chan = isinstance(label, ValMaskExtractable) and label.num_chan > 0
    if has_bool_chan:
        maskN_enc = label.extract_maskNb_enc(index=1)
        masksN.append(maskN_enc)
    if has_val_chan:
        maskN_val = label.extract_maskN()
        masksN.append(maskN_val)
    if len(masksN) > 0:
        masksN = np.concatenate(masksN, axis=2)
        kwargs['mask'] = masksN

    transformed = albu_trans(**kwargs)
    img_aug = transformed['image']
    img_size = (img_aug.shape[1], img_aug.shape[0])
    if isinstance(label, PointsExtractable):
        xys_aug = np.array(transformed['keypoints'])[:, :2]
        label.refrom_xysN(xys_aug, img_size)
    offset = 0
    if has_bool_chan:
        maskN_enc_aug = transformed['mask'][..., 0:1]
        label.refrom_maskNb_enc(maskN_enc_aug, index=1)
        offset = 1
    if has_val_chan:
        maskN_val_aug = transformed['mask'][..., offset:]
        label.refrom_maskN(maskN_val_aug)
    return img_aug, label


def _trans_img_with_albu(albu_trans, img):
    kwargs = dict(image=img2imgN(img))
    transformed = albu_trans(**kwargs)
    img_aug = transformed['image']
    return img_aug


class AlbuInterface(SingleDataTransform, SingleImgTransform):

    def trans_img(self, img):
        return _trans_img_with_albu(self, img)

    def trans_data(self, img, label):
        return _trans_data_with_albu(self, img, label)


class AlbuKeyPointPatch():
    def apply_to_keypoint(self, keypoint, **params):
        keypoint_ofst = (keypoint[0] - 0.5, keypoint[1] - 0.5, keypoint[2], keypoint[3])
        keypoint_trd = super(AlbuKeyPointPatch, self).apply_to_keypoint(
            keypoint_ofst, **params)
        return (keypoint_trd[0] + 0.5, keypoint_trd[1] + 0.5, keypoint_trd[2], keypoint_trd[3])


class A_Compose(A.Compose, AlbuInterface, AlbuKeyPointPatch, SizedTransform):

    def __init__(self, transforms, p: float = 1.0, ):
        keypoint_params = dict(format='xy', remove_invisible=False)
        A.Compose.__init__(self, transforms, keypoint_params=keypoint_params, p=p)

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


class A_HorizontalFlip(A.HorizontalFlip, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_VerticalFlip(A.VerticalFlip, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_Affine(A.Affine, AlbuInterface, AlbuKeyPointPatch):

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        h, w = params["image"].shape[:2]

        translate: Dict[str, Union[int, float]]
        if self.translate_px is not None:
            translate = {key: random.randint(*value) for key, value in self.translate_px.items()}
        elif self.translate_percent is not None:
            translate = {key: random.uniform(*value) for key, value in self.translate_percent.items()}
            translate["x"] = translate["x"] * w
            translate["y"] = translate["y"] * h
        else:
            translate = {"x": 0, "y": 0}

        # Look to issue https://github.com/albumentations-team/albumentations/issues/1079
        shear = {key: -random.uniform(*value) for key, value in self.shear.items()}
        scale = {key: random.uniform(*value) for key, value in self.scale.items()}
        if self.keep_ratio:
            scale["y"] = scale["x"]

        # Look to issue https://github.com/albumentations-team/albumentations/issues/1079
        rotate = -random.uniform(*self.rotate)

        # for images we use additional shifts of (0.5, 0.5) as otherwise
        # we get an ugly black border for 90deg rotations
        shift_x = w / 2 - 0.5
        shift_y = h / 2 - 0.5

        matrix_to_topleft = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_shear_y_rot = skimage.transform.AffineTransform(rotation=-np.pi / 2)
        matrix_shear_y = skimage.transform.AffineTransform(shear=np.deg2rad(shear["y"]))
        matrix_shear_y_rot_inv = skimage.transform.AffineTransform(rotation=np.pi / 2)
        matrix_transforms = skimage.transform.AffineTransform(
            scale=(scale["x"], scale["y"]),
            translation=(translate["x"], translate["y"]),
            rotation=np.deg2rad(rotate),
            shear=np.deg2rad(shear["x"]),
        )
        matrix_to_center = skimage.transform.SimilarityTransform(translation=[shift_x, shift_y])
        matrix = (
                matrix_to_topleft
                + matrix_shear_y_rot
                + matrix_shear_y
                + matrix_shear_y_rot_inv
                + matrix_transforms
                + matrix_to_center
        )
        if self.fit_output:
            matrix, output_shape = self._compute_affine_warp_output_shape(matrix, params["image"].shape)
        else:
            output_shape = params["image"].shape

        return {
            "rotate": rotate,
            "scale": scale,
            "matrix": matrix,
            "output_shape": output_shape,
        }


class A_RandomRotate90(A.RandomRotate90, AlbuInterface, AlbuKeyPointPatch):
    pass


class A_Resize(A.Resize, AlbuInterface, SizedTransform):
    @property
    def img_size(self):
        return (self.width, self.height)

    @img_size.setter
    def img_size(self, img_size):
        self.width, self.height = img_size


class A_RandomResizedCrop(A.RandomResizedCrop, AlbuInterface, SizedTransform):
    @property
    def img_size(self):
        return (self.width, self.height)

    @img_size.setter
    def img_size(self, img_size):
        self.width, self.height = img_size


class A_PadIfNeeded(A.PadIfNeeded, AlbuInterface, SizedTransform):
    @property
    def img_size(self):
        return (self.min_width, self.min_height)

    @img_size.setter
    def img_size(self, img_size):
        self.min_width, self.min_height = img_size


class A_GridDistortion(A.GridDistortion):
    # 旧版本没有实现，这里不做处理
    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

# </editor-fold>
