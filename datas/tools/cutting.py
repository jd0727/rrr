from utils import *


# <editor-fold desc='图像子区域文件名编码'>
class _FNAME_META_XYXY:
    CORD_FMT = '%04d'
    SEP = '_'

    @staticmethod
    def _xyxyN2str(xyxyN: np.ndarray, sep: str = SEP, cord_fmt: str = CORD_FMT) -> str:
        return sep.join([cord_fmt % v for v in xyxyN])

    @staticmethod
    def enc(meta: str, xyxyN: np.ndarray, sep: str = SEP, cord_fmt: str = CORD_FMT) -> str:
        return sep.join([meta, _FNAME_META_XYXY._xyxyN2str(xyxyN, sep, cord_fmt)])

    @staticmethod
    def dec(meta: str, sep: str = SEP) -> (str, np.ndarray):
        pieces = meta.split(sep)
        meta, xyxy = pieces[:-4], pieces[-4:]
        return sep.join(meta), np.array([int(v) for v in xyxy])


class _FNAME_META_INDEX_XYXY:
    CORD_FMT = '%04d'
    INDEX_FMT = '%02d'
    SEP = '_'

    @staticmethod
    def enc(meta: str, index: int, xyxyN: np.ndarray, sep: str = SEP,
            cord_fmt: str = CORD_FMT, index_fmt: str = INDEX_FMT) -> str:
        return sep.join([meta, index_fmt % index, _FNAME_META_XYXY._xyxyN2str(xyxyN, sep, cord_fmt)])

    @staticmethod
    def dec(meta: str, sep: str = SEP) -> (str, int, np.ndarray):
        pieces = meta.split(sep)
        meta, index, xyxy = pieces[:-5], pieces[-5], pieces[-4:]
        return sep.join(meta), int(index), np.array([int(v) for v in xyxy])


# </editor-fold>

# <editor-fold desc='图像子区域方案生成'>
# 检测框样本区域扩展策略
def _xyxysN_expand(xyxysN_obj: np.ndarray, xyxyN_rgn: np.ndarray, expend_ratio: float = 1.1,
                   as_square: bool = False, align_border: bool = False, expand_min: int = 3) -> np.ndarray:
    xywhs = xyxyN2xywhN(xyxysN_obj)
    xywhs[:, 2:4] = np.maximum(xywhs[:, 2:4] * expend_ratio, xywhs[:, 2:4] + 2 * expand_min)
    if as_square:
        xywhs[:, 2:4] = np.max(xywhs[:, 2:4], keepdims=True, axis=1)
    xywhs = np.round(xywhs).astype(np.int32)
    xyxysN_obj = xywhN2xyxyN(xywhs)
    if align_border:
        xyxysN_obj = xyxyN_clip(xyxysN_obj, xyxyN_rgn=xyxyN_rgn)
    return xyxysN_obj


# 子区域裁剪策略
def _genrgns_persize(xyxyN_rgn: np.ndarray, piece_sizeN: np.ndarray = np.array([640, 640]),
                     over_lapN: np.ndarray = np.array([100, 100]), offsetN: Optional[np.ndarray] = np.array([0, 0]),
                     align_border: bool = True, ) -> np.ndarray:
    step_size = piece_sizeN - over_lapN
    if offsetN is None:
        offsetN = -(np.random.rand(2) * step_size).astype(np.int32)
    else:
        offsetN = offsetN % step_size
        offsetN = np.where(offsetN > 0, offsetN - step_size, offsetN)
    full_size = xyxyN_rgn[2:4] - xyxyN_rgn[:2]
    assert np.all(step_size > 0), 'size err'
    nwh = np.ceil((full_size - offsetN - over_lapN) / step_size).astype(np.int32)
    nwh = np.clip(nwh, a_min=1, a_max=None)
    idys, idxs = arange2dN(nwh[1], nwh[0])
    ids = np.stack([idxs, idys], axis=2).reshape(-1, 2)
    xy1s = ids * step_size + xyxyN_rgn[:2] + offsetN
    if align_border:
        xy1s = xyN_clip(xy1s, np.concatenate([xyxyN_rgn[:2], full_size - piece_sizeN], axis=0))
    rgns = np.concatenate([xy1s, xy1s + piece_sizeN], axis=1)
    if align_border:
        rgns = xyxyN_clip(rgns, xyxyN_rgn=xyxyN_rgn)
    return rgns


def _xyxysN_rgn_expby_xyxysN_obj(xyxysN_rgn: np.ndarray, xyxysN_obj: np.ndarray) -> np.ndarray:
    if len(xyxysN_obj) == 0:
        return xyxysN_rgn
    iselt = np.arange(0, xyxysN_rgn.shape[0])
    while len(iselt) > 0:
        xyxysN_rgncur = xyxysN_rgn[iselt]
        xyxysN_rgncpy = copy.deepcopy(xyxysN_rgncur)
        iareas = xyxyN_ropr(xyxysN_rgncur[:, None, :], xyxysN_obj[None, :, :], opr_type=OPR_TYPE.AREAI)
        irates_rgn = iareas / xyxyN2areaN(xyxysN_rgncur)[:, None]
        irates_obj = iareas / xyxyN2areaN(xyxysN_obj)[None, :]

        fltr_cuted = (irates_obj > 0.0) + (irates_rgn > 0.0)
        xy_obj_min = np.min(np.where(fltr_cuted[:, :, None], xyxysN_obj[None, :, :2],
                                     xyxysN_rgncur[:, None, :2]), axis=1)
        xy_obj_max = np.max(np.where(fltr_cuted[:, :, None], xyxysN_obj[None, :, 2:],
                                     xyxysN_rgncur[:, None, 2:]), axis=1)

        xyxysN_rgncur[:, :2] = np.minimum(xyxysN_rgncur[:, :2], xy_obj_min)
        xyxysN_rgncur[:, 2:] = np.maximum(xyxysN_rgncur[:, 2:], xy_obj_max)

        updated = ~np.all(xyxysN_rgncpy == xyxysN_rgncur, axis=1)
        xyxysN_rgn[iselt] = xyxysN_rgncur
        iselt = iselt[updated]

    return xyxysN_rgn


def _genrgns_pyramid(xyxyN_rgn: np.ndarray, piece_sizesN: np.ndarray = np.array([[640, 640], [320, 320]]),
                     over_lapsN: np.ndarray = np.array([[100, 100], [50, 50]]),
                     offsetsN: Optional[np.ndarray] = np.array([[0, 0], [0, 0]]),
                     align_border: bool = True) -> np.ndarray:
    rgns = [np.zeros(shape=(0, 4), dtype=np.float32)]
    for i, (piece_size, over_lap) in enumerate(zip(piece_sizesN, over_lapsN)):
        offset = offsetsN[i] if offsetsN is not None else None
        rgns_i = _genrgns_persize(xyxyN_rgn, piece_sizeN=piece_size, over_lapN=over_lap, offsetN=offset,
                                  align_border=align_border)
        rgns.append(rgns_i)
    rgns = np.concatenate(rgns, axis=0)
    return rgns


def _genrgns_bkd(xyxysN_obj: np.ndarray, xyxyN_rgn: np.ndarray,
                 min_size: int = 0, max_size: int = 16, num_repeat: int = 1,
                 as_square: bool = True) -> np.ndarray:
    wh_rgn = xyxyN_rgn[2:4] - xyxyN_rgn[:2]
    _rand_sim = 1 if as_square else 2
    whsN_bkd = (np.random.rand(num_repeat, _rand_sim) * (max_size - min_size) + min_size).astype(np.int32)
    whsN_bkd = np.minimum(whsN_bkd, wh_rgn)
    if as_square:
        whsN_bkd = np.min(whsN_bkd, axis=1, keepdims=True)
    xy1sN_bkd = (np.random.rand(num_repeat, 2) * (wh_rgn - whsN_bkd)).astype(np.int32)
    xyxysN_bkd = np.concatenate([xy1sN_bkd, xy1sN_bkd + whsN_bkd], axis=1)
    iareas = xyxyN_ropr(xyxysN_bkd[:, None, :], xyxysN_obj[None, :, :], opr_type=OPR_TYPE.AREAI)
    no_obj = np.all(iareas == 0, axis=1)
    xyxysN_bkd = xyxysN_bkd[no_obj]
    return xyxysN_bkd


def _xyxysN_rgn_unique(xyxysN_rgn: np.ndarray, unique_thres: float = 0.8):
    # ious = xyxysN_iou_arr(xyxysN_rgn[:, None, :], xyxysN_rgn[None, :, :])
    # areas_rgn = xyxysN2areasN(xyxysN_rgn)
    #
    # has_unq = ~np.any((ious > unique_thres) * (areas_rgn[:, None] > areas_rgn), axis=1)
    # xyxysN_rgn = xyxysN_rgn[has_unq]
    idxs_presv = xyxysN_nms_byarea(xyxysN_rgn, cindsN=None, iou_thres=unique_thres,
                                   iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv=100)
    xyxysN_rgn = xyxysN_rgn[idxs_presv]
    return xyxysN_rgn


def _xyxysN_rgn_empty_prob(xyxysN_rgn: np.ndarray, xyxysN_obj: np.ndarray, empty_prob: float = 0.8):
    iareas = xyxyN_ropr(xyxysN_rgn[:, None, :], xyxysN_obj[None, :, :], opr_type=OPR_TYPE.AREAI)
    has_obj = np.any(iareas > 0, axis=1) + (np.random.rand(iareas.shape[0]) < empty_prob)
    xyxysN_rgn = xyxysN_rgn[has_obj]
    return xyxysN_rgn


# </editor-fold>

# <editor-fold desc='图像裁剪'>
class ImageDataCutter(metaclass=ABCMeta):

    @abstractmethod
    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        pass

    def _cut_img(self, img, xyxysN_rgn: np.ndarray) -> Sequence[Image.Image]:
        imgP = img2imgP(img)
        pieces = []
        for xyxyN_rgn in xyxysN_rgn.astype(np.int32):
            piece = imgP.crop(list(xyxyN_rgn))
            pieces.append(piece)
        return pieces

    def _cut_label(self, label: ImageItemsLabel, xyxysN_rgn: np.ndarray) \
            -> Sequence[ImageItemsLabel]:
        plabels = []
        for xyxyN_rgn in xyxysN_rgn.astype(np.int32):
            meta_piece = _FNAME_META_XYXY.enc(label.meta, xyxyN_rgn)
            piece_size = tuple(xyxyN_rgn[2:] - xyxyN_rgn[:2])

            label_cp = copy.deepcopy(label)
            label_cp.linear_(biasN=-xyxyN_rgn[:2], size=piece_size)
            label_cp.filt_measure_(1)

            pitems = label.__class__(img_size=piece_size, meta=meta_piece)
            pitems.extend(label_cp)
            plabels.append(pitems)
        return plabels

    def _cut_data(self, img, label: ImageItemsLabel, xyxysN_rgn: np.ndarray) \
            -> (Sequence[Image.Image], Sequence[ImageItemsLabel]):
        pieces = self._cut_img(img, xyxysN_rgn)
        plabels = self._cut_label(label, xyxysN_rgn)
        return pieces, plabels

    def cut_data(self, img: Optional, label: ImageItemsLabel):
        xyxysN_rgn = self._generate_xyxysN_rgn(label)
        plabels = self._cut_label(label, xyxysN_rgn)
        if img is not None:
            pieces = self._cut_img(img, xyxysN_rgn)
        else:
            pieces = [None] * len(plabels)
        return pieces, plabels

    def cut_label(self, label: ImageItemsLabel):
        xyxysN_rgn = self._generate_xyxysN_rgn(label)
        return self._cut_label(label, xyxysN_rgn)


class ImageDataCutterPerBox(ImageDataCutter):
    def __init__(self, align_border: bool = True, box_protect: bool = False,
                 expend_ratio: float = 1.0, expand_min: int = 0, as_square: bool = True, unique_thres: float = 0.8):
        self.unique_thres = unique_thres
        self.align_border = align_border
        self.box_protect = box_protect
        self.expend_ratio = expend_ratio
        self.expand_min = expand_min
        self.as_square = as_square

    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
        xyxysN_obj = xyxyN_clip(label.xyxysN, img_rgn)
        xyxysN_rgn = _xyxysN_expand(xyxysN_obj, xyxyN_rgn=img_rgn, expend_ratio=self.expend_ratio,
                                    as_square=self.as_square, align_border=self.align_border,
                                    expand_min=self.expand_min, )
        if self.box_protect:
            xyxysN_rgn = _xyxysN_rgn_expby_xyxysN_obj(xyxysN_rgn, xyxysN_obj)

        if self.unique_thres < 1:
            xyxysN_rgn = _xyxysN_rgn_unique(xyxysN_rgn, self.unique_thres)
        return xyxysN_rgn


class ImageDataCutterPerBoxSingle(ImageDataCutter):
    def __init__(self, align_border: bool = True,
                 expend_ratio: float = 1.0, expand_min: int = 0, as_square: bool = True):
        self.align_border = align_border
        self.expend_ratio = expend_ratio
        self.expand_min = expand_min
        self.as_square = as_square

    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
        xyxysN_obj = xyxyN_clip(label.xyxysN, img_rgn)
        xyxysN_rgn = _xyxysN_expand(xyxysN_obj, xyxyN_rgn=img_rgn, expend_ratio=self.expend_ratio,
                                    as_square=self.as_square, align_border=self.align_border,
                                    expand_min=self.expand_min, )
        return xyxysN_rgn

    def _cut_label(self, label: ImageItemsLabel, xyxysN_rgn: np.ndarray) \
            -> Sequence[ImageItemsLabel]:
        assert len(label) == len(xyxysN_rgn)
        plabels = []
        for xyxyN_rgn, item in zip(xyxysN_rgn.astype(np.int32), label):
            meta_piece = _FNAME_META_XYXY.enc(label.meta, xyxyN_rgn)
            piece_size = tuple(xyxyN_rgn[2:] - xyxyN_rgn[:2])

            item_cp = copy.deepcopy(item)
            item_cp.linear_(biasN=-xyxyN_rgn[:2], size=piece_size)

            pitems = label.__class__(img_size=piece_size, meta=meta_piece)
            pitems.append(item_cp)
            plabels.append(pitems)
        return plabels


class ImageDataCutterBackground(ImageDataCutter):
    def __init__(self, min_size: int = 0, max_size: int = 16, num_repeat: int = 1, as_square: bool = True,
                 unique_thres: float = 0.8):
        self.min_size = min_size
        self.max_size = max_size
        self.num_repeat = num_repeat
        self.as_square = as_square
        self.unique_thres = unique_thres

    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
        xyxysN_obj = xyxyN_clip(label.xyxysN, img_rgn)
        xyxysN_rgn = _genrgns_bkd(xyxysN_obj, img_rgn, min_size=self.min_size, max_size=self.max_size,
                                  num_repeat=self.num_repeat, as_square=self.as_square)
        if self.unique_thres < 1:
            xyxysN_rgn = _xyxysN_rgn_unique(xyxysN_rgn, self.unique_thres)
        return xyxysN_rgn

    def _cut_label(self, label: ImageItemsLabel, xyxysN_rgn: np.ndarray) \
            -> Sequence[ImageItemsLabel]:
        plabels = []
        for xyxyN_rgn in xyxysN_rgn.astype(np.int32):
            meta_piece = _FNAME_META_XYXY.enc(label.meta, xyxyN_rgn)
            piece_size = tuple(xyxyN_rgn[2:] - xyxyN_rgn[:2])
            pitems = label.__class__(img_size=piece_size, meta=meta_piece)
            plabels.append(pitems)
        return plabels


class ImageDataCutterPerSize(ImageDataCutter):

    def __init__(self, piece_size: Tuple[int, ...] = (640, 640), over_lap: Tuple[int, ...] = (100, 100),
                 offset: Tuple[int, ...] = (0, 0), empty_prob: float = 0.0, align_border: bool = True,
                 box_protect: bool = False, unique_thres: float = 0.8):
        self.unique_thres = unique_thres
        self.empty_prob = empty_prob
        self.align_border = align_border
        self.piece_size = piece_size
        self.offset = offset
        self.over_lap = over_lap
        self.box_protect = box_protect

    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
        xyxysN_rgn = _genrgns_persize(
            img_rgn, piece_sizeN=np.array(self.piece_size),
            offsetN=np.array(self.offset) if self.offset is not None else None,
            over_lapN=np.array(self.over_lap), align_border=self.align_border).astype(np.int32)
        xyxysN_obj = xyxyN_clip(label.xyxysN, img_rgn)
        if self.empty_prob is not None and self.empty_prob < 1:
            xyxysN_rgn = _xyxysN_rgn_empty_prob(xyxysN_rgn, xyxysN_obj, self.empty_prob)

        if self.box_protect:
            xyxysN_rgn = _xyxysN_rgn_expby_xyxysN_obj(xyxysN_rgn, xyxysN_obj)

        if self.unique_thres is not None and self.unique_thres < 1:
            xyxysN_rgn = _xyxysN_rgn_unique(xyxysN_rgn, self.unique_thres)

        return xyxysN_rgn


class PSIZES:
    S_0 = (0, 0)
    S2_0 = ((0, 0), (0, 0))
    S3_0 = ((0, 0), (0, 0), (0, 0))
    S4_0 = ((0, 0), (0, 0), (0, 0), (0, 0))
    S_32 = (32, 32)
    S2_32 = ((32, 32), (64, 64))
    S3_32 = ((32, 32), (64, 64), (128, 128))
    S4_32 = ((32, 32), (64, 64), (128, 128), (256, 256))
    S5_32 = ((32, 32), (64, 64), (128, 128), (256, 256), (512, 512))
    S_64 = (64, 64)
    S2_64 = ((64, 64), (128, 128))
    S3_64 = ((64, 64), (128, 128), (256, 256))
    S4_64 = ((64, 64), (128, 128), (256, 256), (512, 512))
    S5_64 = ((64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024))
    S_128 = (128, 128)
    S2_128 = ((128, 128), (256, 256))
    S3_128 = ((128, 128), (256, 256), (512, 512))
    S4_128 = ((128, 128), (256, 256), (512, 512), (1024, 1024))
    S_160 = (160, 160)
    S2_160 = ((160, 160), (320, 320))
    S3_160 = ((160, 160), (320, 320), (640, 640))
    S4_160 = ((160, 160), (320, 320), (640, 640), (1280, 1280))
    S_256 = (256, 256)
    S2_256 = ((256, 256), (512, 512))
    S3_256 = ((256, 256), (512, 512), (1024, 1024))
    S4_256 = ((256, 256), (512, 512), (1024, 1024), (2048, 2048))
    S_512 = (512, 512)
    S2_512 = ((512, 512), (1024, 1024))
    S3_512 = ((512, 512), (1024, 1024), (2048, 2048))
    S4_512 = ((512, 512), (1024, 1024), (2048, 2048), (4096, 4096))
    S_640 = (640, 640)
    S2_640 = ((640, 640), (1280, 1280))
    S3_640 = ((640, 640), (1280, 1280), (2560, 2560))
    S4_640 = ((640, 640), (1280, 1280), (2560, 2560), (5120, 5120))
    S_1024 = (1024, 1024)
    S2_1024 = ((1024, 1024), (2048, 2048))
    S3_1024 = ((1024, 1024), (2048, 2048), (4096, 4096))
    S4_1024 = ((1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192))
    S2_1536 = ((1536, 1536), (3072, 3072))
    S3_1536 = ((1536, 1536), (3072, 3072), (6144, 6144))
    S_2048 = (2048, 2048)
    S2_2048 = ((2048, 2048), (4096, 4096))
    S3_2048 = ((2048, 2048), (4096, 4096), (8192, 8192))


class ImageDataCutterPyramid(ImageDataCutter):

    def __init__(self, piece_sizes: Tuple[Tuple[int, ...], ...] = PSIZES.S2_640,
                 over_laps: Tuple[Tuple[int, ...], ...] = PSIZES.S2_160,
                 offsets: Optional[Tuple[Tuple[int, ...], ...]] = PSIZES.S2_0, empty_prob: Optional[float] = 0.0,
                 align_border: bool = True,
                 box_protect: bool = False, unique_thres: Optional[float] = 0.8):
        self.empty_prob = empty_prob
        self.unique_thres = unique_thres
        self.align_border = align_border
        self.piece_sizes = piece_sizes
        self.offsets = offsets
        self.over_laps = over_laps
        self.box_protect = box_protect

    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        img_rgn = np.array([0, 0, label.img_size[0], label.img_size[1]])
        xyxysN_rgn = _genrgns_pyramid(
            img_rgn, piece_sizesN=np.array(self.piece_sizes),
            offsetsN=np.array(self.offsets) if self.offsets is not None else None,
            over_lapsN=np.array(self.over_laps), align_border=self.align_border).astype(np.int32)

        xyxysN_obj = xyxyN_clip(label.xyxysN, img_rgn)
        if self.empty_prob is not None and self.empty_prob < 1:
            xyxysN_rgn = _xyxysN_rgn_empty_prob(xyxysN_rgn, xyxysN_obj, self.empty_prob)

        if self.box_protect:
            xyxysN_rgn = _xyxysN_rgn_expby_xyxysN_obj(xyxysN_rgn, xyxysN_obj)

        if self.unique_thres is not None and self.unique_thres < 1:
            xyxysN_rgn = _xyxysN_rgn_unique(xyxysN_rgn, self.unique_thres)
        return xyxysN_rgn


class ImageDataCutterDefine(ImageDataCutter):
    def __init__(self, rgn_mapper: Dict[str, np.ndarray]):
        self.rgn_mapper = rgn_mapper
        pass

    def _generate_xyxysN_rgn(self, label: ImageItemsLabel) -> np.ndarray:
        meta = label.meta
        if meta in self.rgn_mapper.keys():
            return self.rgn_mapper[meta]
        else:
            img_size = label.img_size
            return np.array([[0, 0, img_size[0], img_size[1]]])

    @staticmethod
    def imitate_from_pmetas(pmetas: List[str]):
        rgn_mapper = {}
        for pmeta in pmetas:
            meta, rgn = _FNAME_META_XYXY.dec(pmeta)
            if meta in rgn_mapper.keys():
                rgn_mapper[meta] = np.concatenate([rgn_mapper[meta], rgn[None]], axis=0)
            else:
                rgn_mapper[meta] = rgn[None]
        return ImageDataCutterDefine(rgn_mapper)

    @staticmethod
    def imitate_from_dir(ref_dir: str):
        file_names = os.listdir(ref_dir)
        pmetas = [os.path.splitext(fn)[0] for fn in file_names]
        return ImageDataCutterDefine.imitate_from_pmetas(pmetas)


# </editor-fold>

# <editor-fold desc='标签聚合'>


class LabelMerger(metaclass=ABCMeta):
    @abstractmethod
    def merge_label(self, labels: Sequence[ImageItemsLabel], xyxysN_rgn: np.ndarray, meta: Optional[str] = None,
                    img_size: Optional[tuple] = None) -> ImageItemsLabel:
        pass


def _label_merge_stack(labels: Sequence[ImageItemsLabel], xyxysN_rgn: np.ndarray, meta: Optional[str] = None,
                       img_size: Optional[Tuple] = None, edge_thres: Optional[float] = 20,
                       scale_thres: Optional[float] = 20):
    if img_size is None:
        img_size = tuple(np.max(xyxysN_rgn[:, 2:4], axis=0).astype(np.int32))
    img_rgn_all = np.array([0, 0, img_size[0], img_size[1]])
    fake_edges = ~(xyxysN_rgn == img_rgn_all)
    #
    label_mrd = ImageItemsLabel(img_size=img_size, meta=meta)
    for label, xyxyN_rgn, fake_edge in zip(labels, xyxysN_rgn, fake_edges):
        # 筛选标签
        img_rgn_cur = np.array([0, 0, label.img_size[0], label.img_size[1]])
        xyxys_itm = label.xyxysN
        presv_msk = np.full(shape=xyxys_itm.shape[0], fill_value=True)
        if edge_thres is not None and edge_thres > 0:
            near_egde = np.any((np.abs(xyxys_itm - img_rgn_cur) < edge_thres) * fake_edge, axis=-1)
            presv_msk *= (~near_egde)
        if scale_thres is not None and scale_thres > 0:
            too_small = np.any(xyxys_itm[..., 2:4] - xyxys_itm[..., :2] < scale_thres, axis=-1)
            presv_msk *= (~too_small)
        # 移动标签
        img_size_cur = np.array(label.img_size)
        rgn_size = xyxyN_rgn[2:] - xyxyN_rgn[:2]
        label.linear_(biasN=xyxyN_rgn[:2], scaleN=rgn_size / img_size_cur, size=tuple(xyxyN_rgn[2:]))
        label_mrd.extend([itm for itm, msk in zip(label, presv_msk) if msk])
    return label_mrd


class LabelMergerStack(LabelMerger):

    def __init__(self, edge_thres: Optional[float] = None, scale_thres: Optional[float] = None):
        self.edge_thres = edge_thres
        self.scale_thres = scale_thres

    def merge_label(self, labels: Sequence[ImageItemsLabel], xyxysN_rgn: np.ndarray, meta: Optional[str] = None,
                    img_size: Optional[tuple] = None):
        label = _label_merge_stack(labels, xyxysN_rgn=xyxysN_rgn, meta=meta, img_size=img_size,
                                   edge_thres=self.edge_thres, scale_thres=self.scale_thres)
        return label


class LabelMergerNMS(LabelMerger):

    def __init__(self, iou_thres: float = 0.4, cluster_index=CLUSTER_INDEX.CLASS, iou_type=IOU_TYPE.IRATE2,
                 nms_orderby=NMS_ORDERBY.CONF, edge_thres: Optional[float] = None,
                 scale_thres: Optional[float] = None):
        self.iou_thres = iou_thres
        self.cluster_index = cluster_index
        self.iou_type = iou_type
        self.nms_orderby = nms_orderby
        self.edge_thres = edge_thres
        self.scale_thres = scale_thres

    def merge_label(self, labels: Sequence[ImageItemsLabel], xyxysN_rgn: np.ndarray, meta: Optional[str] = None,
                    img_size: Optional[tuple] = None):
        label = _label_merge_stack(labels, xyxysN_rgn=xyxysN_rgn, meta=meta, img_size=img_size,
                                   edge_thres=self.edge_thres, scale_thres=self.scale_thres)
        label = items_nms(label, iou_thres=self.iou_thres, cluster_index=self.cluster_index,
                          iou_type=self.iou_type, nms_orderby=self.nms_orderby, nms_type=NMS_TYPE.HARD)
        return label

# </editor-fold>


# if __name__ == '__main__':
#     uname = platform.uname()
#     print(uname)
