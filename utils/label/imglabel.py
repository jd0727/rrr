from .imgitem import *


# <editor-fold desc='图像标签'>
class ImageLabel(SettableImageSize, PointsExtractable, Movable):

    def __init__(self, img_size: tuple, meta: Optional[str] = None, **kwargs):
        self.ctx_border = None
        self._init_size = img_size
        self.ctx_border = XYPBorder(xyxyN2xypN(np.array([0, 0, img_size[0], img_size[1]])), size=img_size)
        self.meta = meta
        self.kwargs = kwargs

    @property
    def img_size(self):
        return self.ctx_border.size

    @img_size.setter
    def img_size(self, img_size):
        self.ctx_border.size = img_size

    @property
    def size(self):
        return self.ctx_border.size

    @size.setter
    def size(self, size):
        self.ctx_border.size = size

    @property
    def ctx_size(self):
        return self.ctx_border.size

    # 重置图像标记区域大小
    @ctx_size.setter
    def ctx_size(self, ctx_size):
        self.ctx_border = XYPBorder(xyxyN2xypN(np.array([0, 0, ctx_size[0], ctx_size[1]])), size=ctx_size)

    @property
    def init_size(self):
        return self._init_size

    # 重置初始图像大小和图像标记区域大小
    @init_size.setter
    def init_size(self, init_size):
        self._init_size = init_size
        self.ctx_size = init_size

    # 通过投影变换恢复标注
    def recover(self):
        self.ctx_size = self.init_size
        return self

    def is_recovered(self):
        return np.all(self.ctx_border._xypN == xyxyN2xypN(np.array([0, 0, self.init_size[0], self.init_size[1]])))

    # 迁移原图坐标位置
    def ctx_from(self, label):
        self._init_size = copy.deepcopy(label._init_size)
        self.ctx_border = copy.deepcopy(label.ctx_border)

    # 迁移原图关键信息
    def info_from(self, label):
        self.ctx_from(label)
        self.meta = copy.deepcopy(label.meta)
        self.kwargs = copy.deepcopy(label.kwargs)

    @property
    def num_xysN(self) -> int:
        return 4

    def extract_xysN(self) -> np.ndarray:
        return self.ctx_border._xypN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self.ctx_border._xypN = xysN
        self.img_size = size
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.ctx_border._xypN = self.ctx_border._xypN * scaleN + biasN
        self.img_size = size
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.img_size = size
        self.ctx_border._xypN = xysN_perspective(self.ctx_border._xypN, homographyN=homographyN)
        return self


# </editor-fold>

# <editor-fold desc='分类标签'>


class CategoryLabel(ImageLabel, dict):

    @property
    def num_chan(self) -> int:
        return 0

    def extract_maskN(self):
        return None

    def refrom_maskN(self, maskN, **kwargs):
        return self

    __slots__ = ('category',)

    def __init__(self, category, img_size=(0, 0), meta=None, *seq, **kwargs):
        ImageLabel.__init__(self, img_size=img_size, meta=meta)
        dict.__init__(self, *seq, **kwargs)
        self.category = category

    def conf_scale_(self, scale):
        self.category.conf_scale_(scale)

    @property
    def num_xysN(self):
        return 4

    @property
    def num_bool_chan(self):
        return 0

    def extract_maskNb(self):
        return None

    def refrom_maskNb(self, maskNb, **kwargs):
        return self

    def __repr__(self):
        return self.category.__repr__() + super(CategoryLabel, self).__repr__()

    def __eq__(self, other):
        return isinstance(other, CategoryLabel) and self.category == other.category


def cates2chotsN(cates: list) -> np.ndarray:
    chotsN = []
    for cate in cates:
        assert isinstance(cate, CategoryLabel), 'class err ' + cate.__class__.__name__
        chotsN.append(OneHotCategory.convert(cate.category)._chotN[None, :])
    chotsN = np.concatenate(chotsN, axis=0)
    return chotsN


def cates2cindsN(cates: list) -> np.ndarray:
    cindsN = []
    for cate in cates:
        assert isinstance(cate, CategoryLabel), 'class err ' + cate.__class__.__name__
        cindsN.append(IndexCategory.convert(cate.category)._cindN)
    cindsN = np.array(cindsN)
    return cindsN


def cates2chotsT(cates: list, device=DEVICE) -> torch.Tensor:
    return arrsN2arrsT(cates2chotsN(cates), device=device)


def cates2cindsT(cates: list, device=DEVICE) -> torch.Tensor:
    return arrsN2arrsT(cates2cindsN(cates), device=device)


def chotsT2cates(chotsT: torch.Tensor, img_size: tuple, cind2name=None) -> list:
    cates = []
    chotsN = chotsT.detach().cpu().numpy()
    for chotN in chotsN:
        category = OneHotCategory(chotN=chotN)
        cate = CategoryLabel(category=category, img_size=img_size)
        if cind2name is not None:
            cate['name'] = cind2name(np.argmax(chotN))
        cates.append(cate)
    return cates


# </editor-fold>

# <editor-fold desc='快速导出工具'>


# 基本导出工具
class BaseExportable(Iterable):
    def export_attrsN(self, aname, default) -> np.ndarray:
        return np.array([getattr(item, aname, default) for item in self])

    def export_valsN(self, key, default) -> np.ndarray:
        return np.array([item.get(key, default) for item in self])

    def export_namesN(self) -> np.ndarray:
        return self.export_valsN('name', 'unknown')

    def export_ignoresN(self) -> np.ndarray:
        return self.export_valsN('ignore', False).astype(bool)

    def export_crowdsN(self) -> np.ndarray:
        return self.export_valsN('crowd', False).astype(bool)


class CategoryExportable(BaseExportable):
    ANMAE_CATEGORY = 'category'

    def orderby_conf_(self, ascend=True):
        confs = self.export_confsN()
        confs = confs if ascend else -confs
        order = np.argsort(confs)
        buffer = self[:]
        for i, ind in enumerate(order):
            self[i] = buffer[ind]
        return self

    def export_cindsN(self, aname_cate=ANMAE_CATEGORY, fltr=None):
        cindsN = []
        for item in (self if fltr is None else filter(fltr, self)):
            cindsN.append(IndexCategory.convert(getattr(item, aname_cate))._cindN)
        cindsN = np.array(cindsN).astype(np.int32)
        return cindsN

    def export_confsN(self, aname_cate=ANMAE_CATEGORY, fltr=None):
        confsN = []
        for item in (self if fltr is None else filter(fltr, self)):
            confsN.append(getattr(item, aname_cate).confN)
        confsN = np.array(confsN)
        return confsN

    def export_cindsN_confsN(self, aname_cate=ANMAE_CATEGORY, fltr=None):
        confsN = []
        cindsN = []
        for item in (self if fltr is None else filter(fltr, self)):
            cate = IndexCategory.convert(getattr(item, aname_cate))
            cindsN.append(cate._cindN)
            confsN.append(cate.confN)
        confsN = np.array(confsN)
        cindsN = np.array(cindsN)
        return cindsN, confsN

    def export_chotsN(self, num_cls, aname_cate=ANMAE_CATEGORY, fltr=None):
        chotsN = [np.zeros(shape=(0, num_cls))]
        for item in (self if fltr is None else filter(fltr, self)):
            chotsN.append(OneHotCategory.convert(getattr(item, aname_cate))._chotN[None])
        chotsN = np.concatenate(chotsN, axis=0)
        return chotsN


def xypNs_concat(xypNs: List[np.ndarray], num_pnt_default: int = 4) -> np.ndarray:
    if len(xypNs) == 0:
        return np.zeros(shape=(0, num_pnt_default, 2))
    num_pnt = max([len(xyp) for xyp in xypNs])
    for i, xyp in enumerate(xypNs):
        xypNs[i] = np.concatenate([xyp, np.repeat(xyp[:1], axis=0, repeats=num_pnt - len(xyp))], axis=0)
    xypsN = np.stack(xypNs, axis=0)
    return xypsN


def xypsNs_concat(xypsNs: List[np.ndarray], num_pnt_default: int = 4) -> np.ndarray:
    if len(xypsNs) == 0:
        return np.zeros(shape=(0, num_pnt_default, 2))
    num_pnt = max([xyp.shape[-2] for xyp in xypsNs])
    for i, xyp in enumerate(xypsNs):
        detla = num_pnt - xyp.shape[-2]
        if detla > 0:
            xypsNs[i] = np.concatenate([xyp, np.repeat(xyp[..., :1, :], axis=-2, repeats=detla)], axis=-2)
    xypsN = np.concatenate(xypsNs, axis=0)
    return xypsN


class BorderExportable(CategoryExportable):
    ANMAE_BORDER = 'border'

    def export_measuresN(self):
        return np.array([item.measure for item in self])

    def orderby_measure_(self, ascend=True):
        measures = self.export_measuresN()
        measures = measures if ascend else -measures
        order = np.argsort(measures)
        buffer = self[:]
        for i, ind in enumerate(order):
            self[i] = buffer[ind]
        return self

    def as_border_type(self, border_type, aname_bdr=ANMAE_BORDER, fltr=None):
        for item in (self if fltr is None else filter(fltr, self)):
            border = border_type.convert(getattr(item, aname_bdr))
            setattr(item, aname_bdr, border)
        return self

    def _export_bordersN(self, border_type, aname_bdr=ANMAE_BORDER, fltr=None):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        for item in (self if fltr is None else filter(fltr, self)):
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
        bordersN = np.concatenate(bordersN, axis=0)
        return bordersN

    def _export_bordersN_cindsN(self, border_type, aname_bdr=ANMAE_BORDER,
                                aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        cindsN = []
        for item in (self if fltr is None else filter(fltr, self)):
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            cindsN.append(IndexCategory.convert(getattr(item, aname_cate))._cindN)
        bordersN = np.concatenate(bordersN, axis=0)
        cindsN = np.array(cindsN).astype(np.int32)
        return bordersN, cindsN

    def _export_bordersN_chotsN(self, border_type, num_cls, aname_bdr=ANMAE_BORDER,
                                aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        chotsN = [np.zeros(shape=(0, num_cls))]
        for item in (self if fltr is None else filter(fltr, self)):
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            chotsN.append(OneHotCategory.convert(getattr(item, aname_cate))._chotN[None])
        bordersN = np.concatenate(bordersN, axis=0)
        chotsN = np.concatenate(chotsN, axis=0)
        return bordersN, chotsN

    def _export_bordersN_cindsN_confsN(self, border_type, aname_bdr=ANMAE_BORDER,
                                       aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        cindsN = []
        confsN = []
        for item in (self if fltr is None else filter(fltr, self)):
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            cate = IndexCategory.convert(getattr(item, aname_cate))
            cindsN.append(cate._cindN)
            confsN.append(cate.confN)
        bordersN = np.concatenate(bordersN, axis=0)
        cindsN = np.array(cindsN)
        confsN = np.array(confsN)
        return bordersN, cindsN, confsN

    def export_xyxysN(self, aname_bdr=ANMAE_BORDER, fltr=None):
        return self._export_bordersN(XYXYBorder, aname_bdr, fltr)

    def export_xywhsN(self, aname_bdr=ANMAE_BORDER, fltr=None):
        return self._export_bordersN(XYWHBorder, aname_bdr, fltr)

    def export_xywhasN(self, aname_bdr=ANMAE_BORDER, fltr=None):
        return self._export_bordersN(XYWHABorder, aname_bdr, fltr)

    def export_xypNs(self, aname_bdr=ANMAE_BORDER, fltr=None):
        xyps = []
        for item in (self if fltr is None else filter(fltr, self)):
            border = XYPBorder.convert(getattr(item, aname_bdr))
            xyps.append(border._xypN)
        return xyps

    def export_xypsN(self, aname_bdr=ANMAE_BORDER, fltr=None):
        xyps = self.export_xypNs(aname_bdr=aname_bdr, fltr=fltr)
        return xypNs_concat(xyps, num_pnt_default=4)

    def export_xyxysN_cindsN(self, aname_bdr=ANMAE_BORDER,
                             aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        return self._export_bordersN_cindsN(XYXYBorder, aname_bdr, aname_cate, fltr)

    def export_xywhsN_cindsN(self, aname_bdr=ANMAE_BORDER,
                             aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        return self._export_bordersN_cindsN(XYWHBorder, aname_bdr, aname_cate, fltr)

    def export_xyxysN_chotsN(self, num_cls, aname_bdr=ANMAE_BORDER,
                             aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        return self._export_bordersN_chotsN(XYXYBorder, num_cls, aname_bdr, aname_cate, fltr)

    def export_xywhsN_chotsN(self, num_cls, aname_bdr=ANMAE_BORDER,
                             aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        return self._export_bordersN_chotsN(XYWHBorder, num_cls, aname_bdr, aname_cate, fltr)

    def export_xywhasN_cindsN(self, aname_bdr=ANMAE_BORDER,
                              aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None):
        return self._export_bordersN_cindsN(XYWHABorder, aname_bdr, aname_cate, fltr)

    def export_border_masksN_enc(self, img_size: tuple, num_cls: int, aname_bdr=ANMAE_BORDER,
                                 aname_cate=CategoryExportable.ANMAE_CATEGORY, fltr=None) -> np.ndarray:
        maskN = np.full(shape=(img_size[1], img_size[0]), fill_value=num_cls, dtype=np.int32)
        for item in (self if fltr is None else filter(fltr, self)):
            cind = IndexCategory.convert(getattr(item, aname_cate))._cindN
            border = getattr(item, aname_bdr)
            if isinstance(border, XYXYBorder) or isinstance(border, XYWHBorder):
                xyxy = XYXYBorder.convert(border)._xyxyN.astype(np.int32)
                maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = cind
            else:
                rgn = RefValRegion.convert(border)
                xyxy = rgn._xyxyN.astype(np.int32)
                patch = maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                filler = np.full_like(patch, fill_value=cind, dtype=np.int32)
                maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = np.where(np.array(rgn.maskNb_ref), filler, patch)
        return maskN

    def export_border_masksNb(self, img_size: tuple, aname_bdr=ANMAE_BORDER, fltr=None) -> np.ndarray:
        maskN = np.full(shape=(img_size[1], img_size[0]), fill_value=False, dtype=bool)
        for item in (self if fltr is None else filter(fltr, self)):
            border = getattr(item, aname_bdr)
            if isinstance(border, XYXYBorder) or isinstance(border, XYWHBorder):
                xyxy = XYXYBorder.convert(border)._xyxyN.astype(np.int32)
                maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = True
            else:
                rgn = RefValRegion.convert(border)
                xyxy = rgn._xyxyN.astype(np.int32)
                patch = maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                filler = np.full_like(patch, fill_value=True, dtype=bool)
                maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = np.where(np.array(rgn.maskNb_ref), filler, patch)
        return maskN


class RegionExportable(CategoryExportable):
    ANMAE_REGION = 'rgn'

    def as_rgn_type(self, rgn_type, aname_rgn=ANMAE_REGION, fltr=None):
        for item in (self if fltr is None else filter(fltr, self)):
            rgn = rgn_type.convert(getattr(item, aname_rgn))
            setattr(item, aname_rgn, rgn)
        return self

    def export_masksN_stk(self, img_size: tuple, aname_rgn=ANMAE_REGION, fltr=None) -> np.ndarray:
        masksN = [np.zeros(shape=(0, img_size[1], img_size[0]), dtype=bool)]
        for item in (self if fltr is None else filter(fltr, self)):
            rgn = getattr(item, aname_rgn)
            masksN.append(rgn.maskNb)
        masksN = np.concatenate(masksN, axis=0)
        return masksN

    def export_masksN_abs(self, img_size: tuple, num_cls: int, aname_cate=CategoryExportable.ANMAE_CATEGORY,
                          aname_rgn=ANMAE_REGION, append_bkgd: bool = True, fltr=None) -> np.ndarray:
        masksN = np.zeros(shape=(img_size[1], img_size[0], num_cls), dtype=bool)
        for item in (self if fltr is None else filter(fltr, self)):
            cind = IndexCategory.convert(getattr(item, aname_cate))._cindN
            rgn = getattr(item, aname_rgn)
            if isinstance(rgn, RefValRegion):
                xyxy = rgn.xyxyN.astype(np.int32)
                masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], cind] += np.array(rgn.maskNb_ref)
            else:
                masksN[..., cind] += rgn.maskNb

        if append_bkgd:
            maskN_bkgd = np.all(~masksN, keepdims=True, axis=2)
            masksN = np.concatenate([masksN, maskN_bkgd], axis=2)
        return masksN

    def export_masksN_enc(self, img_size: tuple, num_cls: int, aname_cate=CategoryExportable.ANMAE_CATEGORY,
                          aname_rgn=ANMAE_REGION, fltr=None) -> np.ndarray:
        masksN = np.full(shape=(img_size[1], img_size[0]), fill_value=num_cls, dtype=np.int32)
        for item in (self if fltr is None else filter(fltr, self)):
            cind = IndexCategory.convert(getattr(item, aname_cate))._cindN
            rgn = getattr(item, aname_rgn)
            if isinstance(rgn, RefValRegion):
                xyxy = rgn.xyxyN.astype(np.int32)
                if np.prod(rgn.maskNb_ref.size) > 0:
                    maskN_ref = np.array(rgn.maskNb_ref)
                    patch = masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    merge = np.where(maskN_ref, np.full_like(patch, fill_value=cind, dtype=np.int32), patch)
                    masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = merge
            else:
                masksN[rgn.maskNb] = cind
        return masksN


class InstExportable(BorderExportable, RegionExportable):

    def _export_bordersN_cindsN_masksN_enc(self, border_type, img_size: tuple, num_cls: int,
                                           aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                           aname_bdr=BorderExportable.ANMAE_BORDER,
                                           aname_rgn=RegionExportable.ANMAE_REGION, fltr=None) -> np.ndarray:
        cindsN = []
        bordersN = [np.zeros(shape=(0, border_type.WIDTH))]
        masksN = np.full(shape=(img_size[1], img_size[0]), fill_value=num_cls, dtype=np.int32)
        for item in (self if fltr is None else filter(fltr, self)):
            border = border_type.convert(getattr(item, aname_bdr))
            borderN = getattr(border, border.__slots__[0])
            bordersN.append(borderN[None])
            cind = IndexCategory.convert(getattr(item, aname_cate))._cindN
            cindsN.append(cind)
            rgn = getattr(item, aname_rgn)
            if isinstance(rgn, RefValRegion):
                xyxy = rgn.xyxyN.astype(np.int32)

                maskN_ref = np.array(rgn.maskNb_ref)
                patch = masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                merge = np.where(maskN_ref, np.full_like(patch, fill_value=cind, dtype=np.int32), patch)
                masksN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = merge
            else:
                masksN[rgn.maskNb] = cind
        return masksN

    def export_xyxysN_cindsN_masksN_enc(self, img_size: tuple, num_cls: int,
                                        aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                        aname_bdr=BorderExportable.ANMAE_BORDER,
                                        aname_rgn=RegionExportable.ANMAE_REGION, fltr=None):
        return self._export_bordersN_cindsN_masksN_enc(XYXYBorder, img_size, num_cls, aname_cate, aname_bdr, aname_rgn,
                                                       fltr)

    def export_xywhsN_cindsN_masksN_enc(self, img_size: tuple, num_cls: int,
                                        aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                        aname_bdr=BorderExportable.ANMAE_BORDER,
                                        aname_rgn=RegionExportable.ANMAE_REGION, fltr=None):
        return self._export_bordersN_cindsN_masksN_enc(XYWHBorder, img_size, num_cls, aname_cate, aname_bdr, aname_rgn,
                                                       fltr)

    def export_xywhasN_cindsN_masksN_enc(self, img_size: tuple, num_cls: int,
                                         aname_cate=CategoryExportable.ANMAE_CATEGORY,
                                         aname_bdr=BorderExportable.ANMAE_BORDER,
                                         aname_rgn=RegionExportable.ANMAE_REGION, fltr=None):
        return self._export_bordersN_cindsN_masksN_enc(XYWHABorder, img_size, num_cls, aname_cate, aname_bdr, aname_rgn,
                                                       fltr)

    def export_rgn_xyxysN_maskNs_ref(self, aname_rgn=RegionExportable.ANMAE_REGION, fltr=None):
        xyxysN = [np.zeros(shape=(0, 4))]
        maskNs = []
        for item in (self if fltr is None else filter(fltr, self)):
            rgn = RefValRegion.convert(getattr(item, aname_rgn))
            maskNs.append(rgn.maskNb_ref)
            xyxysN.append(rgn._xyxyN[None, :])
        xyxysN = np.concatenate(xyxysN, axis=0)
        return xyxysN, maskNs


# </editor-fold>

# <editor-fold desc='标签列表'>
class HasEmptyList(list):

    def empty(self) -> list:
        return []


class HasFiltList(HasEmptyList):
    def filt_indexs_(self, inds: Sequence[int]):
        buffer = list(self)
        self.clear()
        self.extend(buffer[ind] for ind in inds)
        return self

    def filt_(self, fltr: Optional[Callable] = None):
        for i in range(len(self) - 1, -1, -1):
            if fltr is not None and not fltr(self[i]):
                self.pop(i)
        return self

    def filt(self, fltr: Optional[Callable] = None):
        buffer = self.empty()
        buffer.extend(self if fltr is None else filter(fltr, self))
        return buffer

    def split(self, fltr: Optional[Callable] = None):
        lb_pos, lb_neg = self.empty(), self.empty()
        for item in self:
            if fltr is None or fltr(item):
                lb_pos.append(item)
            else:
                lb_neg.append(item)
        return lb_pos, lb_neg


class HasPermutationList(HasEmptyList):

    def permutation_(self):
        inds = np.random.permutation(len(self))
        buffer = self[:]
        for i, ind in enumerate(inds):
            self[i] = buffer[ind]
        return self

    def permutation(self):
        inds = np.random.permutation(len(self))
        buffer = self.empty()
        for ind in inds:
            buffer.append(self[ind])
        return buffer


class HasFiltMeasureList(HasEmptyList):

    def filt_measure_(self, thres: float = -1):
        for i in range(len(self) - 1, -1, -1):
            item = self[i]
            if thres > 0 and item.measure < thres:
                self.pop(i)
        return self

    def filt_measure(self, thres: float = -1):
        buffer = self.empty()
        for i, item in enumerate(self):
            if thres > 0 and item.measure < thres:
                continue
            buffer.append(item)
        return buffer


class ImageItemsLabel(ImageLabel, HasFiltList, HasPermutationList, Clipable, HasFiltMeasureList, HasXYXYSN):

    @property
    def xyxysN(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4))
        else:
            return np.stack([itm.xyxyN for itm in self], axis=0)

    def __init__(self, *items, img_size: tuple, meta=None, **kwargs):
        list.__init__(self, *items)
        ImageLabel.__init__(self, img_size=img_size, meta=meta, **kwargs)

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        for item in self:
            item.clip_(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self

    @property
    def img_size(self):
        return self.ctx_border.size

    @img_size.setter
    def img_size(self, img_size):
        self.ctx_border.size = img_size
        for item in self:
            item.img_size = img_size

    @property
    def num_xysN(self):
        return 4 + sum([item.num_xysN for item in self])

    def extract_xysN(self):
        xysN = [ImageLabel.extract_xysN(self)]
        for item in self:
            if isinstance(item, PointsExtractable):
                xysN.append(item.extract_xysN())
        xysN = np.concatenate(xysN, axis=0)
        return xysN

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        xysN_ctx, xysN = xysN[:4], xysN[4:]
        ptr = 0
        for item in self:
            if isinstance(item, PointsExtractable):
                num_pnt = item.num_xysN
                item.refrom_xysN(xysN[ptr:ptr + num_pnt], size, **kwargs)
                ptr = ptr + num_pnt
        ImageLabel.refrom_xysN(self, xysN_ctx, size, **kwargs)
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        for item in self:
            if isinstance(item, Movable):
                item.linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        ImageLabel.linear_(self, biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        for item in self:
            if isinstance(item, Movable):
                item.perspective_(homographyN=homographyN, size=size, **kwargs)
        ImageLabel.perspective_(self, homographyN=homographyN, size=size, **kwargs)
        return self

    def empty(self):
        items = self.__new__(self.__class__)
        items.__init__(img_size=tuple(self.img_size), meta=copy.copy(self.meta), **copy.deepcopy(self.kwargs))
        items.ctx_border = copy.deepcopy(self.ctx_border)
        return items

    def __add__(self, other):
        cmb = self.empty()
        cmb.extend(self)
        cmb.extend(other)
        return cmb

    def recover(self):
        xyxyN_init = np.array([0, 0, self.init_size[0], self.init_size[1]])
        xypN_init = xyxyN2xypN(xyxyN_init)
        if self.img_size == self.init_size and np.all(xypN_init == self.ctx_border._xypN):
            return self
        homography = xysN2perspective(xysN_src=self.ctx_border._xypN, xysN_dst=xypN_init)
        for item in self:
            item.perspective_(homographyN=homography, size=self.init_size)
        self.ctx_size = self.init_size
        return self

    def __getitem__(self, item):
        if isinstance(item, Iterable):
            items = self.empty()
            for ind in item:
                items.append(self[ind])
            return items
        elif isinstance(item, slice):
            items = self.empty()
            items += list.__getitem__(self,item)
            return items
        else:
            return list.__getitem__(self,item)


class PointsLabel(ImageItemsLabel, CategoryExportable, Convertable):
    REGISTER_COVERT = Register()

    def __init__(self, *pnts, img_size, meta=None, **kwargs):
        super(PointsLabel, self).__init__(*pnts, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = PointItem.convert(self[i])
            self[i].size = img_size


class BoxesLabel(ImageItemsLabel, BorderExportable, Convertable):
    REGISTER_COVERT = Register()

    def __init__(self, *boxes, img_size, meta=None, **kwargs):
        super(BoxesLabel, self).__init__(*boxes, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = BoxItem.convert(self[i])
            self[i].size = img_size

    @staticmethod
    def _from_bordersN_confsN_cindsN(border_type, bordersN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray,
                                     img_size: tuple, num_cls: int, cind2name=None):
        boxes = BoxesLabel(img_size=img_size)
        for borderN, conf, cind in zip(bordersN, confsN, cindsN):
            category = IndexCategory(cindN=cind, confN=conf, num_cls=num_cls)
            box = BoxItem(category=category, border=border_type(borderN, size=img_size))
            if cind2name is not None:
                box['name'] = cind2name(category._cindN)
            boxes.append(box)
        return boxes

    @staticmethod
    def _from_bordersN_chotsN(border_type, bordersN: np.ndarray, chotsN: np.ndarray,
                              img_size: tuple, cind2name=None):
        boxes = BoxesLabel(img_size=img_size)
        for borderN, chot in zip(bordersN, chotsN):
            category = OneHotCategory(chot)
            box = BoxItem(category=category, border=border_type(borderN, size=img_size))
            if cind2name is not None:
                box['name'] = cind2name(IndexCategory.convert(category)._cindN)
            boxes.append(box)
        return boxes

    @staticmethod
    def _from_bordersN(border_type, bordersN: np.ndarray, img_size: tuple):
        boxes = BoxesLabel(img_size=img_size)
        for borderN in bordersN:
            category = IndexCategory(cindN=0, num_cls=1)
            box = BoxItem(category=category, border=border_type(borderN, size=img_size))
            boxes.append(box)
        return boxes

    @staticmethod
    def _from_bordersT_confsT_cindsT(border_type, bordersT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor,
                                     img_size: tuple, num_cls: int, cind2name=None):
        bordersN = bordersT.detach().cpu().numpy()
        confsN = confsT.detach().cpu().numpy()
        cindsN = cindsT.detach().cpu().numpy()
        return BoxesLabel._from_bordersN_confsN_cindsN(border_type, bordersN, confsN, cindsN, img_size, num_cls,
                                                       cind2name)

    @staticmethod
    def _from_bordersT(border_type, bordersT: torch.Tensor, img_size: tuple):
        bordersN = bordersT.detach().cpu().numpy()
        return BoxesLabel._from_bordersN(border_type, bordersN, img_size)

    @staticmethod
    def _from_bordersT_chotsT(border_type, bordersT: torch.Tensor, chotsT: torch.Tensor,
                              img_size: tuple, cind2name=None):
        bordersN = bordersT.detach().cpu().numpy()
        chotsN = chotsT.detach().cpu().numpy()
        return BoxesLabel._from_bordersN_chotsN(border_type, bordersN, chotsN, img_size, cind2name)

    @staticmethod
    def from_xyxysN_confsN(xyxysN: np.ndarray, confsN: np.ndarray, img_size: tuple, num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYXYBorder, xyxysN, confsN, np.zeros_like(confsN), img_size,
                                                       num_cls, cind2name)

    @staticmethod
    def from_xyxysT_confsT(xyxysT: torch.Tensor, confsT: torch.Tensor, img_size: tuple, num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYXYBorder, xyxysT, confsT, torch.zeros_like(confsT), img_size,
                                                       num_cls, cind2name)

    @staticmethod
    def from_xyxysT(xyxysT: torch.Tensor, img_size: tuple):
        return BoxesLabel._from_bordersT(XYXYBorder, xyxysT, img_size)

    @staticmethod
    def from_xyxysN_confsN_cindsN(xyxysN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYXYBorder, xyxysN, confsN, cindsN, img_size, num_cls, cind2name)

    @staticmethod
    def from_xyxysT_confsT_cindsT(xyxysT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYXYBorder, xyxysT, confsT, cindsT, img_size, num_cls, cind2name)

    @staticmethod
    def from_xyxysT_chotsT(xyxysT: torch.Tensor, chotsT: torch.Tensor, img_size: tuple, cind2name=None):
        return BoxesLabel._from_bordersT_chotsT(XYXYBorder, xyxysT, chotsT, img_size, cind2name)

    @staticmethod
    def from_xyxysN_chotsN(xyxysN: np.ndarray, chotsN: np.ndarray, img_size: tuple, cind2name=None):
        return BoxesLabel._from_bordersN_chotsN(XYXYBorder, xyxysN, chotsN, img_size, cind2name)

    @staticmethod
    def from_xywhsT_chotsT(xywhsT: torch.Tensor, chotsT: torch.Tensor, img_size: tuple, cind2name=None):
        return BoxesLabel._from_bordersT_chotsT(XYWHBorder, xywhsT, chotsT, img_size, cind2name)

    @staticmethod
    def from_xywhsN_confsN_cindsN(xywhsN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYWHBorder, xywhsN, confsN, cindsN, img_size, num_cls, cind2name)

    @staticmethod
    def from_xywhsT_confsT_cindsT(xywhsT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                  num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYWHBorder, xywhsT, confsT, cindsT, img_size, num_cls, cind2name)

    @staticmethod
    def from_xywhasN_confsN_cindsN(xywhasN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray, img_size: tuple,
                                   num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersN_confsN_cindsN(XYWHABorder, xywhasN, confsN, cindsN, img_size, num_cls,
                                                       cind2name)

    @staticmethod
    def from_xywhasT_confsT_cindsT(xywhasT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                   num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYWHABorder, xywhasT, confsT, cindsT, img_size, num_cls,
                                                       cind2name)

    @staticmethod
    def from_xypsT_confsT_cindsT(xypsT: torch.Tensor, confsT: torch.Tensor, cindsT: torch.Tensor, img_size: tuple,
                                 num_cls: int, cind2name=None):
        return BoxesLabel._from_bordersT_confsT_cindsT(XYPBorder, xypsT, confsT, cindsT, img_size, num_cls,
                                                       cind2name)


class SegsLabel(ImageItemsLabel, RegionExportable, BoolMaskExtractableByIterable, Convertable):
    REGISTER_COVERT = Register()

    def __init__(self, *segs, img_size, meta=None, **kwargs):
        super(SegsLabel, self).__init__(*segs, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = SegItem.convert(self[i])
            self[i].size = img_size

    @staticmethod
    def from_masksN(masksN: np.ndarray, num_cls: int, conf_thres: float = None, cind2name=None):
        _, H, W = masksN.shape
        segs = SegsLabel(img_size=(W, H))
        for cind in range(num_cls):
            masksN_cind = masksN[cind:cind + 1]
            conf_thres_i = conf_thres if conf_thres is not None else np.max(masksN_cind) / 2
            category = IndexCategory(cindN=cind, confN=np.max(masksN_cind), num_cls=num_cls)
            rgn = AbsValRegion(maskN_abs=masksN_cind, conf_thres=conf_thres_i)
            seg = SegItem(rgn=rgn, category=category)
            if cind2name is not None:
                seg['name'] = cind2name(cind)
            segs.append(seg)
        return segs

    @staticmethod
    def from_masksT(masksT: torch.Tensor, num_cls: int, conf_thres: float = None, cind2name=None):
        masksN = masksT.detach().cpu().numpy().astype(np.float32)
        return SegsLabel.from_masksN(masksN, num_cls, conf_thres, cind2name)


class InstsLabel(ImageItemsLabel, InstExportable, BoolMaskExtractableByIterable, Convertable):
    REGISTER_COVERT = Register()

    def __init__(self, *insts, img_size, meta=None, **kwargs):
        super(InstsLabel, self).__init__(*insts, img_size=img_size, meta=meta, **kwargs)
        for i in range(len(self)):
            self[i] = InstItem.convert(self[i])
            self[i].size = img_size

    def align(self):
        for inst in self:
            inst.align()
        return self

    def avoid_overlap(self):
        # 从大到小排序
        measures = np.array([inst.measure() for inst in self])
        order = np.argsort(-measures)
        lst_sortd = [self[ind] for ind in order]
        for i in range(len(self)):
            self[i] = lst_sortd[i]

        for inst in self:
            inst.rgn = AbsBoolRegion.convert(inst.rgn)
        if self.num_bool_chan > 0:
            maskN = self.extract_maskNb_enc(index=1)
            self.refrom_maskNb_enc(maskN, index=1)
        return self

    @staticmethod
    def from_boxes_rgns(boxes: BoxesLabel, rgns: list):
        insts = InstsLabel(img_size=boxes.img_size)
        for box, rgn in zip(boxes, rgns):
            inst = InstItem(category=box.category, border=box.border, rgn=rgn, **box)
            insts.append(inst)
        return insts

    @staticmethod
    def from_boxes_masksN_ref(boxes: BoxesLabel, masksN: np.ndarray, conf_thres: float = 0.2,
                              resample=cv2.INTER_CUBIC, cind: int = None):
        img_size = boxes.img_size
        insts = InstsLabel(img_size=img_size)
        for box, maskcN in zip(boxes, masksN):
            cind_i = IndexCategory.convert(box.category)._cindN if cind is None else cind
            maskN = maskcN[cind_i]
            conf_thres_i = conf_thres if conf_thres is not None else np.max(maskN) / 2
            xyxyN = XYXYBorder.convert(box.border)._xyxyN
            xyxyN = np.round(xyxyN).astype(np.int32)
            size = list(xyxyN[2:4] - xyxyN[:2])
            maskN = cv2.resize(maskN, size, interpolation=resample)
            mask = RefValRegion(xyN=xyxyN[:2], maskN_ref=maskN, size=img_size, conf_thres=conf_thres_i)
            inst = InstItem(border=box.border, category=box.category, rgn=mask, **box)
            insts.append(inst)
        return insts

    @staticmethod
    def from_boxes_masksT_ref(boxes: BoxesLabel, masksT: torch.Tensor, conf_thres: float = 0.2,
                              resample=cv2.INTER_CUBIC, cind: int = None):
        masksN = masksT.detach().cpu().numpy()
        return InstsLabel.from_boxes_masksN_ref(boxes, masksN, conf_thres, resample, cind)

    @staticmethod
    def from_boxes_masksN_abs(boxes: BoxesLabel, masksN: np.ndarray, conf_thres: float = None, cind: int = None,
                              only_inner: bool = True):
        img_size = boxes.img_size
        insts = InstsLabel(img_size=img_size)
        for box in copy.deepcopy(boxes):
            cind_i = IndexCategory.convert(box.category)._cindN if cind is None else cind
            maskN_abs = masksN[cind_i]
            ####################测试
            if only_inner:
                border = box.border2 if hasattr(box, 'border_ref') else box.border
                border = XYWHABorder.convert(border)
                border._xywhaN[2:4] = np.ceil(border._xywhaN[2:4])
                maskN_abs = maskN_abs * border.maskNb.astype(np.float32)
            ####################测试
            conf_thres_i = conf_thres if conf_thres is not None else np.max(maskN_abs) / 2
            rgn = AbsValRegion(maskN_abs, conf_thres=conf_thres_i)
            inst = InstItem(border=box.border, rgn=rgn, category=box.category, **box)
            insts.append(inst)
        return insts

    @staticmethod
    def from_boxes_masksT_abs(boxes: BoxesLabel, masksT: torch.Tensor, conf_thres: float = 0.2,
                              cind: int = None, only_inner: bool = True):
        masksN = masksT.detach().cpu().numpy()
        return InstsLabel.from_boxes_masksN_abs(boxes, masksN, conf_thres, cind, only_inner)


# </editor-fold>


# <editor-fold desc='注册json变换'>

REGISTRY_JSON_ENCDEC_BY_INIT(CategoryLabel)


@REGISTER_JSON_ENC.registry(ImageItemsLabel, PointsLabel, BoxesLabel, SegsLabel, InstsLabel)
def _items_label2json_dct(items_label: ImageItemsLabel) -> dict:
    items = [obj2json_dct(item) for item in items_label]
    return {'img_size': items_label.img_size,
            'meta': items_label.img_size,
            'kwargs': obj2json_dct(items_label.kwargs),
            'items': items}


def _json_dct2items_label(json_dct: dict, cls=None) -> np.ndarray:
    img_size = json_dct['img_size']
    meta = json_dct['meta']
    kwargs = json_dct2obj(json_dct['kwargs'])
    items = [json_dct2obj(item) for item in json_dct['items']]
    return cls(items, img_size=img_size, meta=meta, **kwargs)


REGISTER_JSON_DEC[ImageItemsLabel.__name__] = partial(_json_dct2items_label, cls=ImageItemsLabel)
REGISTER_JSON_DEC[PointsLabel.__name__] = partial(_json_dct2items_label, cls=PointsLabel)
REGISTER_JSON_DEC[BoxesLabel.__name__] = partial(_json_dct2items_label, cls=BoxesLabel)
REGISTER_JSON_DEC[SegsLabel.__name__] = partial(_json_dct2items_label, cls=SegsLabel)
REGISTER_JSON_DEC[InstsLabel.__name__] = partial(_json_dct2items_label, cls=InstsLabel)


# </editor-fold>

# <editor-fold desc='标签列表相互转化'>
@BoxesLabel.REGISTER_COVERT.registry(InstsLabel)
def _insts_label2boxes_label(boxes: InstsLabel):
    boxes_list = [BoxItem(border=inst.border, category=inst.category, **inst) for inst in boxes]
    return BoxesLabel(boxes_list, img_size=boxes.img_size, meta=boxes.meta, **boxes.kwargs)


@BoxesLabel.REGISTER_COVERT.registry(ImageItemsLabel)
def _items_label2boxes_label(boxes: ImageItemsLabel):
    return BoxesLabel(boxes, img_size=boxes.img_size, meta=boxes.meta, **boxes.kwargs)


@BoxesLabel.REGISTER_COVERT.registry(ImageLabel)
def _img_label2boxes_label(boxes: ImageLabel):
    return BoxesLabel(img_size=boxes.img_size, meta=boxes.meta, **boxes.kwargs)


@SegsLabel.REGISTER_COVERT.registry(BoxesLabel)
def _boxes_label2segs_label(segs: BoxesLabel):
    segs_new = SegsLabel(img_size=segs.img_size, meta=segs.meta, **segs.kwargs)
    for box in segs:
        rgn = XYPBorder.convert(box.border)
        seg = SegItem(rgn=rgn, category=box.category, **box)
        segs_new.append(seg)
    return segs_new


@InstsLabel.REGISTER_COVERT.registry(BoxesLabel)
def _boxes_label2insts_label(insts: BoxesLabel):
    insts_new = InstsLabel(img_size=insts.img_size, meta=insts.meta, **insts.kwargs)
    for box in insts:
        rgn = XYPBorder.convert(box.border)
        inst = InstItem(border=box.border, rgn=rgn, category=box.category, **box)
        insts_new.append(inst)
    return insts_new


# </editor-fold>
# 恢复到原图标签大小
def labels_linear(labels: Sequence, img_sizes: List[Tuple], scales: np.ndarray,
                  biass: np.ndarray, reverse: bool = False, thres: float = 0) -> Sequence:
    assert len(labels) == len(scales) and len(labels) == len(img_sizes), 'len err'
    if np.all(scales == 1) and np.all(biass == 0):
        return labels
    labels_scaled = []
    if reverse:
        scales, biass = linear_reverse(scales, biass)
    for label, img_size, scale, bias in zip(labels, img_sizes, scales, biass):
        label_scaled = copy.deepcopy(label)
        # if isinstance(labels_scaled, Movable):
        label_scaled.linear_(scaleN=scale, biasN=bias, size=img_size)
        if isinstance(labels_scaled, HasFiltMeasureList):
            label_scaled.filt_measure_(thres=thres)
        labels_scaled.append(label_scaled)
    return labels_scaled
