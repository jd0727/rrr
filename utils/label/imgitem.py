from .element import *


# <editor-fold desc='标签单体'>

# img_size 对应图像大小
# ctx_border 对应图像有内容区域
# img_size_init 初始图像大小
# meta 图像唯一标识


class ImageItem(dict, SettableImageSize, Movable, Measurable, HasXYXYN):
    __slots__ = []

    def __init__(self, *seq, **kwargs):
        dict.__init__(self, *seq, **kwargs)


class PointItem(ImageItem, Convertable, PointsExtractable, Clipable):
    @property
    def xyxyN(self) -> np.ndarray:
        return self.pnts.xyxyN

    REGISTER_COVERT = Register()

    def measure(self):
        return self.pnts.measure

    __slots__ = ('pnts', 'category')

    def __init__(self, pnts, category, *seq, **kwargs):
        dict.__init__(self, *seq, **kwargs)
        self.pnts = XYSPoint.convert(pnts)
        self.category = category

    @property
    def img_size(self):
        return self.pnts.size

    @property
    def size(self):
        return self.pnts.size

    @img_size.setter
    def img_size(self, img_size):
        self.pnts.size = img_size

    @size.setter
    def size(self, size):
        self.pnts.size = size

    @property
    def num_xysN(self):
        return self.pnts.num_xysN

    def extract_xysN(self):
        return self.pnts._xysN

    def refrom_xysN(self, xysN, size, **kwargs):
        self.pnts.refrom_xysN(xysN, size)
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.pnts.linear_(size, biasN=biasN, scaleN=scaleN)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.pnts.perspective_(homographyN=homographyN)
        return self

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.pnts.clip_(xyxyN_rgn)
        return self

    def __repr__(self):
        return self.category.__repr__() + 'pnt' + str(len(self.pnts._xysN)) + super(PointItem, self).__repr__()

    def __eq__(self, other):
        return isinstance(other, PointItem) and self.category == other.category and np.all(self.pnts == other.pnts)


class BoxItem(ImageItem, Convertable, PointsExtractable, HasArea, Clipable):
    @property
    def xyxyN(self) -> np.ndarray:
        return self.border.xyxyN

    REGISTER_COVERT = Register()

    @property
    def measure(self):
        return self.border.measure

    @property
    def area(self):
        return self.border.area

    @property
    def img_size(self):
        return self.border.size

    @img_size.setter
    def img_size(self, img_size):
        self.border.size = img_size

    @property
    def size(self):
        return self.border.size

    @size.setter
    def size(self, size):
        self.border.size = size

    __slots__ = ('border', 'category',)

    def __init__(self, border, category, *seq, **kwargs):
        super().__init__(*seq, **kwargs)
        # print(border)
        self.border = border
        self.category = category

    def __eq__(self, other):
        return isinstance(other, BoxItem) and self.category == other.category and self.border == other.border

    @property
    def num_xysN(self):
        return self.border.num_xysN

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border.clip_(xyxyN_rgn, **kwargs)
        return self

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.border.linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.border.perspective_(homographyN=homographyN, size=size, **kwargs)
        return self

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + super(BoxItem, self).__repr__()

    def extract_xysN(self):
        return self.border.extract_xysN()

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        self.border.refrom_xysN(xysN, size, **kwargs)
        return self


class DualBoxItem(BoxItem):
    __slots__ = ('border', 'border2', 'category',)

    def __init__(self, border, border2, category, *seq, **kwargs):
        super().__init__(border, category, *seq, **kwargs)
        self.border2 = border2

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + \
               '(' + self.border2.__repr__() + ')' + super(BoxItem, self).__repr__()

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        super(DualBoxItem, self).linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        self.border2.linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        super(DualBoxItem, self).perspective_(homographyN=homographyN, size=size, **kwargs)
        self.border2.perspective_(homographyN=homographyN, size=size, **kwargs)
        return self

    def extract_xysN(self):
        xyp = self.border.extract_xysN()
        xyp2 = self.border2.extract_xysN()
        return np.concatenate([xyp, xyp2], axis=0)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        interval = self.border.num_xysN
        self.border.refrom_xysN(xysN[:interval], size, **kwargs)
        self.border2.refrom_xysN(xysN[interval:], size, **kwargs)
        return self

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.border.clip_(xyxyN_rgn, **kwargs)
        self.border2.clip_(xyxyN_rgn, **kwargs)
        return self

    @property
    def num_xysN(self):
        return self.border.num_xysN + self.border2.num_xysN

    def __eq__(self, other):
        return isinstance(other, DualBoxItem) and self.category == other.category \
               and self.border == other.border and self.border2 == other.border2


class SegItem(ImageItem, Convertable, HasArea, Clipable, BoolMaskExtractable):
    @property
    def xyxyN(self) -> np.ndarray:
        return self.rgn.xyxyN

    @property
    def num_bool_chan(self) -> int:
        return self.rgn.num_bool_chan

    def extract_maskNb(self) -> np.ndarray:
        return self.rgn.extract_maskNb()

    def refrom_maskNb(self, maskNb, **kwargs):
        self.rgn.refrom_maskNb(maskNb)

    REGISTER_COVERT = Register()

    @property
    def img_size(self):
        return self.rgn.size

    @img_size.setter
    def img_size(self, img_size):
        self.rgn.size = img_size

    @property
    def size(self):
        return self.rgn.size

    @size.setter
    def size(self, size):
        self.rgn.size = size

    __slots__ = ('rgn', 'category',)

    def __init__(self, category, rgn, *seq, **kwargs):
        super(SegItem, self).__init__(*seq, **kwargs)
        self.category = category
        self.rgn = rgn

    def __repr__(self):
        return self.category.__repr__() + self.rgn.__repr__()

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.rgn.linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.rgn.perspective_(homographyN=homographyN, size=size, **kwargs)
        return self

    @property
    def measure(self):
        return self.rgn.measure

    @property
    def area(self):
        return self.rgn.area

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.rgn.clip_(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self

    def __eq__(self, other):
        return isinstance(other, SegItem) and self.category == other.category and self.rgn == other.rgn


class InstItem(ImageItem, Convertable, PointsExtractable, BoolMaskExtractable, HasArea):
    REGISTER_COVERT = Register()
    __slots__ = ('border', 'rgn', 'category',)

    def __init__(self, border, rgn, category, *seq, **kwargs):
        super(InstItem, self).__init__(*seq, **kwargs)
        self.category = category
        self.rgn = rgn
        self.border = border

    @property
    def num_xysN(self) -> int:
        if isinstance(self.rgn, PointsExtractable):
            return self.rgn.num_xysN
        else:
            return 0

    def extract_xysN(self) -> np.ndarray:
        if isinstance(self.rgn, PointsExtractable):
            return self.rgn.extract_xysN()
        else:
            return np.array([])

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        if isinstance(self.rgn, PointsExtractable):
            self.rgn.refrom_xysN(xysN, size)
            self.border = self.border.__class__.convert(self.rgn)

    @property
    def num_bool_chan(self) -> int:
        if isinstance(self.rgn, BoolMaskExtractable):
            return self.rgn.num_bool_chan
        else:
            return 0

    def extract_maskNb(self) -> np.ndarray:
        if isinstance(self.rgn, BoolMaskExtractable):
            return self.rgn.extract_maskNb()
        else:
            return np.array([])

    def refrom_maskNb(self, maskNb, **kwargs):
        if isinstance(self.rgn, BoolMaskExtractable):
            self.rgn.refrom_maskNb(maskNb)
            self.border = self.border.__class__.convert(self.rgn)

    @property
    def xyxyN(self) -> np.ndarray:
        return self.border.xyxyN

    @property
    def img_size(self):
        return self.rgn.size

    @img_size.setter
    def img_size(self, img_size):
        self.rgn.size = img_size
        self.border.size = img_size

    @property
    def size(self):
        return self.rgn.size

    @size.setter
    def size(self, size):
        self.rgn.size = size
        self.border.size = size

    @property
    def measure(self):
        return min(self.border.measure, self.rgn.measure)

    @property
    def area(self):
        return self.rgn.area

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        self.rgn.clip_(xyxyN_rgn=xyxyN_rgn, **kwargs)
        self.border = self.border.__class__.convert(self.rgn)
        return self

    def __repr__(self):
        return self.category.__repr__() + self.border.__repr__() + \
               self.rgn.__repr__() + super(InstItem, self).__repr__()

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.rgn.linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        self.border = self.border.__class__.convert(self.rgn)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.rgn.perspective_(homographyN=homographyN, size=size, **kwargs)
        self.border = self.border.__class__.convert(self.rgn)
        return self

    def __eq__(self, other):
        return isinstance(other, InstItem) and self.category == other.category \
               and self.rgn == other.rgn and self.border == other.border


# </editor-fold>

# <editor-fold desc='注册json变换'>
REGISTRY_JSON_ENCDEC_BY_INIT(PointItem)
REGISTRY_JSON_ENCDEC_BY_INIT(BoxItem)
REGISTRY_JSON_ENCDEC_BY_INIT(DualBoxItem)
REGISTRY_JSON_ENCDEC_BY_INIT(SegItem)
REGISTRY_JSON_ENCDEC_BY_INIT(InstItem)


# </editor-fold>

# <editor-fold desc='标签单体相互转化'>


@BoxItem.REGISTER_COVERT.registry(DualBoxItem)
def _dualbox_item2box_item(box: DualBoxItem, border_type=XYXYBorder):
    border = border_type.convert(box.border)
    return BoxItem(border=border, category=box.category, **box)


@BoxItem.REGISTER_COVERT.registry(InstItem)
def _inst_item2box_item(box: InstItem):
    return BoxItem(border=box.border, category=box.category, **box)


@BoxItem.REGISTER_COVERT.registry(Sequence)
def _seq2box_item(box: Sequence, border_type=XYXYBorder):
    border = border_type.convert(box)
    return BoxItem(border=border, category=IndexCategory(cindN=0, num_cls=1))


@DualBoxItem.REGISTER_COVERT.registry(BoxItem)
def _box_item2dualbox_item(box: BoxItem):
    border_ref = copy.deepcopy(box.border)
    return DualBoxItem(border=box.border, category=box.category, border2=border_ref, **box)


@DualBoxItem.REGISTER_COVERT.registry(Sequence)
def _seq2dualbox_item(box: Sequence, border_type=XYXYBorder):
    border = border_type.convert(box)
    return DualBoxItem(border=border, category=IndexCategory(cindN=0, num_cls=1), border2=copy.deepcopy(border))


@SegItem.REGISTER_COVERT.registry(BoxItem)
def _box_item2seg_item(seg: BoxItem):
    rgn = XYPBorder.convert(seg.border)
    return SegItem(rgn=rgn, category=seg.category, **seg)


@SegItem.REGISTER_COVERT.registry(Image.Image, np.ndarray)
def _image2seg_item(seg: Union[Image.Image, np.ndarray]):
    rgn = AbsBoolRegion(seg)
    return SegItem(rgn=rgn, category=IndexCategory(cindN=0, num_cls=1))


@InstItem.REGISTER_COVERT.registry(BoxItem)
def _box_item2inst_item(inst: BoxItem):
    rgn = RefValRegion.convert(inst.border)
    return InstItem(border=inst.border, category=inst.category, rgn=rgn, **inst)

# </editor-fold>
