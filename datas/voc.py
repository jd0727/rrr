from datas.base import *
from datas.base.inplabel import partition_set, merge_set
from datas.base.iotools import _analysis_sizes
from utils import *
import xml.etree.ElementTree as ET
import ast

# <editor-fold desc='ds xml标签转化'>

REGISTER_XMLRD = Register()


def xmlrd_items(node, **kwargs):
    items = []
    for i, sub_node in enumerate(list(node)):
        if node.tag == 'annotation' and sub_node.tag == 'annotation':
            continue
        if sub_node.tag in REGISTER_XMLRD.keys():
            reader = REGISTER_XMLRD[sub_node.tag]
            item = reader(sub_node, **kwargs)
            items.append(item)
    return items


REGISTER_XMLWT = Register()


def xmlwt_item(node, item, **kwargs):
    writer = REGISTER_XMLWT[item.__class__]
    sub_node = writer(node, item, **kwargs)
    return sub_node


def xmlwt_items(node, items, **kwargs):
    for item in items:
        writer = REGISTER_XMLWT[item.__class__]
        writer(node, item, **kwargs)
    return True


TAG_ROOT = 'annotation'
TAG_META = 'filename'
TAG_SIZE = 'size'
TAG_WIDTH = 'width'
TAG_HEIGHT = 'height'
TAG_DEPTH = 'depth'
TAF_CONF = 'conf'
TAG_XYXY = 'bndbox'
TAGS_XYXY_ITEM = ('xmin', 'ymin', 'xmax', 'ymax')

TAG_XYWH = 'bndboxw'
TAGS_XYWH_ITEM = ('cx', 'cy', 'w', 'h')

TAG_XYWHA = 'robndbox'
TAGS_XYWHA_ITEM = ('cx', 'cy', 'w', 'h', 'angle')

TAG_XYP = 'polygon'

TAG_BOXITEM = 'object'
TAG_BOXREFITEM = 'object_ref'
TAGS_BOXITEM = (TAG_BOXITEM, TAG_BOXREFITEM)

TAG_NAME = 'name'
TAG_DIFFICULT = 'difficult'
TAG_TRUNCATED = 'truncated'
KEY_NAME = TAG_NAME
KEY_DIFFICULT = TAG_DIFFICULT
KEY_TRUNCATED = TAG_TRUNCATED
TAGS_IGNORE_BOXITEM = (TAG_NAME, TAG_DIFFICULT, TAG_TRUNCATED, TAG_XYXY, TAG_XYWH, TAG_XYP, TAG_XYWHA)
TAGS_IGNORE_LABEL = (TAG_BOXITEM, TAG_BOXREFITEM, TAG_SIZE, TAG_META)
KEYS_IGNORE_BOXITEM = (KEY_NAME, KEY_DIFFICULT, KEY_TRUNCATED)


def _xmlrd_img_size(node):
    size = node.find(TAG_SIZE)
    img_size = (int(size.find(TAG_WIDTH).text), int(size.find(TAG_HEIGHT).text))
    return img_size


def _xmlwt_img_size(node, img_szie, depth=3):
    size = ET.SubElement(node, TAG_SIZE)
    ET.SubElement(size, TAG_WIDTH).text = str(int(img_szie[0]))
    ET.SubElement(size, TAG_HEIGHT).text = str(int(img_szie[1]))
    ET.SubElement(size, TAG_DEPTH).text = str(int(depth))
    return node


def _xmlrd_meta(node):
    meta = node.find(TAG_META).text.split('.')[0]
    return meta


def _xmlwt_meta(node, meta, img_extend='jpg'):
    tag_meta = ensure_extend(meta, img_extend) if meta is not None else ''
    ET.SubElement(node, TAG_META).text = tag_meta
    return node


def _xmlrd_bool(node, tag, default=False):
    val = node.find(tag)
    val = int(val.text) == 1 if val is not None else default
    return val


def _xmlwt_bool(node, tag, val):
    ET.SubElement(node, tag).text = '1' if val else '0'
    return val


def _xmlrd_dict(node, ignore_tages=None):
    full_dict = {}
    for sub_node in node:
        if ignore_tages is not None and sub_node.tag in ignore_tages:
            continue
        if len(sub_node) == 0:
            try:
                val = ast.literal_eval(sub_node.text)
            except Exception as e:
                val = sub_node.text
            full_dict[sub_node.tag] = val
        else:
            full_dict[sub_node.tag] = _xmlrd_dict(sub_node, ignore_tages=ignore_tages)
    return full_dict


def _xmlwt_dict(node, dct, ignore_keys=None):
    node = ET.Element(node) if isinstance(node, str) else node
    if isinstance(dct, dict):
        for key, val in dct.items():
            if ignore_keys is not None and key in ignore_keys:
                continue
            sub_node = ET.SubElement(node, key) if node.find(key) is None else node.find(key)
            _xmlwt_dict(node=sub_node, dct=val, ignore_keys=ignore_keys)
    else:
        dct = dct.tolist() if isinstance(dct, np.ndarray) and dct.size > 1 else dct
        node.text = str(dct)
    return node


@REGISTER_XMLRD.registry(TAG_XYXY)
def _xmlrd_border_xyxy(node, size, **kwargs):
    xyxy = np.array([float(node.find(item).text) for item in TAGS_XYXY_ITEM])
    border = XYXYBorder(xyxyN=xyxy, size=size)
    return border


@REGISTER_XMLWT.registry(XYXYBorder)
def _xmlwt_border_xyxy(node, border, as_integer=False, **kwargs):
    sub_node = ET.SubElement(node, TAG_XYXY)
    for i, item in enumerate(TAGS_XYXY_ITEM):
        val = border._xyxyN[i]
        if as_integer:
            val = int(val)
        ET.SubElement(sub_node, item).text = str(val)
    return sub_node


@REGISTER_XMLRD.registry(TAG_XYWHA)
def _xmlrd_border_xywha(node, size, **kwargs):
    xywha = np.array([float(node.find(item).text) for item in TAGS_XYWHA_ITEM])
    border = XYWHABorder(xywhaN=xywha, size=size)
    return border


@REGISTER_XMLWT.registry(XYWHABorder)
def _xmlwt_border_xywha(node, border, **kwargs):
    sub_node = ET.SubElement(node, TAG_XYWHA)
    for i, item in enumerate(TAGS_XYWHA_ITEM):
        ET.SubElement(sub_node, item).text = str(border._xywhaN[i])
    return sub_node


@REGISTER_XMLRD.registry(TAG_XYWH)
def _xmlrd_border_xywh(node, size, **kwargs):
    xywh = np.array([float(node.find(item).text) for item in TAGS_XYWH_ITEM])
    border = XYWHBorder(xywhN=xywh, size=size)
    return border


@REGISTER_XMLWT.registry(XYWHBorder)
def _xmlwt_border_xywh(node, border, **kwargs):
    sub_node = ET.SubElement(node, TAG_XYWH)
    for i, item in enumerate(TAGS_XYWH_ITEM):
        ET.SubElement(sub_node, item).text = str(border._xywhN[i])
    return sub_node


@REGISTER_XMLRD.registry(TAG_XYP)
def _xmlrd_border_xyp(node, size, **kwargs):
    vals = ast.literal_eval(node.text)
    xyp = np.array(vals).reshape(-1, 2)
    border = XYPBorder(xypN=xyp, size=size)
    return border


@REGISTER_XMLWT.registry(XYPBorder)
def _xmlwt_border_xyp(node, border):
    xyp = border._xypN.reshape(-1)
    vals = str(list(xyp))
    sub_node = ET.SubElement(node, TAG_XYP)
    sub_node.text = vals
    return sub_node


@REGISTER_XMLRD.registry(TAG_BOXITEM)
def _xmlrd_boxitem(node, size, name2cind=None, num_cls=1, **kwargs):
    name = node.find(TAG_NAME).text
    cind = name2cind(name) if name2cind is not None else 0
    conf = node.find(TAF_CONF)
    conf = float(conf.text) if conf is not None else 1.0
    category = IndexCategory(cindN=cind, num_cls=num_cls, confN=conf)

    attrs = _xmlrd_dict(node, ignore_tages=TAGS_IGNORE_BOXITEM)
    attrs[KEY_DIFFICULT] = _xmlrd_bool(node, tag=TAG_DIFFICULT)
    attrs[KEY_TRUNCATED] = _xmlrd_bool(node, tag=TAG_TRUNCATED)
    attrs[KEY_NAME] = name

    border = xmlrd_items(node, size=size)[0]
    box = BoxItem(border=border, category=category, **attrs)
    return box


@REGISTER_XMLRD.registry(TAG_BOXREFITEM)
def _xmlrd_boxrefitem(node, size, name2cind=None, num_cls=1, **kwargs):
    name = node.find(TAG_NAME).text
    cind = name2cind(name) if name2cind is not None else 0
    category = IndexCategory(cindN=cind, num_cls=num_cls, confN=1)

    attrs = _xmlrd_dict(node, ignore_tages=TAGS_IGNORE_BOXITEM)
    attrs[KEY_DIFFICULT] = _xmlrd_bool(node, tag=TAG_DIFFICULT)
    attrs[KEY_TRUNCATED] = _xmlrd_bool(node, tag=TAG_TRUNCATED)
    attrs[KEY_NAME] = name

    border, border_ref = xmlrd_items(node, size=size)
    box = DualBoxItem(border=border, border2=border_ref, category=category, **attrs)
    return box


@REGISTER_XMLRD.registry(TAG_ROOT)
def _xmlrd_boxes(node, name2cind=None, num_cls=1, ):
    meta = _xmlrd_meta(node)
    img_size = _xmlrd_img_size(node)
    boxes = xmlrd_items(node, size=img_size, name2cind=name2cind, num_cls=num_cls)
    attrs = _xmlrd_dict(node, ignore_tages=TAGS_IGNORE_LABEL)
    boxes = BoxesLabel(boxes, meta=meta, img_size=img_size, **attrs)
    return boxes


@REGISTER_XMLWT.registry(BoxItem, InstItem)
def _xmlwt_boxitem(node, box, **kwargs):
    obj = ET.SubElement(node, TAG_BOXITEM)
    xmlwt_item(obj, box.border, **kwargs)
    ET.SubElement(obj, TAG_NAME).text = box.get(KEY_NAME, 'Unknown')
    _xmlwt_bool(obj, tag=TAG_DIFFICULT, val=box.get(KEY_DIFFICULT, False))
    _xmlwt_bool(obj, tag=TAG_TRUNCATED, val=box.get(KEY_TRUNCATED, False))
    _xmlwt_dict(obj, box, ignore_keys=KEYS_IGNORE_BOXITEM)
    return node


@REGISTER_XMLWT.registry(DualBoxItem)
def _xmlwt_boxrefitem(node, box, **kwargs):
    obj = ET.SubElement(node, TAG_BOXREFITEM)
    xmlwt_item(obj, box.border)
    xmlwt_item(obj, box.border2)
    ET.SubElement(obj, TAG_NAME).text = box.get(KEY_NAME, 'Unknown')
    _xmlwt_bool(obj, tag=TAG_DIFFICULT, val=box.get(KEY_DIFFICULT, False))
    _xmlwt_bool(obj, tag=TAG_TRUNCATED, val=box.get(KEY_TRUNCATED, False))
    _xmlwt_dict(obj, box, ignore_keys=KEYS_IGNORE_BOXITEM)
    return obj


@REGISTER_XMLWT.registry(ImageItemsLabel, BoxesLabel, InstsLabel)
def _xmlwt_boxes(node, label, **kwargs):
    sub_node = ET.SubElement(node, TAG_ROOT)
    _xmlwt_dict(sub_node, label.kwargs)
    _xmlwt_meta(sub_node, label.meta)
    _xmlwt_img_size(sub_node, label.img_size)
    xmlwt_items(sub_node, label, **kwargs)
    return sub_node


# 修饰node标注
def pretty_xml_node(node, indent='\t', newline='\n', level=0):
    if node:
        # 如果element的text没有内容
        if node.text == None or node.text.isspace():
            node.text = newline + indent * (level + 1)
        else:
            node.text = newline + indent * (level + 1) + node.text.strip() + newline + indent * (level + 1)
    # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(node)  # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml_node(subelement, indent, newline, level=level + 1)
    return True


# </editor-fold>

# <editor-fold desc='ds mask标签转化'>
REGISTER_MASKRD = Register()
REGISTER_MASKWT = Register()


def maskwt(items, mask_pth, **kwargs):
    _maskwt = REGISTER_MASKWT[mask_pth.split('.')[-1]]
    return _maskwt(items, mask_pth, **kwargs)


def maskrd(mask_pth, **kwargs):
    _maskrd = REGISTER_MASKRD[mask_pth.split('.')[-1]]
    return _maskrd(mask_pth, **kwargs)


@REGISTER_MASKWT.registry('jpg', 'png', 'JPG')
def _maskwt_pil(items, mask_pth, colors, **kwargs):
    maskN = np.zeros(shape=(items.img_size[1], items.img_size[0], 3))
    for item in items:
        cind = IndexCategory.convert(item.category)._cindN
        maskN[item.rgn.maskNb, :3] = colors[cind]
    maskP = imgN2imgP(maskN)
    maskP.save(mask_pth)
    return True


@REGISTER_MASKRD.registry('jpg', 'png', 'JPG')
def _maskrd_pil(mask_pth, colors, num_cls, **kwargs):
    colors = np.array([colors[i] for i in range(num_cls)])
    maskN = load_img_cv2(mask_pth)
    maskN = np.all(maskN[:, :, None, :] == colors, axis=-1)
    has_cate = np.any(maskN, axis=(0, 1))
    items = SegsLabel(img_size=(maskN.shape[1], maskN.shape[0]))
    for cind in np.where(has_cate)[0]:
        rgn = AbsBoolRegion(maskN[:, :, cind])
        cate = IndexCategory(cindN=cind, num_cls=len(has_cate), confN=1)
        items.append(SegItem(rgn=rgn, category=cate))
    return items


@REGISTER_MASKWT.registry('npy')
def _maskwt_npy(items, mask_pth, num_cls, **kwargs):
    maskN = np.zeros(shape=(items.img_size[1], items.img_size[0], num_cls))
    for item in items:
        cind = IndexCategory.convert(item.category)._cindN
        rgn = AbsValRegion.convert(item.rgn)
        maskN[..., cind] = np.maximum(maskN[..., cind], rgn.maskN)
    np.save(mask_pth, maskN)
    return True


@REGISTER_MASKRD.registry('npy')
def _maskrd_npy(mask_pth, **kwargs):
    maskN = np.load(mask_pth)
    has_cate = np.any(maskN, axis=(0, 1))
    items = SegsLabel(img_size=(maskN.shape[1], maskN.shape[0]))
    for cind in np.where(has_cate)[0]:
        rgn = AbsBoolRegion(maskN[:, :, cind])
        cate = IndexCategory(cindN=cind, num_cls=len(has_cate), confN=1)
        items.append(SegItem(rgn=rgn, category=cate))
    return items


@REGISTER_MASKWT.registry('pkl')
def _maskwt_pkl(items, mask_pth, **kwargs):
    load_pkl(mask_pth, items)
    return True


@REGISTER_MASKRD.registry('pkl')
def _maskrd_pkl(mask_pth, **kwargs):
    return load_pkl(mask_pth)


# </editor-fold>

# <editor-fold desc='ds inst标签转化'>
REGISTER_INSTRD = Register()
REGISTER_INSTWT = Register()


def instwt(items, inst_pth, **kwargs):
    _instwt = REGISTER_INSTWT[inst_pth.split('.')[-1]]
    return _instwt(items, inst_pth, **kwargs)


def instrd(inst_pth, boxes, **kwargs):
    _instrd = REGISTER_INSTRD[inst_pth.split('.')[-1]]
    return _instrd(inst_pth, boxes, **kwargs)


@REGISTER_INSTWT.registry('jpg', 'png', 'JPG')
def _instwt_pil(items, inst_pth, colors, **kwargs):
    maskN = np.zeros(shape=(items.img_size[1], items.img_size[0], 3))
    for i, item in enumerate(items):
        maskN[item.rgn.maskNb, :3] = colors[i]
    maskP = imgN2imgP(maskN)
    maskP.save(inst_pth)
    return True


@REGISTER_INSTRD.registry('jpg', 'png', 'JPG')
def _instrd_pil(inst_pth, boxes, colors, rgn_type=AbsBoolRegion, **kwargs):
    maskN = np.array(Image.open(inst_pth).convert("RGB"))
    size = (maskN.shape[1], maskN.shape[0])
    insts = InstsLabel(img_size=size)
    for i, box in enumerate(boxes):
        if rgn_type == RefValRegion:
            xyxy = XYXYBorder.convert(box.border)._xyxyN.astype(np.int32)
            patchN = maskN[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            patchNb = np.all(patchN == colors[i], axis=2)
            rgn = RefValRegion(xyN=xyxy[:2], maskN_ref=patchNb.astype(np.float32), size=size)
        elif rgn_type == AbsBoolRegion or rgn_type is None:
            rgn = AbsBoolRegion(maskNb_abs=np.all(maskN == colors[i], axis=2))
        else:
            raise Exception('err rgn type')
        inst = InstItem(border=box.border, category=box.category, rgn=rgn, **box)
        insts.append(inst)
    return insts


@REGISTER_INSTWT.registry('npy')
def _instwt_npy(items, inst_pth, **kwargs):
    maskN = [np.zeros(shape=(0, items.img_size[1], items.img_size[0]))]
    for item in items:
        rgn = AbsValRegion.convert(item.rgn)
        maskN.append(rgn.maskN)
    maskN = np.concatenate(maskN, axis=0)
    np.save(inst_pth, maskN)
    return True


@REGISTER_INSTRD.registry('npy')
def _instrd_npy(inst_pth, boxes, **kwargs):
    maskN = np.array(Image.open(inst_pth).convert("RGB"))
    size = (maskN.shape[1], maskN.shape[0])
    insts = InstsLabel(img_size=size)
    for i, box in enumerate(boxes):
        rgn = AbsValRegion(maskN[i])
        inst = InstItem(border=box.border, category=box.category, rgn=rgn, **box)
        insts.append(inst)
    return insts


@REGISTER_INSTWT.registry('pkl')
def _maskwt_pkl(items, inst_pth, **kwargs):
    rgns = [item.rgn for item in items]
    save_pkl(inst_pth, rgns)
    return True


@REGISTER_INSTRD.registry('pkl')
def _maskrd_pkl(inst_pth, boxes, **kwargs):
    rgns = load_pkl(inst_pth)
    insts = InstsLabel(img_size=boxes.img_size)
    for rgn, box in zip(rgns, boxes):
        inst = InstItem(border=box.border, category=box.category, rgn=rgn, **box)
        insts.append(inst)
    return insts


# </editor-fold>

# <editor-fold desc='写入'>

class VOCLabelWriter(LabelWriter):
    def __init__(self, set_pth: Optional[str] = None):
        self.set_pth = set_pth

    def save_all(self, metas: list):
        if self.set_pth is not None:
            save_txt(self.set_pth, metas)


class VOCDetectionWriter(VOCLabelWriter):
    def __init__(self, anno_dir: str, anno_extend: str = 'xml', set_pth: Optional[str] = None):
        VOCLabelWriter.__init__(self, set_pth)
        self.anno_dir = anno_dir
        self.anno_extend = anno_extend

    @staticmethod
    def save_anno(anno_pth, boxes, as_integer: bool = False):
        ensure_file_dir(anno_pth)
        root = xmlwt_item(ET.Element(''), boxes, as_integer=as_integer)
        pretty_xml_node(root, indent='\t', newline='\n', level=0)
        root = ET.ElementTree(root)
        root.write(anno_pth, encoding='utf-8')
        return root

    def save_label(self, label) -> object:
        anno_pth = os.path.join(self.anno_dir, ensure_extend(label.meta, self.anno_extend, overwrite=True))
        VOCDetectionWriter.save_anno(anno_pth, label, as_integer=False)
        return label.meta


class VOCSegmentationWriter(VOCLabelWriter):
    def __init__(self, mask_dir: str, mask_extend: str = 'png', colors=None, set_pth: Optional[str] = None):
        VOCLabelWriter.__init__(self, set_pth)
        self.mask_dir = mask_dir
        self.mask_extend = mask_extend
        self.colors = colors

    @staticmethod
    def save_mask(mask_pth, segs, colors):
        ensure_file_dir(mask_pth)
        assert isinstance(segs, SegsLabel) or isinstance(segs, InstsLabel), \
            'fmt err ' + segs.__class__.__name__
        maskwt(mask_pth=mask_pth, items=segs, colors=colors)
        return True

    def save_label(self, label) -> object:
        mask_pth = os.path.join(self.mask_dir, ensure_extend(label.meta, self.mask_extend, overwrite=True))
        maskwt(mask_pth=mask_pth, items=label, colors=self.colors)
        return label.meta


class VOCInstanceWriter(VOCLabelWriter):
    def __init__(self, anno_dir: str, inst_dir: str, anno_extend: str = 'xml', inst_extend: str = 'png', colors=None,
                 set_pth: Optional[str] = None):
        VOCLabelWriter.__init__(self, set_pth)
        self.inst_dir = inst_dir
        self.inst_extend = inst_extend
        self.anno_dir = anno_dir
        self.anno_extend = anno_extend
        self.colors = colors

    @staticmethod
    def save_inst(inst_pth, colors, insts):
        ensure_file_dir(inst_pth)
        instwt(insts, inst_pth, colors=colors)
        return True

    @staticmethod
    def save_anno_inst(anno_pth, inst_pth, colors, insts):
        assert isinstance(insts, InstsLabel), 'fmt err ' + insts.__class__.__name__
        VOCDetectionWriter.save_anno(anno_pth, insts)
        VOCInstanceWriter.save_inst(inst_pth, colors=colors, insts=insts)
        return True

    def save_label(self, label) -> object:
        anno_pth = os.path.join(self.anno_dir, ensure_extend(label.meta, self.anno_extend, overwrite=True))
        inst_pth = os.path.join(self.inst_dir, ensure_extend(label.meta, self.inst_extend, overwrite=True))
        VOCInstanceWriter.save_anno_inst(anno_pth, inst_pth, colors=self.colors, insts=label)
        return label.meta


# </editor-fold>


# <editor-fold desc='检测'>

def _expand_pths(root: str, folder: str, metas: Sequence[str], extend: str = 'xml'):
    return [os.path.join(root, folder, meta + '.' + extend) for meta in metas]


def _load_metas(root: str, set_name: str, set_folder: str = 'ImageSets/Main', img_folder: str = 'JPEGImages',
                set_extend: str = 'txt'):
    if set_name is None or (set_name == 'all' and not os.path.exists(
            os.path.join(root, set_folder, ensure_extend(set_name, set_extend)))):
        img_dir = os.path.join(root, img_folder)
        metas = [os.path.splitext(img_name)[0] for img_name in os.listdir(img_dir)]
        return metas
    set_pth = os.path.join(root, set_folder, ensure_extend(set_name, set_extend))
    metas = load_txt(set_pth)
    metas = [m for m in metas if len(m) > 0]
    return metas


class VOCDataset(MDataset):
    IMG_EXTEND = 'jpg'
    SET_EXTEND = 'txt'

    def __init__(self, root: str, set_name: str, set_folder: str = 'ImageSets/Main',
                 img_folder: str = 'JPEGImages', img_extend: str = IMG_EXTEND, num_oversamp: int = 1,
                 data_mode=DATA_MODE.FULL):
        set_name = set_name if isinstance(set_name, str) else 'all'
        MDataset.__init__(self, root=root, set_name=set_name, num_oversamp=num_oversamp, data_mode=data_mode)
        # 加载标签
        self.set_folder = set_folder
        self.img_extend = img_extend
        self.img_folder = img_folder
        self._metas = _load_metas(root, set_name, set_folder=set_folder, img_folder=img_folder)

    @property
    def folders(self):
        return [self._img_folder, self._set_folder]

    def ensure_folders(self):
        ensure_folders(self.root, self.folders)
        return self

    @property
    def metas(self):
        return self._metas

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def set_folder(self):
        return self._set_folder

    @property
    def set_dir(self):
        return os.path.join(self._root, self._set_folder)

    @set_folder.setter
    def set_folder(self, set_folder):
        self._set_folder = set_folder

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def img_dir(self):
        return os.path.join(self.root, self._img_folder)

    @property
    def img_pths(self):
        return _expand_pths(self.root, folder=self.img_folder, metas=self.metas, extend=self.img_extend)

    @img_folder.setter
    def img_folder(self, img_folder):
        self._img_folder = img_folder

    @property
    def imgs(self):
        imgs = [load_img_pil(img_pth) for img_pth in self.img_pths]
        return imgs

    @staticmethod
    def collect_img_sizes(img_dir: str, img_extend: str = 'jpg', metas: Optional[Sequence[str]] = None):
        img_sizes = []
        img_names = listdir_extend(img_dir, extends=img_extend) if metas is None else \
            [ensure_extend(meta, img_extend) for meta in metas]
        for img_name in img_names:
            img_pth = os.path.join(img_dir, img_name)
            img_sizes.append(load_img_size(img_pth))
        return np.array(img_sizes)

    @property
    def img_sizes(self):
        return VOCDataset.collect_img_sizes(img_dir=self.img_dir, img_extend=self.img_extend, metas=self._metas)

    def partition_set_(self, split_dict):
        partition_set(set_dir=self.set_dir, metas=self._metas, split_dict=split_dict)
        return self

    def merge_set_(self, set_names, new_name):
        merge_set(set_dir=self.set_dir, set_names=set_names, new_name=new_name)
        return self

    def __repr__(self):
        msg = dsmsgfmtr_create(
            root=self.root, folders=self.folders, prefix=self.__class__.__name__, set_name=self.set_name) + '\n'
        img_size_aver = tuple(np.mean(self.img_sizes, axis=0).astype(np.int32))
        msg += 'Length %d ' % len(self) + 'ImgSize ' + str(img_size_aver)
        return msg

    def _meta2img(self, meta: str):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
        img = load_img_pil(img_pth)
        return img

    def _index2data(self, index: int):
        return self._meta2data(self.metas[index])

    def _index2img(self, index: int):
        return self._meta2img(self.metas[index])

    def _index2label(self, index: int):
        return self._meta2label(self.metas[index])

    def __len__(self):
        return len(self._metas)

    @property
    def labels(self):
        return [self._meta2label(meta) for meta in self._metas]


class VOCDetectionDataset(MNameMapper, VOCDataset, ):
    ANNO_EXTEND = 'xml'
    IMG_EXTEND = VOCDataset.IMG_EXTEND
    SET_EXTEND = VOCDataset.SET_EXTEND

    def __init__(self, root, set_name: str, cls_names=None, set_folder: str = 'ImageSets/Main',
                 img_folder: str = 'JPEGImages',
                 anno_folder: str = 'Annotations', anno_extend: str = ANNO_EXTEND, img_extend: str = IMG_EXTEND,
                 border_type=None, num_oversamp: int = 1, data_mode=DATA_MODE.FULL, **kwargs):
        if cls_names is None:
            names = VOCDetectionDataset.collect_names(anno_dir=os.path.join(root, anno_folder))
            cls_names = sorted(Counter(names).keys())
        MNameMapper.__init__(self, cls_names)

        VOCDataset.__init__(self, root=root, set_name=set_name, set_folder=set_folder,
                            img_folder=img_folder, img_extend=img_extend, num_oversamp=num_oversamp,
                            data_mode=data_mode)
        self.border_type = border_type
        self.anno_extend = anno_extend
        self.anno_folder = anno_folder

    @property
    def anno_folder(self):
        return self._anno_folder

    @property
    def anno_dir(self):
        return os.path.join(self.root, self._anno_folder)

    @anno_folder.setter
    def anno_folder(self, anno_folder):
        self._anno_folder = anno_folder

    @property
    def anno_pths(self):
        return _expand_pths(self.root, folder=self.anno_folder, metas=self.metas, extend=self.anno_extend)

    # <editor-fold desc='VOC工具'>
    # 获取所有图片大小
    @staticmethod
    def collect_img_sizes(anno_dir: str, anno_extend: str = 'xml', metas: Optional[Sequence[str]] = None):
        img_sizes = []
        anno_names = listdir_extend(anno_dir, extends=anno_extend) if metas is None else \
            [ensure_extend(meta, anno_extend) for meta in metas]
        for anno_name in anno_names:
            anno_pth = os.path.join(anno_dir, anno_name)
            root = ET.parse(anno_pth).getroot()
            img_size = _xmlrd_img_size(root)
            img_sizes.append(img_size)
        return np.array(img_sizes)

    # 获取所有标注类名称
    @staticmethod
    def collect_names(anno_dir: str, anno_extend: str = 'xml', metas: Optional[Sequence[str]] = None):
        names = []
        anno_names = listdir_extend(anno_dir, extends=anno_extend) if metas is None else \
            [ensure_extend(meta, anno_extend) for meta in metas]
        for anno_name in anno_names:
            anno_pth = os.path.join(anno_dir, anno_name)
            root = ET.parse(anno_pth).getroot()
            for i, obj in enumerate(root):
                if obj.tag not in TAGS_BOXITEM:
                    continue
                names.append(obj.find(TAG_NAME).text)
        return np.array(names)

    # 获取所有图片大小
    @staticmethod
    def collect_sizes(anno_dir: str, anno_extend: str = 'xml', metas: Optional[Sequence[str]] = None):
        sizes = []
        anno_names = listdir_extend(anno_dir, extends=anno_extend) if metas is None else \
            [ensure_extend(meta, anno_extend) for meta in metas]
        for anno_name in anno_names:
            anno_pth = os.path.join(anno_dir, anno_name)
            root = ET.parse(anno_pth).getroot()
            boxes_list = xmlrd_items(root, size=(0, 0), name2cind=None, num_cls=1)
            for box in boxes_list:
                xyxy = XYXYBorder.convert(box.border)._xyxyN
                sizes.append(xyxy[2:4] - xyxy[:2])
        return np.array(sizes)

    # 解析xml标注
    @staticmethod
    def load_anno(anno_pth, name2cind=None, num_cls=1, img_size=(256, 256), border_type=None):
        if not os.path.exists(anno_pth):
            meta = os.path.splitext(os.path.basename(anno_pth))[0]
            return BoxesLabel(meta=meta, img_size=img_size)
        else:
            root = ET.parse(anno_pth).getroot()
            boxes = _xmlrd_boxes(root, name2cind=name2cind, num_cls=num_cls)

            boxes.meta = os.path.splitext(os.path.basename(anno_pth))[0]
            if border_type is not None: boxes.as_border_type(border_type)
        return boxes

    # 重命名
    @staticmethod
    def rename_anno_obj(anno_pth, anno_pth_new, rename_dict):
        if not os.path.exists(anno_pth):
            BROADCAST('Anno not exist ' + anno_pth)
            return {}
        stat_dict = dict([(key, 0) for key in rename_dict.keys()])
        root = ET.parse(anno_pth).getroot()
        unchanged = True
        objs_rmv = []
        for i, obj in enumerate(root):
            if obj.tag not in TAGS_BOXITEM:
                continue
            name = obj.find(TAG_NAME).text
            if name not in rename_dict.keys():
                continue
            elif rename_dict[name] is None:
                objs_rmv.append(obj)
                unchanged = False
            else:
                name_new = rename_dict[name]
                obj.find(TAG_NAME).text = name_new
                unchanged = False
            stat_dict[name] += 1
        for obj in objs_rmv:
            root.remove(obj)
        if unchanged and anno_pth_new == anno_pth:
            pass
        elif unchanged and not anno_pth_new == anno_pth:
            shutil.copy(anno_pth, anno_pth_new)
        else:
            root_new = ET.ElementTree(root)
            root_new.write(anno_pth_new, encoding='utf-8')
        return stat_dict

    @staticmethod
    def resort_anno_obj(anno_pth: str, anno_pth_new: str, by_area: bool = True, by_name: bool = True):
        root = ET.parse(anno_pth).getroot()
        objs = root.findall('object')
        # 删除
        for obj in objs:
            root.remove(obj)
        # 按名称
        if by_name:
            vals = np.array([obj.find('name').text for obj in objs])
            objs = [objs[ind] for ind in np.argsort(vals, kind='stable')]
        # 按面积
        if by_area:
            vals = []
            for obj in objs:
                box = obj.find('bndbox')
                x1, x2 = float(box.find('xmin').text), float(box.find('xmax').text)
                y1, y2 = float(box.find('ymin').text), float(box.find('ymax').text)
                vals.append(-(x2 - x1) * (y2 - y1))
            vals = np.array(vals)
            objs = [objs[ind] for ind in np.argsort(vals, kind='stable')]
        # 添加
        for obj in objs:
            root.append(obj)
        root = ET.ElementTree(root)
        root.write(anno_pth_new)
        return True

    # </editor-fold>

    @property
    def folders(self):
        return [self._img_folder, self._anno_folder, self._set_folder]

    def __repr__(self):
        msg = super(VOCDetectionDataset, self).__repr__() + '\n'
        num_dict = Counter(self.names)
        width = max([len(name) for name in num_dict.keys()])
        msg += '\n'.join([str(i) + ' | ' + name.ljust(width) + ' | %-6d' % num
                          for i, (name, num) in enumerate(num_dict.items())])
        return msg

    def stat(self) -> pd.DataFrame:
        names = self.names
        sizes = self.sizes
        report = pd.DataFrame(dict(name='image', **_analysis_sizes(self.img_sizes)), index=[0])
        names_u = np.unique(names)
        for i, name_u in enumerate(sorted(names_u)):
            report = pd.concat(
                [report, pd.DataFrame(dict(name=name_u, **_analysis_sizes(sizes[name_u == names])), index=[0])])
        return report

    # 得到物体尺寸
    @property
    def sizes(self):
        return VOCDetectionDataset.collect_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def names(self):
        return VOCDetectionDataset.collect_names(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def img_sizes(self):
        return VOCDetectionDataset.collect_img_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def labels(self):
        labels = [VOCDetectionDataset.load_anno(
            anno_pth, img_size=None, name2cind=self.name2cind, num_cls=self.num_cls, border_type=self.border_type)
            for anno_pth in self.anno_pths]
        return labels

    def _meta2label(self, meta: str):
        anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
        label = VOCDetectionDataset.load_anno(
            anno_pth, name2cind=self.name2cind, num_cls=self.num_cls, img_size=None,
            border_type=self.border_type)
        return label

    def _meta2data(self, meta: str):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
        img = load_img_pil(img_pth)
        anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
        label = VOCDetectionDataset.load_anno(
            anno_pth, name2cind=self.name2cind, num_cls=self.num_cls, img_size=img.size,
            border_type=self.border_type)
        return img, label

    @property
    def objnums(self):
        num_dict = Counter(self.names)
        nums = np.zeros(len(self.cls_names))
        for cls_name, num in num_dict.items():
            nums[self.cls_names.index(cls_name)] = num
        return nums

    # 重命名物体
    def raname_annos_obj_(self, rename_dict, anno_folder='Annotations2'):
        anno_dir = os.path.join(self.root, anno_folder)

        BROADCAST('Rename annos to ' + anno_dir)
        stat_dict = dict([(key, 0) for key in rename_dict.keys()])
        ensure_folder_pth(anno_dir)
        for k, anno_pth in MEnumerate(self.anno_pths):
            anno_pth_new = os.path.join(anno_dir, os.path.basename(anno_pth))
            stat_dict_i = VOCDetectionDataset.rename_anno_obj(anno_pth, anno_pth_new, rename_dict)
            for key in stat_dict.keys():
                stat_dict[key] += stat_dict_i[key]
        for key in stat_dict.keys():
            new_name = rename_dict[key]
            if new_name is not None:
                BROADCAST('%30s' % key + ' -> %-30s' % new_name + ' : ' + '%-10d' % stat_dict[key])
            else:
                BROADCAST('%30s' % key + ' remove : ' + '%-10d' % stat_dict[key])
        return self

    def resort_annos_obj_(self, by_area: bool = True, by_name: bool = True, anno_folder='Annotations2'):
        anno_dir = os.path.join(self.root, anno_folder)
        BROADCAST('Rename annos to ' + anno_dir)
        ensure_folder_pth(anno_dir)
        for i,anno_pth in MEnumerate(self.anno_pths):
            anno_pth_new = os.path.join(anno_dir, os.path.basename(anno_pth))
            VOCDetectionDataset.resort_anno_obj(anno_pth, anno_pth_new, by_area=by_area, by_name=by_name)
        return self

    def dump(self, labels, anno_folder='Annotations2', with_recover=True, anno_extend='xml',
             prefix='Create', broadcast=BROADCAST):
        folders = [anno_folder]
        anno_dir, = ensure_folders(self.root, folders)
        broadcast(dsmsgfmtr_create(self.root, self.set_name, folders, prefix=prefix))
        for i, boxes in MEnumerate(labels, broadcast=broadcast):
            anno_pth = os.path.join(anno_dir, ensure_extend(boxes.meta, anno_extend))
            if with_recover:
                boxes.recover()
            VOCDetectionWriter.save_anno(anno_pth, boxes=boxes)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    def append(self, imgs, labels, anno_folder='Annotations2', img_folder='JPEGImages2', set_name='appendx',
               with_recover=True, prefix='Append', broadcast=BROADCAST):
        folders = [img_folder, anno_folder, self.set_folder]
        img_dir, anno_dir, set_dir = ensure_folders(self.root, folders)
        broadcast(dsmsgfmtr_create(self.root, self.set_name, folders, prefix=prefix))
        metas = []
        for i, boxes in MEnumerate(labels, broadcast=broadcast):
            meta = boxes.meta
            anno_pth = os.path.join(anno_dir, meta + '.' + self.anno_extend)
            img_pth = os.path.join(img_dir, meta + '.' + self.img_extend)
            if with_recover:
                boxes.recover()
            imgs[i].save(img_pth)
            VOCDetectionWriter.save_anno(anno_pth, boxes=boxes)
            metas.append(meta)
        save_txt(os.path.join(set_dir, set_name + '.txt'), metas)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    def update(self, labels_md, anno_folder='Annotations', with_recover=True, anno_extend='xml',
               prefix='Update', broadcast=BROADCAST):
        folders = [anno_folder]
        anno_dir, = ensure_folders(self.root, folders)
        broadcast(dsmsgfmtr_create(self.root, self.set_name, [anno_folder], prefix=prefix))
        for i, boxes in MEnumerate(labels_md, broadcast=broadcast):
            anno_pth = os.path.join(anno_dir, ensure_extend(boxes.meta, anno_extend))
            if with_recover:
                boxes.recover()
            boxes_ori = VOCDetectionDataset.load_anno(anno_pth)
            for box in boxes:
                ind = box['ind']
                box_ori = boxes_ori[ind]
                box_ori.border = box.border
            VOCDetectionWriter.save_anno(anno_pth, boxes=boxes_ori)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    # 执行增广生成新标签
    def apply(self, func, img_folder='JPEGImages2', anno_folder='Annotations2', img_extend='jpg', anno_extend='xml'
              , prefix='Apply', broadcast=BROADCAST, ):
        folders_src = [self.img_folder, self.anno_folder]
        folders_dst = [img_folder, anno_folder]
        img_dir, anno_dir = ensure_folders(self.root, folders_dst)
        broadcast(dsmsgfmtr_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
            img = load_img_pil(img_pth)
            boxes = VOCDetectionDataset.load_anno(anno_pth, name2cind=self.name2cind, num_cls=self.num_cls,
                                                  img_size=img.size, border_type=self.border_type)
            img_cvt, boxes_cvt = func(img, boxes)
            if img_dir is not None and img_cvt is not None:
                img_pth_new = os.path.join(img_dir, ensure_extend(meta, img_extend))
                img2imgP(img_cvt).save(img_pth_new)
            if anno_dir is not None and boxes_cvt is not None:
                anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
                VOCDetectionWriter.save_anno(anno_pth_new, boxes_cvt)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    # 执行增广生成新标签
    def copy(self, func, img_folder='JPEGImages2', anno_folder='Annotations2', img_extend='jpg',
             anno_extend='xml', prefix='Copy', broadcast=BROADCAST):
        folders_src = [self.img_folder, self.anno_folder]
        folders_dst = [img_folder, anno_folder]
        img_dir, anno_dir = ensure_folders(self.root, folders_dst)
        broadcast(dsmsgfmtr_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))

            img = load_img_pil(img_pth)
            label = VOCDetectionDataset.load_anno(anno_pth, name2cind=self.name2cind, num_cls=self.num_cls,
                                                  img_size=img.size, border_type=self.border_type)
            img, label = func(img, label)
            if img is None or label is None:
                continue
            if img_dir is not None:
                img_pth_new = os.path.join(img_dir, ensure_extend(meta, img_extend))
                img2imgP(img).save(img_pth_new)
            if anno_dir is not None:
                anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
                VOCDetectionWriter.save_anno(anno_pth_new, label)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    def create_set(self, set_name, fltr=None):
        if fltr is None:
            metas = self.metas
        else:
            metas = []
            for anno_pth in self.anno_pths:
                label = VOCDetectionDataset.load_anno(anno_pth)
                if fltr(label):
                    metas.append(label.meta)
        BROADCAST('Create set [ ' + set_name + ' ] with %d datas' % len(metas))
        save_txt(os.path.join(self.set_dir, set_name), metas)
        return metas

    # 使用函数对标签形式进行变换
    def label_apply(self, func, anno_folder='Annotations2', anno_extend='xml', prefix='Apply', broadcast=BROADCAST):
        folders_dst = [anno_folder]
        anno_dir, = ensure_folders(self.root, folders_dst)
        broadcast(dsmsgfmtr_apply(self.root, self.set_name, [self.anno_folder], folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast, step=500):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            boxes = VOCDetectionDataset.load_anno(
                anno_pth, name2cind=self.name2cind, num_cls=self.num_cls, border_type=self.border_type)
            boxes_cvt = func(boxes)
            VOCDetectionWriter.save_anno(anno_pth_new, boxes=boxes_cvt)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True


# </editor-fold>

# <editor-fold desc='语义分割'>
class VOCSegmentationDataset(MColorMapper, VOCDataset):
    MASK_EXTEND = 'png'
    IMG_EXTEND = VOCDataset.IMG_EXTEND
    SET_EXTEND = VOCDataset.SET_EXTEND

    def __init__(self, root, set_name, cls_names=None, colors=None, rgn_type=None,
                 set_folder='ImageSets/Segmentation', img_folder='JPEGImages', mask_folder='SegmentationClass',
                 mask_extend=MASK_EXTEND, img_extend=IMG_EXTEND, num_oversamp: int = 1, data_mode=DATA_MODE.FULL,
                 **kwargs):
        MColorMapper.__init__(self, cls_names=cls_names, colors=colors)
        VOCDataset.__init__(self, root=root, set_name=set_name, set_folder=set_folder,
                            img_folder=img_folder, img_extend=img_extend, num_oversamp=num_oversamp,
                            data_mode=data_mode)
        self.mask_extend = mask_extend
        self.mask_folder = mask_folder
        self.rgn_type = rgn_type

    @property
    def mask_folder(self):
        return self._mask_folder

    @property
    def mask_dir(self):
        return os.path.join(self.root, self._mask_folder)

    @property
    def mask_pths(self):
        return _expand_pths(self.root, folder=self.mask_folder, metas=self.metas, extend=self.mask_extend)

    @mask_folder.setter
    def mask_folder(self, mask_folder):
        self._mask_folder = mask_folder

    # <editor-fold desc='VOC工具'>
    @staticmethod
    def prase_mask(mask_pth, colors, cind2name=None, num_cls=1, img_size=(256, 256)):
        meta = os.path.splitext(os.path.basename(mask_pth))[0]
        if not os.path.exists(mask_pth):
            return SegsLabel(img_size=img_size, meta=meta)
        segs = maskrd(mask_pth, colors=colors, num_cls=num_cls)
        segs.meta = meta
        for seg in segs:
            cind = IndexCategory.convert(seg.category)._cindN
            if cind2name is not None:
                seg['name'] = cind2name(cind)
        return segs

    # </editor-fold>

    @property
    def labels(self):
        labels = []
        for meta in self.metas:
            mask_pth = os.path.join(self.root, self.mask_folder, ensure_extend(meta, self.mask_extend))
            segs = VOCSegmentationDataset.prase_mask(mask_pth, colors=self.colors, cind2name=self.cind2name,
                                                     num_cls=self.num_cls, img_size=None)
            labels.append(segs)
        return labels

    def _meta2label(self, meta: str):
        mask_pth = os.path.join(self.root, self.mask_folder, ensure_extend(meta, self.mask_extend))
        segs = VOCSegmentationDataset.prase_mask(
            mask_pth, colors=self.colors, cind2name=self.cind2name, num_cls=self.num_cls, img_size=None)
        if self.rgn_type is not None:
            for seg in segs: seg.rgn = self.rgn_type.convert(seg.rgn)
        return segs

    def _meta2data(self, meta):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
        mask_pth = os.path.join(self.root, self.mask_folder, ensure_extend(meta, self.mask_extend))
        img = load_img_pil(img_pth)
        segs = VOCSegmentationDataset.prase_mask(
            mask_pth, colors=self.colors, cind2name=self.cind2name, num_cls=self.num_cls, img_size=img.size)
        if self.rgn_type is not None:
            for seg in segs: seg.rgn = self.rgn_type.convert(seg.rgn)
        return img, segs

    def dump(self, labels, mask_folder='SegmentationClass'):
        mask_dir = os.path.join(self.root, mask_folder)
        ensure_folder_pth(mask_dir)
        for segs in labels:
            mask_pth = os.path.join(mask_dir, segs.meta + '.' + self.mask_extend)
            VOCSegmentationWriter.save_mask(mask_pth=mask_pth, segs=segs, colors=self.colors)
        return True

    @property
    def folders(self):
        return [self._img_folder, self._mask_folder, self._set_folder]

    # 统计物体个数
    def __repr__(self):
        msg = VOCDataset.__repr__(self) + '\n'
        num_dict = {}
        width = max([len(name) for name in num_dict.keys()])
        msg += '\n'.join([str(i) + ' | ' + name.ljust(width) + ' | %-6d' % num
                          for i, (name, num) in enumerate(num_dict.items())])
        return msg


# </editor-fold>

# <editor-fold desc='实例分割'>

class VOCInstanceDataset(MColorMapper, VOCDataset):
    INST_EXTEND = 'png'
    ANNO_EXTEND = 'xml'
    IMG_EXTEND = VOCDataset.IMG_EXTEND
    SET_EXTEND = VOCDataset.SET_EXTEND

    def __init__(self, root, set_name, cls_names=None, colors=None,
                 set_folder='ImageSets/Segmentation', img_folder='JPEGImages', inst_folder='SegmentationClass',
                 anno_folder='Annotations', img_extend=IMG_EXTEND, inst_extend=INST_EXTEND,
                 anno_extend=ANNO_EXTEND, rgn_type=None, border_type=None, num_oversamp: int = 1,
                 data_mode=DATA_MODE.FULL, **kwargs):
        MColorMapper.__init__(self, cls_names=cls_names, colors=colors)
        VOCDataset.__init__(self, root=root, set_name=set_name, set_folder=set_folder, img_folder=img_folder,
                            img_extend=img_extend, num_oversamp=num_oversamp, data_mode=data_mode)
        self.inst_extend = inst_extend
        self.anno_extend = anno_extend
        self.inst_folder = inst_folder
        self.anno_folder = anno_folder
        self.rgn_type = rgn_type
        self.border_type = border_type
        # 其它属性
        self.rgn_type = rgn_type

    @property
    def anno_folder(self):
        return self._anno_folder

    @property
    def anno_dir(self):
        return os.path.join(self.root, self._anno_folder)

    @anno_folder.setter
    def anno_folder(self, anno_folder):
        self._anno_folder = anno_folder

    @property
    def anno_pths(self):
        return _expand_pths(self.root, folder=self.anno_folder, metas=self.metas, extend=self.anno_extend)

    @property
    def inst_folder(self):
        return self._inst_folder

    @inst_folder.setter
    def inst_folder(self, inst_folder):
        self._inst_folder = inst_folder

    @property
    def inst_pths(self):
        return _expand_pths(self.root, folder=self.inst_folder, metas=self.metas, extend=self.inst_extend)

    @property
    def inst_dir(self):
        return os.path.join(self.root, self._inst_folder)

    @property
    def sizes(self):
        return VOCDetectionDataset.collect_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def names(self):
        return VOCDetectionDataset.collect_names(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def img_sizes(self):
        return VOCDetectionDataset.collect_img_sizes(
            anno_dir=self.anno_dir, anno_extend=self.anno_extend, metas=self._metas)

    @property
    def objnums(self):
        cls_names = np.array(self.cls_names)
        vec = np.sum(cls_names == np.array(self.names)[..., None], axis=0)
        return vec

    # <editor-fold desc='VOC工具'>
    @staticmethod
    def prase_anno_inst(anno_pth, inst_pth, colors, name2cind=None, num_cls=1, img_size=(256, 256), border_type=None,
                        rgn_type=None):
        meta = os.path.splitext(os.path.basename(inst_pth))[0]
        boxes = VOCDetectionDataset.load_anno(anno_pth=anno_pth, name2cind=name2cind, num_cls=num_cls,
                                              img_size=img_size, border_type=border_type)
        if not os.path.exists(inst_pth):
            BROADCAST('inst not exist ' + inst_pth)
            return boxes
        insts = instrd(inst_pth, boxes=boxes, colors=colors, rgn_type=rgn_type)
        insts.meta = meta
        insts.kwargs = boxes.kwargs
        return insts

    # </editor-fold>

    @property
    def folders(self):
        return [self._img_folder, self._anno_folder, self._inst_folder, self._set_folder]

    @property
    def labels(self):
        labels = []
        for anno_pth, inst_pth in zip(self.anno_pths, self.inst_pths):
            label = VOCInstanceDataset.prase_anno_inst(
                anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=None,
                num_cls=self.num_cls, rgn_type=self.rgn_type, border_type=self.border_type)
            labels.append(label)
        return labels

    def _meta2label(self, meta: str):
        anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
        inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
        label = VOCInstanceDataset.prase_anno_inst(
            anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=None, num_cls=self.num_cls,
            rgn_type=self.rgn_type, border_type=self.border_type)
        return label

    def _meta2data(self, meta: str):
        img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
        anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
        inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
        img = load_img_pil(img_pth)
        label = VOCInstanceDataset.prase_anno_inst(
            anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=img.size, num_cls=self.num_cls,
            rgn_type=self.rgn_type, border_type=self.border_type)
        return img, label

    # 统计物体个数
    def __repr__(self):
        msg = VOCDataset.__repr__(self) + '\n'
        num_dict = Counter(self.names)
        width = max([len(name) for name in num_dict.keys()])
        msg += '\n'.join(['%3d' % i + ' | ' + name.ljust(width) + ' | %-6d' % num
                          for i, (name, num) in enumerate(num_dict.items())])
        return msg

    def update(self, labels_md, anno_folder='Annotations', inst_folder='SegmentationClass', with_recover=True,
               anno_extend='xml', inst_extend='png', prefix='Update ', broadcast=BROADCAST):
        folders_dst = [anno_folder, inst_folder]
        anno_dir, inst_dir = ensure_folders(self.root, folders_dst)
        broadcast(dsmsgfmtr_create(self.root, self.set_name, folders_dst, prefix=prefix))
        for i, label in MEnumerate(labels_md, broadcast=broadcast):
            meta = label.meta
            assert meta is not None
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))

            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            inst_pth_new = os.path.join(inst_dir, ensure_extend(meta, inst_extend))
            assert isinstance(label, InstsLabel), 'fmt err ' + label.__class__.__name__
            if with_recover:
                label.recover()
            insts_ori = VOCInstanceDataset.prase_anno_inst(anno_pth, inst_pth, self.colors)
            for inst, inst_ori in zip(label, insts_ori):
                inst_ori.rgn = inst.rgn
                inst_ori.update(inst)

            VOCInstanceWriter.save_anno_inst(anno_pth_new, inst_pth_new, self.colors, insts_ori)

        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    def label_apply(self, func, anno_folder='Annotations2', inst_folder='SegmentationClass2', anno_extend='xml',
                    inst_extend='png', prefix='Apply', broadcast=BROADCAST):
        folders_src = [self.anno_folder, self.inst_folder]
        folders_dst = [anno_folder, inst_folder]
        anno_dir, inst_dir = ensure_folders(self.root, folders_dst)
        broadcast(dsmsgfmtr_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
            insts = VOCInstanceDataset.prase_anno_inst(
                anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind, img_size=None, num_cls=self.num_cls,
                rgn_type=self.rgn_type, border_type=self.border_type)
            insts_cvt = func(insts)
            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            inst_pth_new = os.path.join(inst_dir, ensure_extend(meta, inst_extend))
            VOCInstanceWriter.save_anno_inst(anno_pth_new, inst_pth_new, colors=self.colors, insts=insts_cvt, )
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True

    def apply(self, func, img_folder='JPEGImages2', anno_folder='Annotations2', inst_folder='SegmentationClass2',
              anno_extend='xml', inst_extend='png', img_extend='jpg', prefix='Apply', broadcast=BROADCAST):
        folders_src = [self.img_folder, self.anno_folder, self.inst_folder]
        folders_dst = [img_folder, anno_folder, inst_folder]
        img_dir, anno_dir, inst_dir = ensure_folders(self.root, folders_dst)
        broadcast(dsmsgfmtr_apply(self.root, self.set_name, folders_src, folders_dst, prefix=prefix))
        for i, meta in MEnumerate(self.metas, broadcast=broadcast):
            anno_pth = os.path.join(self.root, self.anno_folder, ensure_extend(meta, self.anno_extend))
            img_pth = os.path.join(self.root, self.img_folder, ensure_extend(meta, self.img_extend))
            inst_pth = os.path.join(self.root, self.inst_folder, ensure_extend(meta, self.inst_extend))
            img = load_img_pil(img_pth)
            insts = VOCInstanceDataset.prase_anno_inst(anno_pth, inst_pth, colors=self.colors, name2cind=self.name2cind,
                                                       img_size=None, num_cls=self.num_cls, rgn_type=self.rgn_type,
                                                       border_type=self.border_type)
            img_cvt, insts_cvt = func(img, insts)
            anno_pth_new = os.path.join(anno_dir, ensure_extend(meta, anno_extend))
            inst_pth_new = os.path.join(inst_dir, ensure_extend(meta, inst_extend))
            img_pth_new = os.path.join(img_dir, ensure_extend(meta, img_extend))
            img_cvt.save(img_pth_new)
            VOCInstanceWriter.save_anno_inst(anno_pth_new, inst_pth_new, colors=self.colors, insts=insts_cvt, )
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True


# </editor-fold>

class ColorGenerator():
    def __init__(self, low=30, high=200):
        self.low = low
        self.high = high

    def __getitem__(self, index):
        return random_color(low=self.low, high=self.high, index=index, unit=False)


class VOCCommon(MDataSource):
    SET_FOLDER = 'ImageSets/Main'
    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = 'SegmentationClass'
    INST_FOLDER = 'SegmentationObject'
    COLORS = ColorGenerator(low=30, high=200)

    CLS_NAMES = ()

    REGISTER_BUILDER = {
        TASK_TYPE.DETECTION: VOCDetectionDataset,
        TASK_TYPE.SEGMENTATION: VOCSegmentationDataset,
        TASK_TYPE.INSTANCESEG: VOCInstanceDataset,
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS,
                 task_type=TASK_TYPE.DETECTION,
                 data_mode=DATA_MODE.FULL,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER,
                 img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER, with_border=True, set_names=None, **kwargs):
        root = self.__class__.get_root() if root is None else root
        if set_names is None:
            set_dir = os.path.join(root, set_folder)
            if os.path.exists(set_dir):
                set_names = [os.path.splitext(file_name)[0] for file_name in listdir_extend(set_dir, 'txt')]
            else:
                set_names = tuple()
        MDataSource.__init__(self, root=root, set_names=set_names, task_type=task_type)
        self.set_folder = set_folder
        self.img_folder = img_folder
        self.anno_folder = anno_folder
        self.mask_folder = mask_folder
        self.inst_folder = inst_folder
        self.cls_names = cls_names
        self.colors = colors
        self.data_mode = data_mode
        self.with_border = with_border
        self.kwargs = kwargs

    def _dataset(self, set_name='train', **kwargs):
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names, root=self.root,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors,
                             task_type=self.task_type, data_mode=self.data_mode)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        builder = self.__class__.REGISTER_BUILDER[kwargs_update.get('task_type')]
        dataset = builder(**kwargs_update)
        return dataset


class VOC(VOCCommon):
    SET_FOLDER_DET = 'ImageSets/Main'
    SET_FOLDER_SEG = 'ImageSets/Segmentation'
    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = 'SegmentationClass'
    INST_FOLDER = 'SegmentationObject'
    CLS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                 'bus', 'car', 'cat', 'chair', 'cow',
                 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    COLORS = ((128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
              (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
              (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
              (128, 64, 128), (0, 192, 128), (128, 192, 128), (64, 64, 0), (192, 64, 0),
              (64, 192, 0), (192, 192, 0), (64, 64, 128), (192, 64, 128), (64, 192, 128),
              (192, 192, 128), (0, 0, 64), (128, 0, 64), (0, 128, 0), (128, 128, 64),
              (0, 0, 192), (128, 0, 192), (0, 128, 192))
    BORDER_COLOR = (224, 224, 192)
    RAND_COLORS = ColorGenerator(low=30, high=200)

    SUB_2007 = '2007'
    SUB_2012 = '2012'
    SUB_0712 = '0712'
    SUB_NONE = None

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//VOC//',
        PLATFORM_DESTOPLAB: 'D://Datasets//VOC//',
        PLATFORM_SEV3090: '//home//data-storage//VOC',
        PLATFORM_SEV4090: '//home//data-storage//VOC',
        PLATFORM_SEVTAITAN: '/home/exspace/dataset//VOC2007',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 data_mode=DATA_MODE.FULL,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER_DET,
                 img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val'), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder, data_mode=data_mode,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)

    def _dataset(self, set_name='train', sub=SUB_2007, **kwargs):
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names, data_mode=self.data_mode,
                             task_type=self.task_type,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name,
                             mask_folder=self.mask_folder, inst_folder=self.inst_folder, colors=self.colors,
                             with_border=self.with_border)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        builder = self.__class__.REGISTER_BUILDER[kwargs_update.get('task_type')]
        if sub == VOC.SUB_0712:
            ds07 = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2007'),
                           fmt=set_name + '_07_%d', **kwargs_update)
            ds12 = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2012'),
                           fmt=set_name + '_12_%d', **kwargs_update)
            dataset = MConcatDataset([ds07, ds12])

        elif sub == VOC.SUB_2007:
            dataset = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2007'),
                              **kwargs_update)
        elif sub == VOC.SUB_2012:
            dataset = builder(root=os.path.join(self.root, 'VOCdevkit', 'VOC2012'),
                              **kwargs_update)
        elif sub == VOC.SUB_NONE:
            dataset = builder(root=self.root, **kwargs_update)
        else:
            raise Exception('err sub ' + sub.__class__.__name__)
        return dataset

# if __name__ == '__main__':
#     voc = Voc.SEV_NEW()
#     loader = voc.loader(set_name='test', batch_size=4, num_workers=0, aug_seq=None)
#     imgs, labels = next(iter(loader))
# a = np.array(labels[0])

# if __name__ == '__main__':
#     ds = Voc.SEV_NEW(task_type=TASK_TYPE.INSTANCE, set_folder=Voc.SET_FOLDER_SEG)
#     dataset = ds.dataset('train', sub=Voc.SUB_2012)
#     for img, label in dataset:
#         print(label.meta)
#         pass
