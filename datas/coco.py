try:
    import pycocotools
    import pycocotools.mask as mask_utils
    from pycocotools.coco import COCO as PyCOCO
    from pycocotools.cocoeval import COCOeval
except Exception as e:
    pass
from datas.base import *


def load_json_labelme(json_pth: str, img_size: tuple = (500, 500), name2cind=None, num_cls: int = 1):
    if not os.path.exists(json_pth):
        return []
    json_dct = load_json(json_pth)
    shapes = json_dct['shapes']
    items = []
    for shape in shapes:
        xyp = np.array(shape['points']).reshape(-1, 2)
        border = XYPBorder(xyp, size=img_size)
        name = shape['label']
        cind = 0 if name2cind is None else name2cind(name)
        category = IndexCategory(cind, num_cls=num_cls)
        items.append(BoxItem(category=category, border=border, name=name))
    return items


def _prase_coco_dct(coco_dct: dict) -> (Dict, List, List):
    id2name_dct = dict()
    for cate in coco_dct['categories']:
        id2name_dct[cate['id']] = cate['name']
    anno_dct = dict()
    for obj_anno in coco_dct['annotations']:
        id = obj_anno['image_id']
        if id in anno_dct.keys():
            anno_dct[id].append(obj_anno)
        else:
            anno_dct[id] = [obj_anno]
    img_annos = coco_dct['images']
    obj_annoss = []
    for img_anno in img_annos:
        obj_annoss.append(anno_dct.get(img_anno['id'], []))
    return id2name_dct, img_annos, obj_annoss


# <editor-fold desc='写入'>
class COCOWriter(LabelWriter):

    def __init__(self, anno_pth: str, ann_id_init: int = 0, img_id_init: int = 0, name2id: Optional[Callable] = None,
                 with_score: bool = False, img_extend: str = 'jpg'):
        self.anno_pth = anno_pth
        self.ann_id_init = ann_id_init
        self.img_id_init = img_id_init
        self.name2id = name2id
        self.with_score = with_score
        self.img_extend = img_extend

    @staticmethod
    def binary_mask2rle(maskN: np.ndarray, as_list=True) -> Dict:
        if not as_list:
            rle = mask_utils.encode(np.array(maskN, order='F', dtype=np.uint8))
            rle['counts'] = rle['counts'].decode('utf-8')
        else:
            binary_arr = maskN.ravel(order='F')
            binary_diff = np.diff(binary_arr, axis=0, prepend=0, append=0)
            binary_diff[-1] = 1
            counts_abs = np.nonzero(binary_diff)[0]
            counts = np.diff(counts_abs, axis=0, prepend=0).astype(int)
            rle = {'counts': counts.tolist(), 'size': list(maskN.shape)}
        return rle

    @staticmethod
    def label2json_anno(label: ImageItemsLabel, img_id: int = 0, name2id: Optional[Callable] = None,
                        with_score: bool = False, with_rgn: bool = True, seg_as_list: bool = True, ):
        annotations = []
        for ann_id, item in enumerate(label):
            name = item['name']
            cind = int(item.category.cindN) if name2id is None else name2id(name)
            xyxy = item.xyxyN.astype(float)
            xydwdh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            ann_dict = {'id': ann_id, 'image_id': img_id, 'category_name': name, 'category_id': cind,
                        'bbox': xydwdh, 'iscrowd': 0, 'area': xydwdh[2] * xydwdh[3]}
            if with_score:
                ann_dict['score'] = item.category.confN
            if with_rgn:
                item = InstItem.convert(item)
                if isinstance(item.rgn, XYPBorder):
                    segment = [list(item.rgn._xypN.reshape(-1).astype(float))]
                else:
                    segment = COCOWriter.binary_mask2rle(item.rgn.maskNb, as_list=seg_as_list)
                area = float(item.rgn.area)
                ann_dict['segmentation'] = segment
                ann_dict['area'] = area
            annotations.append(ann_dict)
        return annotations

    @staticmethod
    def label2json_item(label: ImageItemsLabel, img_id: int = 0, img_extend: str = 'jpg',
                        name2id: Optional[Callable] = None,
                        with_score: bool = False, with_rgn: bool = True, seg_as_list: bool = True, ):
        assert isinstance(label, ImageItemsLabel)
        w, h = label.img_size
        img_dct = {
            'id': img_id,
            'width': int(w), 'height': int(h),
            'file_name': ensure_extend(label.meta, img_extend, overwrite=True),
        }
        annotations = COCOWriter.label2json_anno(label, img_id=img_id, name2id=name2id, with_score=with_score,
                                                 with_rgn=with_rgn, seg_as_list=seg_as_list)
        return (img_dct, annotations)

    @staticmethod
    def json_itemsjson_dct(json_items: List, ann_id_init: int = 0, img_id_init: int = 0):
        img_infos = []
        categories_all = {}
        annotations_all = []
        ann_id = ann_id_init
        img_id = img_id_init
        for img_dct, annotations in json_items:
            img_dct['id'] = img_id

            for anno in annotations:
                anno['image_id'] = img_id
                anno['id'] = ann_id
                categories_all[anno['category_id']] = anno['category_name']
                ann_id = ann_id + 1
                annotations_all.append(anno)
            img_infos.append(img_dct)
            img_id = img_id + 1
        categories_all = [{'id': id, 'name': name} for id, name in categories_all.items()]
        json_dict = {
            'images': img_infos,
            'categories': categories_all,
            'annotations': annotations_all,
        }
        return json_dict

    @staticmethod
    def labels2json_dct(labels: Sequence[ImageItemsLabel], img_extend: str = 'jpg',
                        name2id: Optional[Callable] = None, ann_id_init: int = 0, img_id_init: int = 0,
                        with_score: bool = False, with_rgn: bool = True, seg_as_list: bool = True, ):
        json_items = []
        for label in labels:
            json_item = COCODetectionWriter.label2json_item(
                label, img_id=0, img_extend=img_extend, with_score=with_score, with_rgn=with_rgn,
                seg_as_list=seg_as_list, name2id=name2id)
            json_items.append(json_item)
        json_dct = COCOWriter.json_itemsjson_dct(json_items, ann_id_init=ann_id_init, img_id_init=img_id_init)
        return json_dct

    @staticmethod
    def json_dct2coco_obj(json_dct):
        coco_obj = PyCOCO()
        coco_obj.dataset = json_dct
        coco_obj.createIndex()
        return coco_obj

    @staticmethod
    def labels2coco_obj(labels: Sequence[ImageItemsLabel], img_extend: str = 'jpg',
                        name2id: Optional[Callable] = None, ann_id_init: int = 0, img_id_init: int = 0,
                        with_score: bool = False, with_rgn: bool = True, seg_as_list: bool = True, ):
        json_dct = COCOWriter.labels2json_dct(labels, name2id=name2id, img_extend=img_extend,
                                              with_score=with_score, with_rgn=with_rgn, ann_id_init=ann_id_init,
                                              img_id_init=img_id_init, seg_as_list=seg_as_list)
        return COCODataset.json_dct2coco_obj(json_dct)

    @staticmethod
    def save_labels2json_dct(labels: Sequence[ImageItemsLabel], img_extend: str = 'jpg',
                             name2id: Optional[Callable] = None, ann_id_init: int = 0, img_id_init: int = 0,
                             with_score: bool = False, with_rgn: bool = True, seg_as_list: bool = True, ):
        json_dct = COCOWriter.labels2json_dct(labels, name2id=name2id, img_extend=img_extend,
                                              with_score=with_score, with_rgn=with_rgn, ann_id_init=ann_id_init,
                                              img_id_init=img_id_init, seg_as_list=seg_as_list)

        return json_dct

    def save_all(self, caches: List):
        json_dct = COCOWriter.json_itemsjson_dct(caches, ann_id_init=self.ann_id_init, img_id_init=self.img_id_init)
        save_json(self.anno_pth, json_dct)
        return json_dct


class COCODetectionWriter(COCOWriter):

    def save_label(self, label: ImageItemsLabel) -> object:
        return COCOWriter.label2json_item(
            label=label, img_id=0, img_extend=self.img_extend, name2id=self.name2id,
            with_score=self.with_score, with_rgn=False, seg_as_list=True)


class COCOSegmentationWriter(COCOWriter):

    def __init__(self, anno_pth: str, ann_id_init: int = 0, img_id_init: int = 0, name2id: Optional[Callable] = None,
                 with_score: bool = False, img_extend: str = 'jpg', seg_as_list: bool = True):
        COCOWriter.__init__(self, anno_pth, ann_id_init=ann_id_init, img_id_init=img_id_init, with_score=with_score,
                            name2id=name2id, img_extend=img_extend)
        self.seg_as_list = seg_as_list

    def save_label(self, label: ImageItemsLabel) -> object:
        return COCOWriter.label2json_item(
            label=label, img_id=0, img_extend=self.img_extend, name2id=self.name2id,
            with_score=self.with_score, with_rgn=True, seg_as_list=self.seg_as_list)


class COCOInstanceWriter(COCOWriter):

    def __init__(self, anno_pth: str, ann_id_init: int = 0, img_id_init: int = 0, name2id: Optional[Callable] = None,
                 with_score: bool = False, img_extend: str = 'jpg', seg_as_list: bool = True):
        COCOWriter.__init__(self, anno_pth, ann_id_init=ann_id_init, img_id_init=img_id_init, with_score=with_score,
                            name2id=name2id, img_extend=img_extend)
        self.seg_as_list = seg_as_list

    def save_label(self, label: ImageItemsLabel) -> object:
        return COCOWriter.label2json_item(
            label=label, img_id=0, img_extend=self.img_extend, name2id=self.name2id,
            with_score=self.with_score, with_rgn=True, seg_as_list=self.seg_as_list)


# </editor-fold>

# <editor-fold desc='检测'>

class COCODataset(MNameMapper, MDataset):

    def __init__(self, root: str, set_name: str, json_name: str, img_folder: str = 'images',
                 json_folder: str = 'annotation', cls_names: Optional[Tuple[str]] = None,
                 num_oversamp: int = 1, data_mode=DATA_MODE.FULL, **kwargs):
        MDataset.__init__(self, root=root, set_name=set_name, num_oversamp=num_oversamp, data_mode=data_mode)
        self.img_folder = img_folder.replace(PLACEHOLDER.SET_NAME, set_name)
        self.json_folder = json_folder.replace(PLACEHOLDER.SET_NAME, set_name)
        self.json_name = json_name.replace(PLACEHOLDER.SET_NAME, set_name)
        json_pth = os.path.join(self.json_dir, ensure_extend(json_name, 'json'))
        if os.path.exists(json_pth):
            coco_dct = load_json(json_pth)
            self._id2name_dct, self.img_annos, self.obj_annoss = _prase_coco_dct(coco_dct)
            if cls_names is None:
                cls_names = list(sorted(self._id2name_dct.values()))
            self._metas = [img_anno['file_name'].split('.')[0] for img_anno in self.img_annos]
        else:
            img_names = os.listdir(self.img_dir)
            self.obj_annoss = [[]] * len(img_names)
            self.img_annos = [{'file_name': img_name} for img_name in img_names]
            self._id2name_dct = None
            self._metas = [img_name.split('.')[0] for img_name in img_names]
        MNameMapper.__init__(self, cls_names=cls_names)

    def id2name(self, id: int) -> str:
        if self._id2name_dct is not None:
            return self._id2name_dct[id]
        else:
            return ''

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def img_folder(self):
        return self._img_folder

    @img_folder.setter
    def img_folder(self, img_folder):
        self._img_folder = img_folder

    @property
    def img_dir(self):
        return os.path.join(self.root, self._img_folder)

    @property
    def json_folder(self):
        return self._json_folder

    @json_folder.setter
    def json_folder(self, json_folder):
        self._json_folder = json_folder

    @property
    def json_dir(self):
        return os.path.join(self.root, self._json_folder)

    @property
    def metas(self):
        return self._metas

    @property
    def img_pths(self):
        return [os.path.join(self.root, self._img_folder, img_anno['file_name']) for img_anno in self.img_annos]

    # <editor-fold desc='coco工具集'>

    @staticmethod
    def collect_names(obj_annoss, id2name=None):
        names = []
        for obj_annos in obj_annoss:
            for obj_anno in obj_annos:
                name = obj_anno['category_name'] if id2name is None \
                    else id2name(int(obj_anno['category_id']))
                names.append(name)
        return np.array(names)

    @staticmethod
    def collect_img_sizes(img_annos):
        img_sizes = []
        for img_anno in img_annos:
            img_sizes.append((img_anno['width'], img_anno['height']))
        return np.array(img_sizes)

    @staticmethod
    def collect_sizes(obj_annoss):
        sizes = []
        for obj_annos in obj_annoss:
            for obj_anno in obj_annos:
                xyxy = np.array(obj_anno['bbox'])
                sizes.append(xyxy[2:4])
        return np.array(sizes)

    @staticmethod
    def json_dct2coco_obj(json_dct):
        coco_obj = pycocotools.coco.COCO()
        coco_obj.dataset = json_dct
        coco_obj.createIndex()
        return coco_obj

    # </editor-fold>
    @property
    def sizes(self):
        return COCODataset.collect_sizes(self.obj_annoss)

    @property
    def names(self):
        return COCODataset.collect_names(self.obj_annoss, self.id2name)

    @property
    def objnums(self):
        num_dict = Counter(self.names)
        nums = np.zeros(len(self.cls_names))
        for cls_name, num in num_dict.items():
            nums[self.cls_names.index(cls_name)] = num
        return nums

    def __len__(self):
        return len(self._metas)

    def _index2img(self, index: int):
        img_ann = self.img_annos[index]
        img_pth = os.path.join(self.img_dir, img_ann['file_name'])
        img = load_img_pil(img_pth)
        return img

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _meta2data(self, meta: str):
        return self._index2data(self._metas.index(meta))

    def _meta2label(self, meta: str):
        return self._index2label(self._metas.index(meta))

    def rename(self, rename_dict, json_name='instances2'):
        json_pth = ensure_extend(os.path.join(self.root, self.json_folder, self.json_name), 'json')
        json_pth_new = ensure_extend(os.path.join(self.root, self.json_folder, json_name), 'json')
        if not os.path.exists(json_pth):
            return self
        json_dct = load_json(json_pth)
        categories = json_dct['categories']
        annotations = json_dct['annotations']
        for cate in categories:
            if cate['name'] in rename_dict.keys():
                cate['name'] = rename_dict[cate['name']]
        for anno in annotations:
            if anno['category_name'] in rename_dict.keys():
                anno['category_name'] = rename_dict[anno['category_name']]
        save_json(json_pth_new, json_dct)
        return self

    def rename_cind(self, name2cind, json_name='instances2'):
        json_pth = ensure_extend(os.path.join(self.root, self.json_folder, self.json_name), 'json')
        json_pth_new = ensure_extend(os.path.join(self.root, self.json_folder, json_name), 'json')
        if not os.path.exists(json_pth):
            return self
        json_dct = load_json(json_pth)
        categories = json_dct['categories']
        annotations = json_dct['annotations']
        for cate in categories:
            cate['id'] = name2cind(cate['name'])
        for anno in annotations:
            anno['category_id'] = name2cind(anno['category_name'])
        save_json(json_pth_new, json_dct)
        return self

    # 使用函数对标签形式进行变换
    def label_apply(self, func, json_name='instances2', img_extend='jpg'):
        json_pth_new = ensure_extend(os.path.join(self.root, self.json_folder, json_name), 'json')
        print('Convert Instance to ' + self.root + ' < ' + os.path.join(self.json_folder, json_name) + ' >')
        labels = []
        for i, img_ann in MEnumerate(self.img_annos, step=500):
            obj_annos = self.obj_annoss[i]
            insts = COCOInstanceDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_annos, num_cls=self.num_cls, id2name=self.id2name,
                name2cind=self.name2cind)
            insts = func(insts)
            labels.append(insts)
        json_dct = COCOWriter.labels2json_dct(labels, name2id=None, img_extend=img_extend)
        save_json(json_pth_new, json_dct)
        return self

    # 部分标注形式没有对应图像尺寸
    # 添加图像尺寸标注
    def add_img_size_anno(self):
        json_name = ensure_extend(self.json_name, 'json')
        json_pth = os.path.join(self.root, self.json_folder, json_name)
        json_dct = load_json(json_pth)
        img_annos = json_dct['images']
        print('Add image size anno ', json_pth)
        for i, img_ann in MEnumerate(img_annos):
            img_pth = os.path.join(self.root, self.img_folder, img_ann['file_name'])
            img = Image.open(img_pth)
            width, height = img.size
            img_ann['width'] = width
            img_ann['height'] = height
        save_json(json_pth, json_dct)
        return None

    def __repr__(self):
        num_dict = Counter(self.names)
        msg = '\n'.join(['%-30s ' % name + ' %-6d' % num for name, num in num_dict.items()])
        return msg


class COCODetectionDataSet(COCODataset):
    def __init__(self, root, set_name, json_name, img_folder='images', json_folder='annotation',
                 id2name=None, cls_names=None, border_type=None, num_oversamp: int = 1, data_mode=DATA_MODE.FULL,
                 **kwargs):
        self.border_type = border_type
        COCODataset.__init__(self, root, set_name, json_name=json_name, img_folder=img_folder, json_folder=json_folder,
                             id2name=id2name, cls_names=cls_names, num_oversamp=num_oversamp, data_mode=data_mode,
                             **kwargs)

    @property
    def labels(self):
        labels = []
        for img_ann, obj_anns in zip(self.img_annos, self.obj_annoss):
            boxes = COCODetectionDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_anns, num_cls=self.num_cls, id2name=self.id2name,
                name2cind=self.name2cind, border_type=self.border_type)
            labels.append(boxes)
        return labels

    @staticmethod
    def ann2cate_name(obj_ann, id2name=None, name2cind=None, num_cls=1):
        cate_id = int(obj_ann.get('category_id', 0))
        name = obj_ann.get('category_name', str(cate_id)) if id2name is None else id2name(cate_id)
        cind = cate_id if name2cind is None else name2cind(name)
        category = IndexCategory(cindN=cind, num_cls=num_cls, confN=1)
        return category, name

    @staticmethod
    def ann2border(obj_ann, img_size, border_type=None):
        xyxy = np.array(obj_ann['bbox'])
        xyxy[2:4] += xyxy[:2]
        border = XYXYBorder(xyxyN=xyxy, size=img_size)
        if border_type is not None:
            border = border_type.convert(border)
        return border

    @staticmethod
    def prase_anns(img_ann, obj_anns, num_cls, id2name, name2cind, border_type=None):
        meta = img_ann['file_name'].split('.')[0]
        img_size = (img_ann.get('width', 0), img_ann.get('height', 0))
        boxes = BoxesLabel(meta=meta, img_size=img_size)
        for obj_ann in obj_anns:
            border = COCODetectionDataSet.ann2border(obj_ann, img_size=img_size, border_type=border_type)
            category, name = COCODetectionDataSet.ann2cate_name(obj_ann, id2name, name2cind, num_cls)
            boxes.append(BoxItem(border=border, category=category, name=name))
        return boxes

    def _index2label(self, index: int):
        img_ann = self.img_annos[index]
        boxes = COCODetectionDataSet.prase_anns(
            img_ann=img_ann, obj_anns=self.obj_annoss[index], num_cls=self.num_cls,
            id2name=self.id2name, name2cind=self.name2cind, border_type=self.border_type)
        return boxes

    def _index2data(self, index: int):
        img_ann = self.img_annos[index]
        img_pth = os.path.join(self.img_dir, img_ann['file_name'])
        img = load_img_pil(img_pth)
        img_ann.update(dict(file_name=os.path.basename(img_pth), width=img.size[0], height=img.size[1]))
        boxes = COCODetectionDataSet.prase_anns(
            img_ann=img_ann, obj_anns=self.obj_annoss[index], num_cls=self.num_cls,
            id2name=self.id2name, name2cind=self.name2cind, border_type=self.border_type)
        return img, boxes


class COCOSegmentationDataSet(COCODataset):

    def __init__(self, root, set_name, json_name, img_folder='images', json_folder='annotation',
                 id2name=None, cls_names=None, rgn_type=None, num_oversamp: int = 1, data_mode=DATA_MODE.FULL,
                 **kwargs):
        self.rgn_type = rgn_type
        super().__init__(root, set_name, json_name=json_name, img_folder=img_folder, json_folder=json_folder,
                         id2name=id2name, cls_names=cls_names, num_oversamp=num_oversamp, data_mode=data_mode, **kwargs)

    @staticmethod
    def ann2rgn(obj_ann, img_size, rgn_type=None):
        segmentation = obj_ann['segmentation']
        xyxy = np.array(obj_ann['bbox'])
        xyxy[2:4] += xyxy[:2]
        if isinstance(segmentation, list):
            if len(segmentation) == 1:
                xypN = np.array(segmentation[0]).reshape(-1, 2)
                rgn = XYPBorder(xypN, img_size)
            else:
                xypNs = [np.array(xypN).reshape(-1, 2) for xypN in segmentation]
                maskNb = xypNs2maskNb(xypNs, size=img_size)
                rgn = RefValRegion.from_maskNb_xyxyN(maskNb, xyxy)
        else:
            # sp0, sp1 = segmentation['size']
            # rle = mask_utils.frPyObjects(segmentation, sp0, sp1)
            maskN = np.array(mask_utils.decode(segmentation), dtype=bool)
            rgn = RefValRegion.from_maskNb_xyxyN(maskN, xyxy)
        if rgn_type is not None: rgn = rgn_type.convert(rgn)
        return rgn

    @staticmethod
    def prase_anns(img_ann, obj_anns, num_cls, cind2name_remapper, name2cind, rgn_type=None):
        meta = img_ann['file_name'].split('.')[0]
        img_size = (img_ann['width'], img_ann['height'])
        segs = SegsLabel(meta=meta, img_size=img_size)
        for obj_ann in obj_anns:
            category, name = COCODetectionDataSet.ann2cate_name(obj_ann, cind2name_remapper, name2cind, num_cls)
            rgn = COCOSegmentationDataSet.ann2rgn(obj_ann, img_size, rgn_type=rgn_type)
            inserted = False
            for seg in segs:
                if seg.category._cindN == category._cindN:
                    seg.rgn = AbsBoolRegion(maskNb_abs=seg.rgn.maskNb + rgn.maskNb)
                    inserted = True
                    break
            if not inserted:
                segs.append(SegItem(rgn=rgn, category=category, name=name))
        if rgn_type is not None: segs.as_rgn_type(rgn_type)
        return segs

    @property
    def labels(self):
        labels = []
        for img_ann, obj_anns in zip(self.img_annos, self.obj_annoss):
            boxes = COCOSegmentationDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_anns, num_cls=self.num_cls,
                cind2name_remapper=self.id2name, name2cind=self.name2cind, rgn_type=self.rgn_type)
            labels.append(boxes)
        return labels

    def _index2label(self, index: int):
        img_ann = self.img_annos[index]
        boxes = COCOSegmentationDataSet.prase_anns(
            img_ann=img_ann, obj_anns=self.obj_annoss[index], num_cls=self.num_cls,
            cind2name_remapper=self.id2name, name2cind=self.name2cind, rgn_type=self.rgn_type)
        return boxes

    def _index2data(self, index):
        img_ann = self.img_annos[index]
        img_pth = os.path.join(self.img_dir, img_ann['file_name'])
        img = load_img_pil(img_pth)
        img_ann.update(dict(file_name=os.path.basename(img_pth), width=img.size[0], height=img.size[1]))
        boxes = COCOSegmentationDataSet.prase_anns(
            img_ann=img_ann, obj_anns=self.obj_annoss[index], num_cls=self.num_cls,
            cind2name_remapper=self.id2name, name2cind=self.name2cind, rgn_type=self.rgn_type)
        return img, boxes


class COCOInstanceDataSet(COCODataset):
    def __init__(self, root, set_name, json_name, img_folder='images', json_folder='annotation',
                 id2name=None, cls_names=None, rgn_type=None, border_type=None, num_oversamp: int = 1,
                 data_mode=DATA_MODE.FULL, **kwargs):
        self.rgn_type = rgn_type
        self.border_type = border_type
        super().__init__(root, set_name, json_name=json_name, img_folder=img_folder, json_folder=json_folder,
                         id2name=id2name, cls_names=cls_names, num_oversamp=num_oversamp, data_mode=data_mode, **kwargs)

    @staticmethod
    def prase_anns(img_ann, obj_anns, num_cls=1, id2name=None, name2cind=None, rgn_type=None, border_type=None):
        meta = img_ann['file_name'].split('.')[0]
        img_size = (img_ann.get('width', 0), img_ann.get('height', 0))
        insts = InstsLabel(meta=meta, img_size=img_size)
        for obj_ann in obj_anns:
            border = COCODetectionDataSet.ann2border(obj_ann, img_size=img_size, border_type=border_type)
            category, name = COCODetectionDataSet.ann2cate_name(obj_ann, id2name, name2cind, num_cls)
            rgn = COCOSegmentationDataSet.ann2rgn(obj_ann, img_size, rgn_type=rgn_type)
            inst = InstItem(border=border, rgn=rgn, category=category, name=name)
            insts.append(inst)
        return insts

    def labels(self):
        labels = []
        for img_ann, obj_anns in zip(self.img_annos, self.obj_annoss):
            insts = COCOInstanceDataSet.prase_anns(
                img_ann=img_ann, obj_anns=obj_anns, num_cls=self.num_cls, id2name=self.id2name,
                name2cind=self.name2cind, rgn_type=self.rgn_type, border_type=self.border_type)
            labels.append(insts)
        return labels

    def _index2label(self, index: int):
        img_ann = self.img_annos[index]
        insts = COCOInstanceDataSet.prase_anns(
            img_ann=img_ann, obj_anns=self.obj_annoss[index], num_cls=self.num_cls,
            id2name=self.id2name, name2cind=self.name2cind, rgn_type=self.rgn_type,
            border_type=self.border_type)
        return insts

    def _index2data(self, index):
        img_ann = self.img_annos[index]
        img_pth = os.path.join(self.img_dir, img_ann['file_name'])
        img = load_img_pil(img_pth)
        img_ann.update(dict(file_name=os.path.basename(img_pth), width=img.size[0], height=img.size[1]))
        insts = COCOInstanceDataSet.prase_anns(
            img_ann=img_ann, obj_anns=self.obj_annoss[index], num_cls=self.num_cls,
            id2name=self.id2name, name2cind=self.name2cind, rgn_type=self.rgn_type,
            border_type=self.border_type)
        return img, insts


# </editor-fold>

class COCOCommon(MDataSource):
    CLS_NAMES = ('object',)
    CIND2NAME_REMAPPER = None
    IMG_FOLDER = 'images'
    JSON_NAME = PLACEHOLDER.SET_NAME
    JSON_FOLDER = ''
    IMG_EXTEND = 'jpg'

    REGISTER_BUILDER = {
        TASK_TYPE.DETECTION: COCODetectionDataSet,
        TASK_TYPE.SEGMENTATION: COCOSegmentationDataSet,
        TASK_TYPE.INSTANCESEG: COCOInstanceDataSet,
    }

    SET_NAMES = ('train', 'test', 'val')

    def __init__(self, root=None, img_folder=IMG_FOLDER, json_name=JSON_NAME, json_folder=JSON_FOLDER,
                 cind2name_remapper=CIND2NAME_REMAPPER, task_type=TASK_TYPE.DETECTION, data_mode=DATA_MODE.FULL,
                 cls_names=CLS_NAMES, set_names=SET_NAMES, **kwargs):
        MDataSource.__init__(self, root=root, set_names=set_names, task_type=task_type)
        self.img_folder = img_folder
        self.json_folder = json_folder
        self.json_name = json_name
        self.cind2name_remapper = cind2name_remapper
        self.kwargs = kwargs
        self.cls_names = cls_names
        self.data_mode = data_mode

    def _dataset(self, set_name, **kwargs):
        kwargs_update = dict(root=self.root, json_folder=self.json_folder, json_name=self.json_name,
                             set_name=set_name, img_folder=self.img_folder, data_mode=self.data_mode,
                             task_type=self.task_type, cind2name_remapper=self.cind2name_remapper,
                             cls_names=self.cls_names)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        kwargs_update['img_folder'] = format_set_folder(set_name, formatter=kwargs_update['img_folder'])
        kwargs_update['json_folder'] = format_set_folder(set_name, formatter=kwargs_update['json_folder'])
        kwargs_update['json_name'] = format_set_folder(set_name, formatter=kwargs_update['json_name'])
        builder = COCO.REGISTER_BUILDER[kwargs_update.get('task_type')]
        dataset = builder(**kwargs_update)
        return dataset


class COCO(COCOCommon):
    ID2NAME_DICT = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
        40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
        44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
        51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
        70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
        82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
        88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    NEME2ID_DICT = dict([val, key] for key, val in ID2NAME_DICT.items())
    CLS_NAMES = tuple(ID2NAME_DICT.values())
    IMG_FOLDER = 'images_' + PLACEHOLDER.SET_NAME
    JSON_NAME = 'instances_' + PLACEHOLDER.SET_NAME
    JSON_FOLDER = 'annotations'
    IMG_EXTEND = 'jpg'

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//COCO//',
        PLATFORM_DESTOPLAB: 'D://Datasets//COCO//',
        PLATFORM_SEV3090: '/home/data-storage/COCO',
        PLATFORM_SEV4090: '/home/data-storage/COCO',
        PLATFORM_SEVTAITAN: '//home//exspace//dataset//COCO',
        PLATFORM_BOARD: ''
    }

    REGISTER_BUILDER = {
        TASK_TYPE.DETECTION: COCODetectionDataSet,
        TASK_TYPE.SEGMENTATION: COCOSegmentationDataSet,
        TASK_TYPE.INSTANCESEG: COCOInstanceDataSet,
    }

    SET_NAMES = ('train', 'test', 'val')

    def __init__(self, root=None, img_folder=IMG_FOLDER, json_name=JSON_NAME, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, data_mode=DATA_MODE.FULL,
                 cls_names=CLS_NAMES, set_names=SET_NAMES, **kwargs):
        COCOCommon.__init__(self, root=root, img_folder=img_folder, json_name=json_name, json_folder=json_folder,
                            task_type=task_type, cls_names=cls_names, set_names=set_names, data_mode=data_mode,
                            **kwargs)


if __name__ == '__main__':
    # maskN = np.zeros((10, 10))
    maskN = np.random.rand(1000, 1000)
    maskN = np.where(maskN > 0.7, np.ones_like(maskN), np.zeros_like(maskN))
    time1 = time.time()
    for i in range(100):
        rle = COCOWriter.binary_mask2rle(maskN)
        # rle2 = binary_mask2rle(maskN)

        # print(rle['counts'])
        # print(rle3['counts'])
        # print(np.sum(np.abs(np.array(rle['counts']) - np.array(rle3['counts']))))
    time2 = time.time()
    print(time2 - time1)
