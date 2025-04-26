from datas.base import *
from datas.voc import _load_metas
from utils import *


class YOLDDetectionWriter(LabelWriter):
    def __init__(self, anno_dir: str, set_pth: Optional[str] = None, anno_extend: str = 'txt', ):
        self.set_pth = set_pth
        self.anno_dir = anno_dir
        self.anno_extend = anno_extend

    @staticmethod
    def save_yolotxt(anno_pth: str, label: BoxesLabel):
        lines = []
        for item in label:
            xywh = XYWHBorder.convert(item.border)._xywhN
            xywh[0:4:2] /= label.img_size[0]
            xywh[1:4:2] /= label.img_size[1]
            cind = item.category.cindN
            line = str(cind) + ' ' + ' '.join(['%6.5f' % v for v in xywh])
            lines.append(line)
        save_txt(anno_pth, lines)
        return True

    def save_label(self, label) -> object:
        anno_pth = os.path.join(self.anno_dir, ensure_extend(label.meta, self.anno_extend))
        YOLDDetectionWriter.save_yolotxt(anno_pth, label)
        return label.meta

    def save_all(self, metas: list):
        if self.set_pth is not None:
            save_txt(self.set_pth, metas)


class YOLDDetectionDataset(MNameMapper,  MDataset):
    ANNO_EXTEND = 'txt'
    IMG_EXTEND = 'jpg'

    def __init__(self, root: str, set_name: str, cls_names: Tuple[str], set_folder: str = 'sets',
                 img_folder: str = 'images', anno_folder: str = 'labels',
                 anno_extend: str = ANNO_EXTEND, img_extend: str = IMG_EXTEND, **kwargs):
        self._root = root
        self._set_name = set_name if isinstance(set_name, str) else 'all'
        self._metas = _load_metas(root, set_name, set_folder=set_folder, img_folder=img_folder)
        # 加载标签
        self.anno_extend = anno_extend
        self.img_extend = img_extend
        self._img_folder = img_folder
        self._set_folder = set_folder
        self._anno_folder = anno_folder
        MNameMapper.__init__(self, cls_names)

    @staticmethod
    def load_yolotxt(anno_pth: str, meta: str, img_size: tuple, num_cls: int, cind2name: Optional[Callable] = None):
        lines = load_txt(anno_pth)
        label = BoxesLabel(meta=meta, img_size=img_size)
        for line in lines:
            pieces = line.split(' ')
            cind = int(pieces[0])
            category = IndexCategory(cindN=cind, num_cls=num_cls)
            xywh = np.array([float(v) for v in pieces[1:5]])
            xywh[0:4:2] *= img_size[0]
            xywh[1:4:2] *= img_size[1]
            border = XYWHBorder(xywhN=xywh, size=img_size)
            item = BoxItem(border=border, category=category)
            if cind2name is not None:
                item['name'] = cind2name(cind)
            label.append(item)
        return label

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def img_dir(self):
        return os.path.join(self._root, self._img_folder)

    @property
    def anno_folder(self):
        return self._anno_folder

    @property
    def anno_dir(self):
        return os.path.join(self._root, self._anno_folder)

    @property
    def set_folder(self):
        return self._set_folder

    @property
    def set_dir(self):
        return os.path.join(self._root, self._set_folder)

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def labels(self):
        return [self._meta2label(m) for m in self._metas]

    @property
    def metas(self):
        return self._metas

    def __len__(self):
        return len(self._metas)

    def _index2data(self, index: int):
        return self._meta2data(self._metas.index(index))

    def _meta2data(self, meta: str):
        img = self._meta2img(meta)
        img_size = img2size(img)
        anno_pth = os.path.join(self.anno_dir, ensure_extend(meta, self.anno_extend))
        label = YOLDDetectionDataset.load_yolotxt(anno_pth, meta=meta, img_size=img_size, num_cls=self.num_cls,
                                                  cind2name=self.cind2name)
        return img, label

    def _meta2img(self, meta: str):
        img_pth = os.path.join(self.img_dir, ensure_extend(meta, self.img_extend))
        return load_img_cv2(img_pth)

    def _meta2label(self, meta: str):
        img_pth = os.path.join(self.img_dir, ensure_extend(meta, self.img_extend))
        img_size = imagesize.get(img_pth)
        anno_pth = os.path.join(self.anno_dir, ensure_extend(meta, self.anno_extend))
        label = YOLDDetectionDataset.load_yolotxt(anno_pth, meta=meta, img_size=img_size, num_cls=self.num_cls,
                                                  cind2name=self.cind2name)
        return label

    def dump(self, labels, anno_folder='labels', anno_extend='xml',
             prefix='Create', broadcast=BROADCAST):
        folders = [anno_folder]
        anno_dir, = ensure_folders(self.root, folders)
        broadcast(dsmsgfmtr_create(self.root, self.set_name, folders, prefix=prefix))
        for i, label in MEnumerate(labels, broadcast=broadcast):
            anno_pth = os.path.join(anno_dir, ensure_extend(label.meta, anno_extend))
            YOLDDetectionDataset.save_yolotxt(anno_pth, label=label)
        broadcast(dsmsgfmtr_end(prefix=prefix))
        return True


class YOLD(MDataSource):
    SET_FOLDER = ''
    IMG_FOLDER = 'images'
    ANNO_FOLDER = 'labels'

    REGISTER_BUILDER = {
        TASK_TYPE.DETECTION: YOLDDetectionDataset,
    }

    def __init__(self, root, cls_names, task_type=TASK_TYPE.DETECTION,
                 set_folder=SET_FOLDER,
                 img_folder=IMG_FOLDER, anno_folder=ANNO_FOLDER, set_names=None, **kwargs):
        root = self.__class__.get_root() if root is None else root
        MDataSource.__init__(self, root=root, set_names=set_names, task_type=task_type)
        self.set_folder = set_folder
        self.img_folder = img_folder
        self.anno_folder = anno_folder
        self.cls_names = cls_names
        self.kwargs = kwargs

    def _dataset(self, set_name='train', task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = self.__class__.REGISTER_BUILDER[task_type]
        kwargs_update = dict(img_folder=self.img_folder, cls_names=self.cls_names, root=self.root,
                             anno_folder=self.anno_folder, set_folder=self.set_folder, set_name=set_name)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = builder(**kwargs_update)
        return dataset
