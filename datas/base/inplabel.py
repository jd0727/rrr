import glob
import shutil
from abc import abstractmethod
from typing import Sequence, Optional, Dict, Callable

from utils import ImageLabel, load_pkl, save_pkl, ensure_folder_pth, pd, CategoryLabel, listdir_recursive, \
    IndexCategory, ensure_file_dir
from .define import MNameMapper, MDataset, ensure_extend, load_txt, load_img_cv2, load_json, json_dct2obj, MDataSource, \
    copy, LabelWriter, obj2json_dct, save_json, save_txt, load_img_size, TASK_TYPE, DATA_MODE, BROADCAST, \
    ImageItemsLabel
import os
import numpy as np

# <editor-fold desc='辅助函数'>
from .iotools import _analysis_sizes


def partition_set(set_dir: str, metas: Sequence[str], split_dict: Dict,
                  set_extend: str = 'txt', brocdcast: Callable = BROADCAST):
    set_names = list(split_dict.keys())
    ratios = list(split_dict.values())
    ratios_cum = np.cumsum(ratios)
    ensure_folder_pth(set_dir)
    np.random.shuffle(metas)
    last_ptr = 0
    for set_name, ratio, ratio_cum in zip(set_names, ratios, ratios_cum):
        cur_ptr = int(np.ceil(len(metas) * ratio_cum))
        metas_set = sorted(metas[last_ptr:cur_ptr])
        last_ptr = cur_ptr
        set_pth = os.path.join(set_dir, ensure_extend(set_name, set_extend))
        save_txt(set_pth, lines=metas_set)
        brocdcast('Split completed ' + set_name + ' : %d datas' % len(metas_set))
    return True


def merge_set(set_dir: str, set_names: Sequence[str], new_name: str,
              set_extend: str = 'txt', brocdcast: Callable = BROADCAST):
    metas = []
    for set_name in set_names:
        set_pth = os.path.join(set_dir, ensure_extend(set_name, set_extend))
        metas += load_txt(set_pth, extend='txt')
    new_set_pth = os.path.join(set_dir, ensure_extend(new_name, set_extend))
    save_txt(new_set_pth, lines=metas)
    brocdcast('Merge completed [ ' + ' '.join(set_names) + ' ] -> [ ' + new_name + ' ] with %d datas' % len(metas))
    return True


def collect_cls_names(labels):
    name_mapper = {}
    num_cls = 0
    for label in labels:
        if isinstance(label, CategoryLabel):
            num_cls = max(num_cls, label.category.num_cls)
            if 'name' in label.keys():
                name_mapper[label.category.cind] = label['name']
        elif isinstance(label, ImageItemsLabel):
            for item in label:
                num_cls = max(num_cls, item.category.num_cls)
                if 'name' in item.keys():
                    name_mapper[item.category.cind] = item['name']
    cls_names = []
    for cind in range(num_cls):
        cls_names.append(name_mapper.get(cind, 'category_%d' % cind))
    return cls_names


def build_cls_labels(set_dir: str, name2cind: Optional[Callable] = None, num_cls: int = 1, extend: str = 'jpg'):
    if name2cind is None:
        cls_names = [dn for dn in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, dn))]
        cls_names = list(sorted(cls_names))
        mapper = MNameMapper(cls_names)
        name2cind = mapper.name2cind
        num_cls = mapper.num_cls

    labels = []
    img_pths = listdir_recursive(set_dir, extends=extend)
    for img_pth in img_pths:
        basename = os.path.basename(img_pth)
        cls_name = os.path.relpath(img_pth, set_dir).split(os.path.sep)[0]
        meta = os.path.splitext(basename)[0]
        cind = name2cind(cls_name)
        img_size = load_img_size(img_pth)
        category = IndexCategory(cindN=cind, num_cls=num_cls)
        label = CategoryLabel(meta=meta, img_size=img_size, name=cls_name, category=category)
        labels.append(label)
    return labels


def apply_cls_labels(set_dir: str, labels: Sequence[CategoryLabel],
                     cind2name: Optional[Callable] = None, extend: str = 'jpg'):
    img_pths = listdir_recursive(set_dir, extends=extend)
    label_mapper = {}
    for label in labels:
        label_mapper[label.meta] = label

    for img_pth in img_pths:
        basename = os.path.basename(img_pth)
        meta = os.path.splitext(basename)[0]
        if meta not in label_mapper.keys():
            continue
        label = label_mapper[meta]
        cind = label.category.cind
        folder_name = cind2name(cind) if cind2name is not None else None
        folder_name = label['name'] if folder_name is None and 'name' in label else folder_name
        if folder_name is None:
            continue
        img_pth_dst = os.path.join(set_dir, folder_name, os.path.basename(img_pth))
        if not img_pth == img_pth_dst:
            shutil.move(img_pth, img_pth_dst)
        # print(img_pth, img_pth_dst)
    return labels


# </editor-fold>

# <editor-fold desc='独立标注数据集'>
class InpDataset(MNameMapper, MDataset):
    LABEL_EXTEND = 'none'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'

    def __init__(self, root: str, set_name: str, cls_names, img_folder: str = IMG_FOLDER, data_mode=DATA_MODE.FULL,
                 label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND, **kwargs):
        MDataset.__init__(self, root=root, set_name=set_name, num_oversamp=1, data_mode=data_mode)
        self._img_folder = img_folder
        self._label_folder = label_folder
        self.label_extend = label_extend
        self.img_extend = img_extend

        # img_pths_all = listdir_recursive(self.img_dir, extends=self.img_extend)
        # label_pths_all = listdir_recursive(self.label_dir, extends=self.label_extend)

        set_pth = os.path.join(root, ensure_extend(set_name, 'txt'))
        if os.path.exists(set_pth):
            metas = load_txt(set_pth)
            metas = [m for m in metas if len(m) > 0]
        elif set_name == 'all':
            img_pths_all = listdir_recursive(self.img_dir, extends=self.img_extend)
            metas = [os.path.basename(os.path.splitext(ip)[0]) for ip in img_pths_all]
        else:
            raise Exception(set_name + ' not exist')

        self._metas = metas
        self._img_pths = [os.path.join(self.img_dir, ensure_extend(meta, self.IMG_EXTEND)) for meta in metas]
        self._label_pths = [os.path.join(self.label_dir, ensure_extend(meta, self.LABEL_EXTEND)) for meta in metas]
        # _dct_label = dict((os.path.basename(os.path.splitext(lp)[0]), lp) for lp in label_pths_all)
        # self._label_pths = [_dct_label[meta] for meta in metas]
        # _dct_img = dict((os.path.basename(os.path.splitext(ip)[0]), ip) for ip in img_pths_all)
        # self._img_pths = [_dct_img[meta] for meta in metas]

        if cls_names is None:
            cls_names = []
        MNameMapper.__init__(self, cls_names)

    def clear_labels_(self):
        for meta in self.metas:
            label_pth = os.path.join(self.label_dir, ensure_extend(meta, self.label_extend))
            if os.path.exists(label_pth):
                os.remove(label_pth)

    def partition_set_(self, split_dict):
        partition_set(set_dir=self.root, metas=self._metas, split_dict=split_dict)
        return self

    def merge_set_(self, set_names, new_name):
        merge_set(set_dir=self.root, set_names=set_names, new_name=new_name)
        return self

    @property
    def img_dir(self) -> str:
        return os.path.join(self._root, self._img_folder)

    @property
    def label_dir(self) -> str:
        return os.path.join(self._root, self._label_folder)

    @property
    def root(self) -> str:
        return self._root

    @property
    def set_name(self) -> str:
        return self._set_name

    @property
    def img_sizes(self):
        return np.array([label.img_size for label in self.labels])

    @property
    def labels(self) -> Sequence[ImageLabel]:
        labels = [self._index2label(index) for index in range(len(self))]
        return labels

    @property
    def metas(self) -> Sequence[str]:
        return self._metas

    def _index2data(self, index: int):
        label = self._index2label(index)
        img = self._index2img(index)
        return img, label

    def _index2img(self, index: int):
        img_pth = self._img_pths[index]
        return load_img_cv2(img_pth)

    def _meta2label(self, meta: str):
        return self._index2label(self._metas.index(meta))

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _meta2data(self, meta: str):
        label = self._meta2label(meta)
        img = self._meta2img(meta)
        return img, label

    def __len__(self) -> int:
        return len(self._metas)


class InpDataSource(MDataSource):
    REGISTER_ROOT = {}
    LABEL_EXTEND = 'none'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'

    REGISTER_BUILDER = {}

    def __init__(self, root=None, img_folder: str = IMG_FOLDER,
                 label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND,
                 cls_names=None, set_names=None, task_type=TASK_TYPE.AUTO, **kwargs):
        MDataSource.__init__(self, root=root, set_names=set_names)
        self._img_folder = img_folder
        self._label_folder = label_folder
        self.img_extend = img_extend
        self.label_extend = label_extend
        self.kwargs = kwargs
        self.cls_names = cls_names
        self.task_type = task_type

    @property
    def img_folder(self) -> str:
        return self._img_folder

    @property
    def label_folder(self) -> str:
        return self._label_folder

    @property
    def img_dir(self) -> str:
        return os.path.join(self.root, self._img_folder)

    @property
    def label_dir(self) -> str:
        return os.path.join(self.root, self._label_folder)

    def _dataset(self, set_name, task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        kwargs_update = dict(img_folder=self.img_folder, label_folder=self.label_folder, img_extend=self.img_extend,
                             label_extend=self.label_extend, cls_names=self.cls_names, root=self.root)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = self.REGISTER_BUILDER[task_type](set_name=set_name, **kwargs_update)
        return dataset


# </editor-fold>

# <editor-fold desc='单一标注数据集'>


class SingleDataset(MNameMapper, MDataset):
    LABEL_EXTEND = 'pkl'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_NAME = 'label'

    def __init__(self, root: str, set_name: str, cls_names, img_folder: str = IMG_FOLDER,
                 label_name: str = LABEL_NAME, data_mode=DATA_MODE.FULL, img_extend=IMG_EXTEND,
                 label_extend=LABEL_EXTEND, **kwargs):
        MDataset.__init__(self, root=root, set_name=set_name, num_oversamp=1, data_mode=data_mode)

        self._img_folder = img_folder
        self._label_name = label_name
        self.img_extend = img_extend
        self.label_extend = label_extend

        img_pths_all = listdir_recursive(self.img_dir, extends=self.img_extend)
        if os.path.exists(self.label_pth):
            labels_all = self._load_labels(self.label_pth)
            _dct_label = dict((lb.meta, lb) for lb in labels_all)
        else:
            labels_all = None
            _dct_label = None

        set_pth = os.path.join(root, set_name + '.txt')
        if os.path.exists(set_pth):
            metas = load_txt(set_pth)
            metas = [m for m in metas if len(m) > 0]
        elif set_name == 'all':
            metas = [os.path.basename(os.path.splitext(ip)[0]) for ip in img_pths_all]
        else:
            raise Exception(set_name + ' not exist')

        self._metas = metas
        _dct_img = dict((os.path.basename(os.path.splitext(ip)[0]), ip) for ip in img_pths_all)
        self._img_pths = [_dct_img[meta] for meta in metas]

        labels = []
        for i, meta in enumerate(metas):
            if _dct_label is not None and meta in _dct_label.keys():
                labels.append(_dct_label[meta])
            else:
                img_size = load_img_size(self._img_pths[i])
                labels.append(ImageLabel(meta=meta, img_size=img_size))
        self._labels = labels

        if cls_names is None:
            if labels_all is not None:
                cls_names = collect_cls_names(labels_all)
            else:
                cls_names = []

        MNameMapper.__init__(self, cls_names=cls_names)

    def partition_set_(self, split_dict):
        partition_set(set_dir=self.root, metas=self._metas, split_dict=split_dict)
        return self

    @abstractmethod
    def _load_labels(self, labels_pth):
        pass

    @property
    def metas(self) -> Sequence[str]:
        return self._metas

    def __len__(self):
        return len(self._metas)

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def img_dir(self):
        return os.path.join(self._root, self._img_folder)

    @property
    def img_pths(self):
        return self._img_pths

    @property
    def label_pth(self):
        return os.path.join(self._root, ensure_extend(self._label_name, self.label_extend))

    @property
    def label_name(self):
        return self._label_name

    @property
    def labels(self):
        return self._labels

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _index2img(self, index: int):
        img_pth = self._img_pths[index]
        return load_img_cv2(img_pth)

    def _meta2label(self, meta: str):
        return self._labels[self._metas.index(meta)] if self._labels is not None else None

    def _index2label(self, index: int):
        return self._labels[index] if self._labels is not None else None

    def _meta2data(self, meta):
        return self._index2data(self._metas.index(meta))

    def _index2data(self, index):
        img = load_img_cv2(self._img_pths[index])
        label = self._labels[index] if self._labels is not None else None
        return img, label


class SingleDataSource(MDataSource):
    REGISTER_BUILDER = {
        TASK_TYPE.AUTO: SingleDataset
    }
    LABEL_EXTEND = 'pkl'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_NAME = 'labels'

    def __init__(self, root=None, img_folder: str = IMG_FOLDER,
                 label_name: str = LABEL_NAME, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND,
                 cls_names=None, set_names=None, task_type=TASK_TYPE.AUTO, data_mode=DATA_MODE.FULL, **kwargs):
        MDataSource.__init__(self, root=root, set_names=set_names)
        self._img_folder = img_folder
        self._label_name = label_name
        self.img_extend = img_extend
        self.label_extend = label_extend
        self.kwargs = kwargs
        self.cls_names = cls_names
        self.task_type = task_type
        self.data_mode = data_mode

    @property
    def img_folder(self) -> str:
        return self._img_folder

    @property
    def label_name(self) -> str:
        return self._label_name

    @property
    def label_pth(self) -> str:
        return os.path.join(self.root, ensure_extend(self.label_name, self.label_extend))

    @property
    def img_dir(self) -> str:
        return os.path.join(self.root, self._img_folder)

    def _dataset(self, set_name, task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        kwargs_update = dict(img_folder=self.img_folder, label_name=self._label_name, img_extend=self.img_extend,
                             label_extend=self.label_extend, cls_names=self.cls_names, root=self.root,
                             data_mode=self.data_mode)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = self.REGISTER_BUILDER[task_type](set_name=set_name, **kwargs_update)
        return dataset


# </editor-fold>

# <editor-fold desc='PKL数据集'>

class SinglePKLWriter(LabelWriter):
    def __init__(self, label_pth: str, set_pth: Optional[str] = None):
        self.set_pth = set_pth
        self.label_pth = label_pth
        self.labels = []

    def save_label(self, label) -> object:
        self.labels.append(label)
        return label.meta

    def save_all(self, metas: list):
        ensure_file_dir(self.label_pth)
        save_pkl(self.label_pth, self.labels)
        if self.set_pth is not None:
            save_txt(self.set_pth, metas)


class SinglePKLDataset(SingleDataset):
    LABEL_EXTEND = 'pkl'

    def _load_labels(self, labels_pth):
        return load_pkl(labels_pth)

    def build_cls_labels(self, name2cind: Optional[Callable] = None, num_cls: int = 1, ):
        if name2cind == None:
            name2cind = self.name2cind
            num_cls = self.num_cls
        labels = build_cls_labels(
            set_dir=self.img_dir, name2cind=name2cind,
            num_cls=num_cls, extend=self.img_extend)
        save_pkl(self.label_pth, labels)
        return self

    def apply_cls_labels(self, cind2name: Optional[Callable] = None, ):
        if cind2name == None:
            cind2name = self.cind2name
        apply_cls_labels(
            set_dir=self.img_dir, labels=self.labels,
            cind2name=cind2name, extend=self.img_extend)
        return self


class SinglePKLDataSource(SingleDataSource):
    REGISTER_BUILDER = {
        TASK_TYPE.AUTO: SinglePKLDataset
    }
    LABEL_EXTEND = 'json'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'


class InpPKLWriter(LabelWriter):
    def __init__(self, label_dir: str, set_pth: Optional[str] = None, label_extend: str = 'pkl', ):
        self.set_pth = set_pth
        self.label_dir = label_dir
        self.label_extend = label_extend

    def save_label(self, label) -> object:
        label_pth = os.path.join(self.label_dir, ensure_extend(label.meta, self.label_extend, ))
        ensure_folder_pth(self.label_dir)
        save_pkl(label_pth, label)
        return label.meta

    def save_all(self, metas: list):
        if self.set_pth is not None:
            save_txt(self.set_pth, metas)


class InpPKLDataset(InpDataset):
    LABEL_EXTEND = 'pkl'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'

    def __init__(self, root: str, set_name: str, cls_names, img_folder: str = IMG_FOLDER, data_mode=DATA_MODE.FULL,
                 label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND, **kwargs):
        InpDataset.__init__(self, root=root, set_name=set_name, cls_names=cls_names,
                            img_folder=img_folder, label_folder=label_folder, img_extend=img_extend,
                            label_extend=label_extend, data_mode=data_mode)

    def delete(self):
        for meta in self.metas:
            label_pth = os.path.join(self.label_dir, ensure_extend(meta, self.label_extend))
            img_pth = os.path.join(self.img_dir, ensure_extend(meta, self.img_extend))
            if os.path.exists(label_pth):
                os.remove(label_pth)
            if os.path.exists(img_pth):
                os.remove(img_pth)
        return self

    def stat(self) -> pd.DataFrame:
        labels = self.labels
        xyxys = [np.zeros(shape=(0, 4))]
        names = []
        img_sizes = []
        for label in labels:
            img_sizes.append(label.img_size)
            if isinstance(label, ImageItemsLabel):
                for item in label:
                    names.append(item.get('name', 'object'))
                    xyxys.append(item.xyxyN[None])
        xyxys = np.concatenate(xyxys, axis=0)
        names = np.array(names)
        img_sizes = np.array(img_sizes)
        sizes = xyxys[:, 2:] - xyxys[:, :2]
        report = pd.DataFrame(dict(name='image', **_analysis_sizes(img_sizes)), index=[0])
        names_u = np.unique(names)
        for i, name_u in enumerate(sorted(names_u)):
            report = pd.concat(
                [report, pd.DataFrame(dict(name=name_u, **_analysis_sizes(sizes[name_u == names])), index=[0])])
        return report

    def _index2label(self, index):
        label_pth = self._label_pths[index]
        img_pth = self._img_pths[index]
        if os.path.exists(label_pth):
            label = load_pkl(label_pth)
        else:
            meta = os.path.splitext(os.path.basename(img_pth))[0]
            return ImageLabel(meta=meta, img_size=load_img_size(img_pth))
        return label


class InpPKLDataSource(InpDataSource):
    REGISTER_BUILDER = {
        TASK_TYPE.AUTO: InpPKLDataset
    }
    LABEL_EXTEND = 'pkl'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'

    def __init__(self, root=None, img_folder: str = IMG_FOLDER,
                 label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND,
                 cls_names=None, set_names=None, task_type=TASK_TYPE.AUTO, data_mode=DATA_MODE.FULL, **kwargs):
        InpDataSource.__init__(
            self, root=root, img_folder=img_folder, label_folder=label_folder, data_mode=data_mode,
            img_extend=img_extend, label_extend=label_extend, cls_names=cls_names, set_names=set_names,
            task_type=task_type, **kwargs)


# </editor-fold>

# <editor-fold desc='JSON数据集'>
class SingleJSONDataset(SingleDataset):
    LABEL_EXTEND = 'json'

    def _load_labels(self, labels_pth):
        label_dct = load_json(labels_pth)
        labels = json_dct2obj(label_dct)
        return labels

    def build_cls_label(self, name2cind: Optional[Callable] = None, num_cls: int = 1, ):
        if name2cind == None:
            name2cind = self.name2cind
            num_cls = self.num_cls
        labels = build_cls_labels(
            set_dir=self.img_dir, name2cind=name2cind,
            num_cls=num_cls, extend=self.img_extend)
        json_dct = obj2json_dct(labels)
        save_json(self.label_pth, dct=json_dct, indent=4)
        return self


class SingleJSONDataSource(SingleDataSource):
    REGISTER_BUILDER = {
        TASK_TYPE.AUTO: SingleJSONDataset
    }
    LABEL_EXTEND = 'json'


class InpJSONWriter(LabelWriter):
    def __init__(self, label_dir: str, set_pth: Optional[str] = None, label_extend: str = 'json', ):
        self.set_pth = set_pth
        self.label_dir = label_dir
        self.label_extend = label_extend

    def save_label(self, label) -> object:
        label_pth = os.path.join(self.label_dir, ensure_extend(label.meta, self.label_extend, ))
        json_dct = obj2json_dct(label)
        save_json(label_pth, dct=json_dct, indent=4)
        return label.meta

    def save_all(self, metas: list):
        if self.set_pth is not None:
            save_txt(self.set_pth, metas)


class InpJSONDataset(InpDataset):
    LABEL_EXTEND = 'json'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'

    def __init__(self, root: str, set_name: str, cls_names, img_folder: str = IMG_FOLDER,
                 label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND, **kwargs):
        InpDataset.__init__(self, root=root, set_name=set_name, cls_names=cls_names,
                            img_folder=img_folder, label_folder=label_folder, img_extend=img_extend,
                            label_extend=label_extend)

    def _index2label(self, index):
        label_pth = self._label_pths[index]
        img_pth = self._img_pths[index]
        if os.path.exists(label_pth):
            label_dct = load_json(label_pth)
            label = json_dct2obj(label_dct)
        else:
            meta = os.path.splitext(os.path.basename(img_pth))[0]
            return ImageLabel(meta=meta, img_size=load_img_size(img_pth))
        return label


class InpJSONDataSource(InpDataSource):
    REGISTER_BUILDER = {
        TASK_TYPE.AUTO: InpJSONDataset
    }
    LABEL_EXTEND = 'json'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'

    def __init__(self, root=None, img_folder: str = IMG_FOLDER,
                 label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND, label_extend=LABEL_EXTEND,
                 cls_names=None, set_names=None, task_type=TASK_TYPE.AUTO, data_mode=DATA_MODE.FULL, **kwargs):
        InpDataSource.__init__(
            self, root=root, img_folder=img_folder, label_folder=label_folder, data_mode=data_mode,
            img_extend=img_extend, label_extend=label_extend, cls_names=cls_names, set_names=set_names,
            task_type=task_type, **kwargs)

# </editor-fold>
