import os

try:
    import imagesize
except Exception as e:
    pass
from .define import *

from collections import Counter
from utils.label import IndexCategory, CategoryLabel, img2imgP

# <editor-fold desc='folder编辑'>
EXTENDS_IMAGE = ['jpg', 'JPEG', 'png']


def resample_by_names(cls_names, resample):
    presv_inds = []
    for i, cls_name in enumerate(cls_names):
        if not cls_name in resample.keys():
            presv_inds.append(i)
            continue
        resamp_num = resample[cls_name]
        low = np.floor(resamp_num)
        high = np.ceil(resamp_num)
        resamp_num_rand = np.random.uniform(low=low, high=high)
        resamp_num = int(low if resamp_num_rand > resamp_num else high)
        for j in range(resamp_num):
            presv_inds.append(i)
    return presv_inds


def get_pths(root: str, cls_names=None, name_remapper=None, num_limt=-1, extends: Optional = None):
    img_pths = []
    names = []
    for folder_name in os.listdir(root):
        folder_dir = os.path.join(root, folder_name)
        if not os.path.isdir(folder_dir):
            continue
        if name_remapper is not None:
            cls_name = name_remapper[folder_name]
        else:
            cls_name = folder_name
        if cls_names is not None and cls_name not in cls_names:
            continue
        num = 0
        for img_dir, _, img_names in os.walk(folder_dir):
            for img_name in img_names:
                if extends is not None:
                    extend = os.path.splitext(img_name)[1]
                    if extend not in extends:
                        continue
                img_pths.append(os.path.join(folder_dir, img_dir, img_name))
                names.append(cls_name)
                num += 1
                if num_limt > 0 and num >= num_limt:
                    break
            if num_limt > 0 and num >= num_limt:
                break

    return img_pths, names


def get_cls_names(root, name_remapper=None):
    cls_names = []
    for dir_name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dir_name)):
            continue
        if name_remapper is not None:
            dir_name = name_remapper[dir_name]
        cls_names.append(dir_name)
    return cls_names


# </editor-fold>


class FolderClassificationDataset(MNameMapper, MDataset):
    def __init__(self, root: str, set_name: str, cls_names=None, resample=None, name_remapper=None, num_limt=-1,
                 **kwargs):
        root_set = os.path.join(root, set_name)
        cls_names = get_cls_names(root_set, name_remapper=name_remapper) if cls_names is None else cls_names
        MNameMapper.__init__(self, cls_names)
        MDataset.__init__(self, root=root, set_name=set_name)
        self._img_pths, self._names = get_pths(root_set, cls_names=cls_names, name_remapper=name_remapper,
                                               num_limt=num_limt)
        if resample is not None:
            presv_inds = resample_by_names(self._names, resample=resample)
            self._img_pths = [self._img_pths[ind] for ind in presv_inds]
            self._names = [self._names[ind] for ind in presv_inds]
        self._metas = [os.path.split(os.path.basename(img_pth))[0] for img_pth in self._img_pths]

    def __len__(self):
        return len(self._metas)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def set_name(self):
        return self._set_name

    @property
    def metas(self):
        return self._metas

    @property
    def labels(self):
        lbs = []
        for img_pth, name, meta in zip(self._img_pths, self._names, self._metas):
            label = CategoryLabel(
                category=IndexCategory(cindN=int(self.name2cind(name)), confN=1, num_cls=self.num_cls),
                img_size=imagesize.get(img_pth), meta=meta, name=name)
            lbs.append(label)
        return lbs

    def _meta2data(self, meta):
        return self._index2data(self._metas.index(meta))

    def _index2img(self, index: int):
        return load_img_cv2(self._img_pths[index])

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _meta2label(self, meta: str):
        return self._index2label(self._metas.index(meta))

    def _index2label(self, index: int):
        img_pth, name, meta = self._img_pths[index], self._names[index], self._metas[index]
        img_size = imagesize.get(img_pth)
        label = CategoryLabel(
            category=IndexCategory(cindN=int(self.name2cind(name)), confN=1, num_cls=self.num_cls),
            img_size=img_size, meta=meta, name=name)
        return label

    def _index2data(self, index):
        img_pth, name, meta = self._img_pths[index], self._names[index], self._metas[index]
        img = load_img_pil(img_pth)
        label = CategoryLabel(
            category=IndexCategory(cindN=int(self.name2cind(name)), confN=1, num_cls=self.num_cls),
            img_size=img.size, meta=meta, name=name)
        return img, label

    def __repr__(self):
        num_dict = Counter(self._names)
        msg = '\n'.join(['%10s ' % name + ' %5d' % num for name, num in num_dict.items()])
        return msg


class FolderUnlabelDataset(MNameMapper, MDataset):

    def __init__(self, root: str, set_name: str = 'unlabel', cls_names: Tuple[str] = ('obj',), **kwargs):
        MDataset.__init__(self, root=root, set_name=set_name)
        MNameMapper.__init__(self, cls_names)
        root_set = os.path.join(root, set_name)
        file_names = os.listdir(root_set)
        self._img_pths = [os.path.join(root_set, file_name) for file_name in file_names]
        self._metas = [os.path.splitext(file_name)[0] for file_name in file_names]

    def __len__(self):
        return len(self._metas)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, root):
        self._root = root

    @property
    def set_name(self):
        return self._set_name

    @property
    def metas(self):
        return self._metas

    @property
    def labels(self):
        labels = [self._meta2label(meta) for meta in self._metas]
        return labels

    def _meta2data(self, meta):
        return self._index2data(self._metas.index(meta))

    def _index2img(self, index: int):
        return load_img_cv2(self._img_pths[index])

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _meta2label(self, meta: str):
        return self._index2label(self._metas.index(meta))

    def _index2label(self, index: int):
        img_pth, meta = self._img_pths[index], self._metas[index]
        img_size = load_img_size(img_pth)
        label = ImageLabel(img_size=img_size, meta=meta)
        return label

    def _index2data(self, index):
        img_pth, meta = self._img_pths[index], self._metas[index]
        img = load_img_pil(img_pth)
        label = ImageLabel(img_size=img.size, meta=meta)
        return img, label

    def __repr__(self):
        msg = 'FolderUnlabeledDataset ' + self._root
        return msg


class FolderDataSource(MDataSource):
    REGISTER_ROOT = {}

    REGISTER_BUILDER = {
        TASK_TYPE.CLASSIFICATION: FolderClassificationDataset,
        TASK_TYPE.AUTO: FolderClassificationDataset,
    }

    def __init__(self, root=None, resample=None, cls_names=None, set_names=None, task_type=TASK_TYPE.AUTO,
                 data_mode=DATA_MODE.FULL, **kwargs):
        MDataSource.__init__(self, root=root, set_names=set_names, task_type=task_type)
        self.resample = resample
        self.kwargs = kwargs
        self.cls_names = cls_names
        self.data_mode = data_mode

    def _dataset(self, set_name, **kwargs):
        kwargs_update = dict(root=self.root, set_name=set_name, resample=self.resample,
                             cls_names=self.cls_names, task_type=self.task_type, data_mode=self.data_mode)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = self.REGISTER_BUILDER[kwargs_update.get('task_type')](**kwargs_update)
        return dataset



