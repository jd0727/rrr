try:
    import imagesize
except Exception as e:
    pass
import shutil
import threading
from utils import *


# <editor-fold desc='文件快速操作'>
def load_img_size(img_pth: str) -> Tuple:
    return imagesize.get(img_pth)


def ensure_folders(root: str, folders: Union[List, Tuple, None]) -> List:
    dirs = []
    for folder in folders:
        if folder is None:
            dirs.append(None)
        else:
            root_dir = os.path.join(root, folder)
            ensure_folder_pth(root_dir)
            dirs.append(root_dir)
    return dirs


def dsmsgfmtr_create(root, set_name, folders, prefix='Create'):
    folders = [f for f in folders if f is not None]
    msg = ' [ ' + ' | '.join(folders) + ' ] '
    return prefix + ' ' + root + ' ( ' + str(set_name) + ' ) ' + msg


def dsmsgfmtr_end(prefix='Apply'):
    return prefix + ' completed'


def dsmsgfmtr_apply(root, set_name, folders_src, folders_dst, prefix='Apply'):
    folders_dst = [f for f in folders_dst if f is not None]
    folders_src = [f for f in folders_src if f is not None]
    msg_dst = ' [ ' + ' | '.join(folders_dst) + ' ] '
    msg_src = ' [ ' + ' | '.join(folders_src) + ' ] '
    return prefix + ' ' + root + ' ( ' + str(set_name) + ' ) ' + msg_src + '->' + msg_dst


# </editor-fold>


# <editor-fold desc='数据源'>
class TASK_TYPE:
    CLASSIFICATION = 'classification'
    DETECTION = 'detection'
    SEGMENTATION = 'segmentation'
    INSTANCESEG = 'instance segmentation'
    PANOPTICSEG = 'panoptic segmentation'
    STEREODET = 'stereo detection'
    AUTO = 'auto'


class DATA_MODE:
    IMAGEONLY = 'imageonly'
    LABELONLY = 'labelonly'
    FULL = 'full'


class MDataSource(metaclass=ABCMeta):
    REGISTER_ROOT = {}

    def __init__(self, root: Optional[str] = None, set_names=('train', 'test'), task_type=TASK_TYPE.AUTO):
        self.root = self.__class__.get_root() if root is None else root
        self.set_names = set_names
        self.task_type = task_type

    @classmethod
    def registry_root(cls, root: str, platform: Optional[str] = None):
        cls.REGISTER_ROOT[PLATFORM if platform is None else platform] = root

    @classmethod
    def get_root(cls, platform: Optional[str] = None) -> str:
        return cls.REGISTER_ROOT[PLATFORM if platform is None else platform]

    @abstractmethod
    def _dataset(self, set_name, **kwargs):
        pass

    def dataset(self, set_name, **kwargs):
        if isinstance(set_name, MDataset):
            dataset = set_name
        elif isinstance(set_name, str):
            dataset = self._dataset(set_name=set_name, **kwargs)
        elif isinstance(set_name, Iterable):
            dataset = MConcatDataset([self._dataset(set_name=sub_name, **kwargs) for sub_name in set_name])
        else:
            raise Exception('err setname ' + str(set_name))
        return dataset

    def stat(self, save_pth: Optional[str] = None, set_names: Optional[Sequence[str]] = None) \
            -> Dict[str, pd.DataFrame]:
        if set_names is None:
            set_names = self.set_names
        reports = {}
        for set_name in set_names:
            dataset = self.dataset(set_name)
            reports[set_name] = dataset.stat()
        if save_pth is not None:
            save_dfdct2xlsx(save_pth, reports)
        return reports

    def loader(self, set_name, batch_size: int = 8, pin_memory: bool = False, num_workers: int = 0,
               aug_seq=None, shuffle: bool = True, drop_last: bool = False, **kwargs):
        dataset = self.dataset(set_name, **kwargs)
        loader = MDataLoader(
            dataset,
            shuffle=shuffle,
            aug_seq=aug_seq,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return loader

    def __repr__(self):
        return self.__class__.__name__ + ' ' + self.root + ' ( ' + str(self.set_names) + ' )'


# </editor-fold>

# <editor-fold desc='数据集'>


class MDataset(Iterable, Sized, IMDataset):

    def __init__(self, root: str, set_name: str, num_oversamp: int = 1, data_mode=DATA_MODE.FULL):
        self._root = root
        self._set_name = set_name
        self._num_oversamp = num_oversamp
        self.data_mode = data_mode

    @property
    @abstractmethod
    def root(self) -> str:
        pass

    @property
    @abstractmethod
    def set_name(self) -> str:
        pass

    @property
    @abstractmethod
    def labels(self) -> Sequence[ImageLabel]:
        pass

    @property
    @abstractmethod
    def metas(self) -> Sequence[str]:
        pass

    @property
    def num_oversamp(self) -> int:
        return self._num_oversamp

    # 数据出口
    def __getitem__(self, index: Union[str, int, Iterable, slice]):
        if isinstance(index, str):
            data = self.meta2data(index)
            if self.num_oversamp > 1:
                metas = np.random.choice(size=self.num_oversamp - 1, a=self.metas)
                data = [data] + [self.meta2data(m) for m in metas]
            return data
        elif isinstance(index, int):
            data = self.index2data(index)
            if self.num_oversamp > 1:
                inds = np.random.choice(size=self.num_oversamp - 1, a=len(self))
                data = [data] + [self.index2data(i) for i in inds]
            return data
        elif isinstance(index, Iterable):
            datas = [self.__getitem__(item_sub) for item_sub in index]
            if self.num_oversamp > 1:
                datas = list(chain(datas))
            return datas
        elif isinstance(index, slice):
            return self.__getitem__(range(len(self))[index])
        else:
            raise Exception('not support')

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index == self.__len__():
            raise StopIteration
        else:
            data = self.__getitem__(self._index)
            self._index = self._index + 1
            return data

    def meta2data(self, meta: str):
        if self.data_mode == DATA_MODE.FULL:
            return self._meta2data(meta)
        elif self.data_mode == DATA_MODE.IMAGEONLY:
            return self._meta2img(meta), None
        elif self.data_mode == DATA_MODE.LABELONLY:
            return None, self._meta2label(meta)
        else:
            raise Exception('data mode err')

    @abstractmethod
    def _meta2data(self, meta: str):
        pass

    def _meta2img(self, meta: str):
        return None

    def _meta2label(self, meta: str):
        return None

    def index2data(self, index: int):
        if self.data_mode == DATA_MODE.FULL:
            return self._index2data(index)
        elif self.data_mode == DATA_MODE.IMAGEONLY:
            return self._index2img(index), None
        elif self.data_mode == DATA_MODE.LABELONLY:
            return None, self._index2label(index)
        else:
            raise Exception('data mode err')

    @abstractmethod
    def _index2data(self, index: int):
        pass

    def _index2img(self, index: int):
        return None

    def _index2label(self, index: int):
        return None


class MConcatDataset(MDataset):

    def __init__(self, datasets: Union[List, Tuple]):
        MDataset.__init__(self, root=datasets[0].root, set_name=datasets[0].set_name)
        self.datasets = datasets
        self._indexs = []
        self._metas = []
        for i, dataset in enumerate(datasets):
            for k in range(len(dataset)):
                self._indexs.append((i, k))
            self._metas.extend(dataset.metas)

    def name2cind(self, name: str) -> int:
        return self.datasets[0].name2cind(name)

    def cind2name(self, cind: int) -> str:
        return self.datasets[0].cind2name(cind)

    @property
    def num_cls(self) -> int:
        return self.datasets[0]._num_cls

    @property
    def root(self):
        return self.datasets[0].root

    @property
    def set_name(self):
        return self.datasets[0].set_name

    @property
    def labels(self):
        labels = []
        for dataset in self.datasets:
            labels.extend(dataset.labels)
        return labels

    @property
    def metas(self):
        return self._metas

    def __len__(self):
        return sum(self._indexs)

    def _index2data(self, index: int):
        i, k = self._indexs[index]
        return self.datasets[i][k]

    def _meta2data(self, meta: str):
        return self._index2data(self._metas.index(meta))


class MNameMapper(HasNumClass):
    def __init__(self, cls_names: Union[List, Tuple]):
        self.cls_names = list(cls_names)

    def name2cind(self, name: str) -> int:
        if name not in self.cls_names:
            BROADCAST('Auto add class [ ' + name + ' ]')
            self.cls_names.append(name)
        return self.cls_names.index(name)

    def cind2name(self, cind: int) -> str:
        num_cls = len(self.cls_names)
        if cind >= num_cls:
            expect_len = cind + 1
            BROADCAST('Index out bound %d' % num_cls + ' -> ' + '%d' % expect_len)
            self.cls_names += ['C' + str(i) for i in range(num_cls, expect_len)]
        return self.cls_names[cind]

    @property
    def num_cls(self) -> int:
        return len(self.cls_names)

    @staticmethod
    def create_cluster_index(cls_names: Union[List, Tuple], group_names: Union[List, Tuple], offset: int = 0) -> list:
        mapper = {}
        for i, gn in enumerate(group_names):
            if isinstance(gn, str):
                mapper[gn] = i + offset
            elif isinstance(gn, (list, tuple)):
                for name in gn:
                    mapper[name] = i + offset
            else:
                raise Exception('err fmt')
        inds_clus = []
        for name in cls_names:
            if name in mapper.keys():
                inds_clus.append(mapper[name])
            else:
                inds_clus.append(len(group_names))
        return inds_clus


class MColorMapper(MNameMapper):
    def __init__(self, cls_names: Union[List, Tuple], colors):
        super(MColorMapper, self).__init__(cls_names)
        self.colors = colors

    def col2name(self, color):
        return self.cls_names[self.colors.index(color)]

    def name2col(self, name):
        return self.colors[self.cls_names.index(name)]

    def col2cind(self, color):
        return self.colors.index(color)

    def cind2col(self, cind):
        return self.colors[cind]


# </editor-fold>


# <editor-fold desc='数据加载'>

class MDataLoader(IMDataLoader):
    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size) -> NoReturn:
        self._batch_size = batch_size

    @property
    def set_name(self) -> str:
        return self.dataset.set_name

    def __init__(self, dataset: MDataset, shuffle: bool = False, num_workers: int = 0,
                 batch_size: int = 1, pin_memory: bool = False,
                 aug_seq: SizedTransform = None, drop_last=True, **kwargs):
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
        assert isinstance(dataset, MDataset)
        self.dataset = dataset
        self.aug_seq = aug_seq
        TorchDataLoader.__init__(self, dataset=dataset, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                                 batch_size=batch_size, pin_memory=pin_memory, drop_last=drop_last,
                                 collate_fn=self._collate_fn, **kwargs)

    def name2cind(self, name: str) -> int:
        return self.dataset.name2cind(name)

    def cind2name(self, cind: int) -> str:
        return self.dataset.cind2name(cind)

    @property
    def num_cls(self):
        return self.dataset.num_cls

    @property
    def num_data(self):
        return len(self.dataset)

    @property
    def img_size(self):
        if self.aug_seq is not None and isinstance(self.aug_seq, SizedTransform):
            return self.aug_seq.img_size
        else:
            return None

    @img_size.setter
    def img_size(self, img_size):
        if self.aug_seq is not None and isinstance(self.aug_seq, SizedTransform):
            self.aug_seq.img_size = img_size

    def _collate_fn(self, batch):
        imgs, labels = [], []
        for data in batch:
            if isinstance(data, list):
                img, label = zip(*data)
                labels.extend(label)
                imgs.extend(img)
            else:
                labels.append(data[1])
                imgs.append(data[0])
        if self.aug_seq is not None:
            imgs, labels = self.aug_seq.trans_datas(imgs, labels)
        return imgs, labels

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__getitem__([item])
        elif isinstance(item, list) or isinstance(item, tuple):
            batch = [self.dataset.__getitem__(index) for index in item]
            imgs, labels = self.collate_fn(batch)
            return imgs, labels
        else:
            raise Exception('index err')


# </editor-fold>

# <editor-fold desc='标签写入'>
class LabelWriter(metaclass=ABCMeta):

    @abstractmethod
    def save_label(self, label) -> object:
        pass

    @abstractmethod
    def save_all(self, caches: list):
        pass

    def save_labels(self, labels: Sequence[ImageLabel]):
        caches = []
        for i, label in enumerate(labels):
            cache = self.save_label(label)
            caches.append(cache)
        self.save_all(caches)

    def save_labels_of(self, dataset: MDataset):
        metas = dataset.metas
        caches = []
        for i, meta in enumerate(metas):
            label = dataset._meta2label(meta)
            cache = self.save_label(label)
            caches.append(cache)
        self.save_all(caches)

# </editor-fold>
