from torchvision.datasets.mnist import read_image_file, read_label_file

from .base import *


# if __name__ == '__main__':
#     d = torchvision.datasets.MNIST(root='D://Datasets//MNIST//', download=True)


class MNISTDataset(MNameMapper, MDataset):
    @property
    def metas(self):
        return self._metas

    IMG_SIZE = (28, 28)

    def __len__(self):
        return len(self._imgs)

    def __init__(self, root: str, set_name: str, cls_names):
        MNameMapper.__init__(self, cls_names)
        MDataset.__init__(self,root=root, set_name=set_name)
        prefix = 'train' if set_name == 'train' else 't10k'
        self._imgs = read_image_file(os.path.join(self._root, prefix + '-images-idx3-ubyte')).numpy()
        self._targets = read_label_file(os.path.join(self._root, prefix + '-labels-idx1-ubyte')).numpy()
        self._metas = ['mnist_' + set_name + '_' + str(i) for i in range(len(self._targets))]

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def labels(self):
        labels = []
        for index, target in enumerate(self._targets):
            labels.append(CategoryLabel(category=IndexCategory(target, num_cls=self.num_cls),
                                        img_size=MNISTDataset.IMG_SIZE, meta=str(index)))
        return labels

    def _index2data(self, index):
        img = self._imgs[index][..., None]
        label = CategoryLabel(category=IndexCategory(self._targets[index], num_cls=self.num_cls),
                              img_size=MNISTDataset.IMG_SIZE, meta=str(index))
        return img, label

    def _meta2data(self, meta):
        return self._index2data(int(meta))


class SVHNDataset(MNameMapper, MDataset):
    @property
    def metas(self):
        return self._metas

    def __len__(self):
        return self._tragets.shape[0]

    IMG_SIZE = (32, 32)

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
    def labels(self):
        lbs = []
        for index, cind in enumerate(self._tragets):
            category = IndexCategory(cindN=cind, num_cls=self.num_cls, confN=1)
            lbs.append(CategoryLabel(category=category, img_size=SVHNDataset.IMG_SIZE, meta=str(index)))
        return lbs

    def _index2data(self, index):
        img, cind = self._imgs[index], int(self._tragets[index])
        # img = np.transpose(img, (1, 2, 0))
        category = IndexCategory(cindN=cind, num_cls=self.num_cls, confN=1)
        cate = CategoryLabel(category=category, img_size=SVHNDataset.IMG_SIZE, meta=str(index))
        return img, cate

    def _meta2data(self, meta):
        return self._index2data(int(meta))

    def __init__(self, root, cls_names, set_name, **kwargs):
        MNameMapper.__init__(self, cls_names=cls_names)
        self._root = root
        self._set_name = set_name
        import scipy.io as sio
        datas = sio.loadmat(os.path.join(self.root, ensure_extend(set_name + '_32x32', 'mat')))
        self._imgs = datas['X'].transpose(3, 0, 1, 2)
        self._tragets = datas['y'].astype(np.int64).squeeze()
        self._metas = ['mnist_' + set_name + '_' + int(i) for i in range(len(self._tragets))]


class MNIST(MDataSource):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//MNIST//',
        PLATFORM_DESTOPLAB: 'D://Datasets//MNIST//',
        PLATFORM_SEV3090: '//home//data-storage//MNIST',
        PLATFORM_SEV4090: '//home//data-storage//MNIST',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/MNIST',
        PLATFORM_BOARD: '/home/jd/datas/DataSets/MNIST'
    }
    CLS_NAMES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    SET_NAMES = ('train', 'test',)

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=SVHN.SET_NAMES)
        self.kwargs = kwargs

    def _dataset(self, set_name, **kwargs):
        assert set_name in MNIST.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=SVHN.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = MNISTDataset(**kwargs_update)
        return dataset


class SVHN(MDataSource):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//SVHN//',
        PLATFORM_DESTOPLAB: 'D://Datasets//SVHN//',
        PLATFORM_SEV3090: '//home//datas-storage//SVHN',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/SVHN',
        PLATFORM_BOARD: '/home/jd/datas/DataSets/SVHN'
    }
    CLS_NAMES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10')
    SET_NAMES = ('train', 'test', 'extra')

    def __init__(self, root=None, **kwargs):
        MDataSource.__init__(self, root=root, set_names=SVHN.SET_NAMES)
        self.kwargs = kwargs

    def _dataset(self, set_name, **kwargs):
        assert set_name in SVHN.SET_NAMES
        kwargs_update = dict(root=self.root, cls_names=SVHN.CLS_NAMES, set_name=set_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        dataset = SVHNDataset(**kwargs_update)
        return dataset


if __name__ == '__main__':
    ds = MNIST()
    dataset = ds._dataset('train')
    img, label = dataset[5]

# if __name__ == '__main__':
#     ds = SVHN()
#     loader = ds.loader(set_name='train', batch_size=4, pin_memory=False, num_workers=0, aug_seqTp=None)
#     imgs, labs = next(iter(loader))
