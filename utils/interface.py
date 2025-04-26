from abc import abstractmethod
from typing import NoReturn, Optional

import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset

from .define import *


# <editor-fold desc='基础属性接口'>
class HasImageSize(metaclass=ABCMeta):
    @property
    @abstractmethod
    def img_size(self) -> Optional[Tuple[int, int]]:
        pass


class SettableImageSize(HasImageSize):

    @property
    @abstractmethod
    def img_size(self) -> Optional[Tuple[int, int]]:
        pass

    @img_size.setter
    @abstractmethod
    def img_size(self, img_size: Optional[Tuple[int, int]]) -> NoReturn:
        pass


class HasSize(metaclass=ABCMeta):
    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        pass


class SettableSize(HasSize):

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        pass

    @size.setter
    @abstractmethod
    def size(self, size) -> NoReturn:
        pass


class HasDevice(metaclass=ABCMeta):

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass


class HasNumData(metaclass=ABCMeta):

    @property
    @abstractmethod
    def num_data(self) -> int:
        pass


class SettableDevice(metaclass=ABCMeta):

    @HasDevice.device.setter
    @abstractmethod
    def device(self, device) -> NoReturn:
        pass


class HasNumClass(metaclass=ABCMeta):

    @property
    @abstractmethod
    def num_cls(self) -> int:
        pass


class HasClassMapper(HasNumClass):
    @abstractmethod
    def name2cind(self, name: str) -> int:
        pass

    @abstractmethod
    def cind2name(self, cind: int) -> str:
        pass


class HasColorMapper(HasNumClass):

    @abstractmethod
    def name2col(self, name: str) -> tuple:
        pass


class HasSetName(metaclass=ABCMeta):

    @property
    @abstractmethod
    def set_name(self) -> str:
        pass


class HasBatchSize(metaclass=ABCMeta):

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass


class SettableBatchSize(HasBatchSize):

    @property
    @abstractmethod
    def batch_size(self) -> int:
        pass

    @batch_size.setter
    @abstractmethod
    def batch_size(self, batch_size) -> NoReturn:
        pass


# </editor-fold>

# <editor-fold desc='基础属性接口'>


class ImageRecognizable(HasImageSize, HasNumClass):
    pass


class ImageGeneratable(ImageRecognizable):
    @abstractmethod
    def gen_imgs(self, **kwargs):
        pass


class IMTorchModel(nn.Module, ImageRecognizable, HasDevice):
    pass


class TrainableModel(ImageRecognizable):

    @abstractmethod
    def labels2tars(self, labels, **kwargs):
        pass

    @abstractmethod
    def act_iter_train(self, trainer, imgs, targets, **kwargs):
        pass

    @abstractmethod
    def act_init_train(self, trainer, **kwargs):
        pass


class EvalableModel(ImageRecognizable):

    @abstractmethod
    def act_iter_eval(self, container, imgs, labels, **kwargs):
        pass

    @abstractmethod
    def act_init_eval(self, container, **kwargs):
        pass


class AnnotatableModel(ImageRecognizable):

    @abstractmethod
    def act_iter_annotate(self, container, imgs, labels, **kwargs):
        pass

    @abstractmethod
    def act_init_annotate(self, container, **kwargs):
        pass


class IMDataset(TorchDataset, HasClassMapper, HasSetName):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class IVirtualDataLoader(HasClassMapper, SettableImageSize, SettableBatchSize, HasNumData, HasSetName):
    pass


class IMDataLoader(TorchDataLoader, IVirtualDataLoader):
    pass

# </editor-fold>
