import threading

from torch.utils.data.distributed import DistributedSampler as TorchDistributedSampler

from .define import *


# <editor-fold desc='数据预读取'>


class PrefetcherOld(TimeManager, Iterable, Sized):

    def __init__(self, loader: IVirtualDataLoader, device: torch.device = None, processor: Optional[Callable] = None):
        TimeManager.__init__(self)
        self.loader = loader
        self.cuda_stream = None
        self.device = device
        self.loader_iter = None
        self.processor = processor

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = torch.device(device) if device is not None else DEVICE
        if self.device.index is not None:
            torch.cuda.set_device(self.device.index)
            self.cuda_stream = torch.cuda.Stream(self.device)
        else:
            self.cuda_stream = None

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    @property
    def batch_size(self):
        return self.loader.batch_size

    @property
    def num_batch(self):
        return len(self.loader)

    @property
    def num_data(self):
        return self.loader.num_data

    def set_epoch(self, ind_epoch):
        if isinstance(self.loader.sampler, TorchDistributedSampler):
            self.loader.sampler.set_epoch(ind_epoch)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.thread = threading.Thread(target=self._prefetch_data, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        self.thread.join()
        if self.imgs is None:
            self.loader_iter = None
            raise StopIteration
        else:
            imgs, labels = self.imgs, self.labels
            self.thread = threading.Thread(target=self._prefetch_data, daemon=True)
            self.thread.start()
            return imgs, labels

    def _prefetch_data(self):
        try:
            time_before = time.time()
            self.imgs, self.labels = next(self.loader_iter)
            self.update_time(TIMENODE.AFTER_LOAD)
            self.update_time(TIMENODE.BEFORE_LOAD, time_cur=time_before)
            if self.cuda_stream is not None and isinstance(self.imgs, torch.Tensor):
                with torch.cuda.stream(self.cuda_stream):
                    self.imgs = self.imgs.to(device=self.device, non_blocking=True)
            if self.processor is not None:
                time_before = time.time()
                self.labels = self.processor(self.labels)
                self.update_time(TIMENODE.AFTER_TARGET)
                self.update_time(TIMENODE.BEFORE_TARGET, time_cur=time_before)
        except StopIteration:
            self.imgs, self.labels = None, None
        return None


class Prefetcher(TimeManager, Iterable, Sized):
    def __init__(self, loader: IVirtualDataLoader, device=None, processor=None):
        TimeManager.__init__(self)
        self.loader = loader
        self.cuda_stream = None
        self.device = device
        self.loader_iter = None
        self.processor = processor

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = torch.device(device) if device is not None else DEVICE
        # self.cuda_stream = torch.cuda.Stream(self.device) if self.device.index is not None else None

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    @property
    def batch_size(self):
        return self.loader.batch_size

    @property
    def num_batch(self):
        return len(self.loader)

    @property
    def num_data(self):
        return self.loader.num_data

    def __len__(self):
        return len(self.loader)

    def set_epoch(self, ind_epoch):
        if isinstance(self.loader, IMDataLoader) \
                and isinstance(self.loader.sampler, torch.utils.data.distributed.DistributedSampler):
            self.loader.sampler.set_epoch(ind_epoch)

    def preload(self):
        try:
            time_before = time.time()
            self.imgs, self.labels = next(self.loader_iter)
            self.update_time(TIMENODE.BEFORE_LOAD, time_cur=time_before)
            self.update_time(TIMENODE.AFTER_LOAD)
        except StopIteration:
            self.imgs = None
            self.labels = None
            return
        self.update_time(TIMENODE.BEFORE_TARGET)
        if self.processor is not None:
            self.labels = self.processor(self.labels)
        self.update_time(TIMENODE.AFTER_TARGET)

        if self.cuda_stream is not None:
            with torch.cuda.stream(self.cuda_stream):
                if isinstance(self.imgs, torch.Tensor):
                    self.imgs = self.imgs.to(self.device, non_blocking=True)
                if is_arrsN(self.labels):
                    self.labels = arrsN2arrsT(self.labels, self.device)

    def __iter__(self):
        self.loader_iter = iter(self.loader)  # re-generate an iter for each epoch
        self.preload()
        return self

    def __next__(self):
        # torch.cuda.current_stream().wait_stream(self.cuda_stream)
        imgs = self.imgs
        labels = self.labels
        # if imgs is not None:
        #     imgs.record_stream(torch.cuda.current_stream())
        # if labels is not None:
        #     labels.record_stream(torch.cuda.current_stream())
        self.preload()
        if imgs is None:
            raise StopIteration
        return imgs, labels


class VirtualDataLoader(IVirtualDataLoader):
    def name2cind(self, name: str) -> int:
        return 0

    def cind2name(self, cind: int) -> str:
        return 'obj'

    @property
    def num_cls(self) -> int:
        return 1

    @property
    def img_size(self) -> Optional[Tuple[int, int]]:
        return None

    @property
    def set_name(self) -> str:
        return 'simu'


class SimuLoader(VirtualDataLoader):

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_data(self) -> int:
        return self.size

    def __init__(self, size: int = 10, delay_next: float = 0.2, delay_iter: float = 1.0, ptr: int = 0):
        self.size = size
        self.delay_next = delay_next
        self.delay_iter = delay_iter
        self.ptr = ptr

    def __len__(self):
        return self.size

    def __iter__(self):
        print('Build iter')
        time.sleep(self.delay_iter)
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr == self.size:
            raise StopIteration
        else:
            print('Fetching ', self.ptr)
            time.sleep(self.delay_next)
            self.ptr = self.ptr + 1
            imgs = torch.zeros(size=(1,))
            labels = []
            return imgs, labels


class SingleSampleLoader(VirtualDataLoader):
    @property
    def img_size(self) -> Optional[Tuple[int, int]]:
        if isinstance(self.imgs, torch.Tensor):
            return img2size(self.imgs)
        else:
            return None

    @property
    def batch_size(self) -> int:
        return len(self.imgs)

    def __init__(self, imgs, labels, total_iter: int = 10, ptr: int = 0):
        self.imgs = imgs
        self.labels = labels
        self.total_iter = total_iter
        self.ptr = ptr

    @property
    def num_data(self):
        return self.total_iter * self.batch_size

    def __len__(self):
        return self.total_iter

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.total_iter:
            raise StopIteration
        else:
            self.ptr = self.ptr + 1
            return self.imgs, self.labels

# </editor-fold>
