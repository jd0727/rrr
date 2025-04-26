import itertools

import torch.distributed as dist

from .prefetcher import *


# <editor-fold desc='杂项'>

def _set_model_state(model, train_mode: bool = True, enable_half: bool = False) -> bool:
    if isinstance(model, nn.Module):
        if not enable_half:
            model.float()
        model.train(train_mode)
        return True
    else:
        return False


def dist_barrier():
    if dist.is_initialized():
        dist.barrier()


def all_extend_object_list(obj_list: list) -> list:
    if not dist.is_initialized():
        return obj_list
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device('cuda:' + str(rank))
        torch.cuda.set_device(device)
        obj_listss = [None] * world_size
        dist.all_gather_object(obj_listss, obj_list)
        obj_list = list(itertools.chain(*obj_listss))
        return obj_list


def sec2msg(sec):
    if sec == float('inf'):
        return 'inf'
    else:
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m, s)


# </editor-fold>


class IterBasedTemplate(Prefetcher, ActorContainer, TimeManager, BroadcastManager):

    def __init__(self,
                 loader: IVirtualDataLoader,
                 total_epoch: Optional[int] = None,
                 total_iter: Optional[int] = None,
                 device: torch.device = DEVICE,
                 processor: Optional[Callable] = None):
        ActorContainer.__init__(self)
        TimeManager.__init__(self)
        Prefetcher.__init__(self, loader=loader, device=device, processor=processor)
        self._ind_epoch = 0
        self._ind_iter = 0
        self._ind_iter_inep = 0
        self._total_epoch = total_epoch
        self._total_iter = total_iter
        self._running = True
        self._main_proc = IS_MAIN_PROC()

    @property
    def main_proc(self) -> bool:
        return self._main_proc

    @property
    def running(self) -> bool:
        return self._running

    @running.setter
    def running(self, running: bool) -> NoReturn:
        self._running = running

    @property
    def ind_epoch(self) -> int:
        return self._ind_epoch

    @ind_epoch.setter
    def ind_epoch(self, ind_epoch: int) -> NoReturn:
        self._ind_epoch = ind_epoch

    @property
    def ind_iter(self) -> int:
        return self._ind_iter

    @ind_iter.setter
    def ind_iter(self, ind_iter: int) -> NoReturn:
        self._ind_iter = ind_iter

    @property
    def ind_iter_inep(self) -> int:
        return self._ind_iter_inep

    @ind_iter_inep.setter
    def ind_iter_inep(self, ind_iter_inep: int) -> NoReturn:
        self._ind_iter_inep = ind_iter_inep

    @property
    def total_iter_inep(self) -> int:
        return self.__len__()

    @property
    def total_epoch(self) -> int:
        return self._total_epoch

    @total_epoch.setter
    def total_epoch(self, total_epoch: int) -> NoReturn:
        self._total_epoch = total_epoch

    @property
    def total_iter(self) -> int:
        return self._total_iter

    @total_iter.setter
    def total_iter(self, total_iter: int) -> NoReturn:
        self._total_iter = total_iter

    @property
    def eta(self) -> float:
        time_start = self._time_dct.get(TIMENODE.BEFORE_CYCLE, 0)
        time_cur = self._time_dct.get(TIMENODE.AFTER_ITER, 0)
        sec_cycled = max(time_cur - time_start, 0.001)

        total_iter = self.total_iter
        total_epoch = self.total_epoch
        ind_iter = self.ind_iter
        ind_epoch = self.ind_epoch

        if total_iter is None and total_epoch is not None:
            total_iter = total_epoch * len(self)

        scale_epoch = float('inf') if total_epoch is None or ind_epoch == 0 \
            else 1 / ind_epoch * (total_epoch - ind_epoch)
        scale_iter = float('inf') if total_iter is None or ind_iter == 0 \
            else 1 / ind_iter * (total_iter - ind_iter)

        sec = sec_cycled * min(scale_epoch, scale_iter)
        return sec

    @abstractmethod
    def act_iter(self):
        pass

    @abstractmethod
    def act_init(self, *args, **kwargs):
        pass

    @abstractmethod
    def act_ending(self):
        pass

    @abstractmethod
    def act_return(self):
        pass

    def start(self, *args, **kwargs):
        self.update_time(TIMENODE.BEFORE_PROCESS, TIMENODE.BEFORE_INIT)
        self.act_init(*args, **kwargs)
        for actor in self.get_actors(InitialActor):
            actor.act_init(self, )
        self.update_time(TIMENODE.AFTER_INIT, TIMENODE.BEFORE_CYCLE)
        while self.running:
            self.update_time(TIMENODE.BEFORE_EPOCH)
            for actor in self.get_actors(BeforeEpochActor):
                actor.act_before_epoch(self, )
            self.set_epoch(self.ind_epoch)
            self.ind_iter_inep = 0
            iterator = iter(self)
            while self.running:
                self.update_time(TIMENODE.BEFORE_ITER)
                for actor in self.get_actors(BeforeIterActor):
                    actor.act_before_iter(self, )
                try:
                    self.batch_data = next(iterator)
                except StopIteration:
                    break
                self.act_iter()
                self.update_time(TIMENODE.AFTER_ITER)
                for actor in self.get_actors(AfterIterActor):
                    actor.act_after_iter(self, )
                self.ind_iter = self.ind_iter + 1
                self.ind_iter_inep = self.ind_iter_inep + 1
                self.running = self.running and (self.total_iter is None or self.ind_iter < self.total_iter)

            self.update_time(TIMENODE.AFTER_EPOCH)
            for actor in self.get_actors(AfterEpochActor):
                actor.act_after_epoch(self, )
            self.ind_epoch = self.ind_epoch + 1
            self.running = self.running and (self.total_epoch is None or self.ind_epoch < self.total_epoch)

        self.update_time(TIMENODE.AFTER_CYCLE)
        for actor in self.get_actors(AfterCycleActor):
            actor.act_after_cycle(self, )
        self.act_ending()
        for actor in self.get_actors(EndingActor):
            actor.act_ending(self, )
        self.update_time(TIMENODE.AFTER_PROCESS)
        return self.act_return()

# </editor-fold>
