import copy
import random
import time
from collections.abc import Iterable
from itertools import repeat
from typing import List, Union, Tuple, Sized, Optional, Sequence

import numpy as np

from .define import stylize_msg, MSG_STYLE, destylize_msg, memory_size
from .iotools import PLACEHOLDER, BROADCAST, COLORFUL, format_period, format_memsize

LTV_Int = Union[List[int], Tuple[int, ...], int]
T_Int = Tuple[int, ...]
TV_Int = Union[int, Tuple[int, ...]]
TV_Int2 = Union[int, Tuple[int, int]]
TV_Int3 = Union[int, Tuple[int, int, int]]
TV_Int4 = Union[int, Tuple[int, int, int, int]]
TV_Int5 = Union[int, Tuple[int, int, int, int, int]]

T_Flt = Tuple[float, ...]
TV_Flt = Union[float, Tuple[float]]
TV_Flt2 = Union[float, Tuple[float, float]]

LTV_Str = Union[List[str], Tuple[str, ...], str]
LV_Str = Union[List[str], str]
LT_Str = Union[List[str], Tuple[str]]

SN_Flt = Union[np.ndarray, Sequence[float]]
SN_Int = Union[np.ndarray, Sequence[int]]
NV_Int = Union[np.ndarray, int]
NV_Flt = Union[np.ndarray, float]


def ps_int_multiply(value: Union[float, int], reference: Union[float, int]) -> int:
    return value if isinstance(value, int) else int(value * reference)


def ps_int_randcvt(value: Union[float, int]) -> int:
    if isinstance(value, int):
        return value
    value_base = int(value)
    dt = 1 if random.random() < value - value_base else 0
    return value_base + dt


def ps_int2_repeat(value: TV_Int2) -> (int, int):
    if isinstance(value, Iterable):
        return tuple(value[:2])
    return tuple(repeat(value, 2))


def ps_int3_repeat(value: TV_Int3) -> (int, int):
    if isinstance(value, Iterable):
        return tuple(value[:3])
    return tuple(repeat(value, 3))


def ps_flt2_repeat(value: TV_Flt2) -> (float, float):
    if isinstance(value, Iterable):
        return tuple(value[:2])
    return tuple(repeat(value, 2))


# <editor-fold desc='显示进度的迭代器'>

def _round_exp(val: np.ndarray, base: int = 10) -> np.ndarray:
    lev = np.round(np.log(val) / np.log(base)) * np.log(base)
    return np.exp(lev)


def _round_step(step: int = 10, base: int = 10, groups: Tuple[int] = (1, 2, 5)) -> int:
    groups = np.array(groups)
    approx = _round_exp(step / groups, base=base) * groups
    return max(int(approx[np.argmin(np.abs(approx - step))]), 1)


class IntervalTrigger():
    def __init__(self, step: int, offset: int = 0, first: bool = False, last: bool = False):
        self.first = first
        self.last = last
        self.step = step
        self.offset = offset

    def trigger(self, ind: int, total: Optional[int]) -> bool:
        flag = self.first and ind == 0
        flag |= self.last and (ind + 1) == total
        flag |= (ind + 1 - self.offset) % self.step == 0
        return flag


# 显示进度的迭代器
class MEnumerate(Iterable, IntervalTrigger):
    FORMATTER = stylize_msg('Iter ', style=MSG_STYLE.BOLD) + PLACEHOLDER.IND + ' / ' + PLACEHOLDER.TOTAL + \
                ' | ' + stylize_msg('Time ', style=MSG_STYLE.BOLD) + PLACEHOLDER.TIME + \
                ' | ' + stylize_msg('ETA ', style=MSG_STYLE.BOLD) + PLACEHOLDER.ETA

    def __init__(self, seq, formatter=FORMATTER, step: int = None, broadcast=BROADCAST, num_step: int = 10,
                 total: int = None, offset: int = 0, first: bool = True, last: bool = True, colorful=COLORFUL):
        self.seq = seq
        assert total is not None or isinstance(seq, Sized)
        self.total = len(seq) if total is None else total
        step = _round_step(self.total // num_step) if step is None else step
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.colorful = colorful
        self.formatter = formatter if colorful else destylize_msg(formatter)
        self.broadcast = broadcast
        self._width = int(np.ceil(np.log(self.total) / np.log(10))) if not self.total == 0 else 1
        self._msizes = []

    def __iter__(self):
        self._ind = 0
        self._core = iter(self.seq)
        self.time_start = time.time()
        return self

    def __next__(self):
        ind = copy.deepcopy(self._ind)
        val = next(self._core)
        if self.trigger(self._ind, self.total):
            time_cur = time.time()
            time_aver = (time_cur - self.time_start) / ind if ind > 0 else 0
            eta = time_aver * (self.total - ind)
            msg = time.strftime(self.formatter, time.localtime(time_cur))
            msg = msg.replace(PLACEHOLDER.IND, ('%d' % (ind + 1)).center(self._width))
            msg = msg.replace(PLACEHOLDER.TOTAL, ('%d' % self.total).center(self._width))
            msg = msg.replace(PLACEHOLDER.ETA, format_period(eta))
            msg = msg.replace(PLACEHOLDER.TIME, '%.3f' % time_aver)
            if PLACEHOLDER.MSIZE in msg or PLACEHOLDER.EMU in msg:
                msize = memory_size(val)
                msg = msg.replace(PLACEHOLDER.MSIZE, format_memsize(msize))
                self._msizes.append(msize)
                emu = np.mean(self._msizes) * self.total
                msg = msg.replace(PLACEHOLDER.EMU, format_memsize(emu))
            self.broadcast(msg)
        self._ind = self._ind + 1
        return ind, val

# </editor-fold>
