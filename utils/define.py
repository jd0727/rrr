import inspect
import math
import random
import re
import sys
from abc import ABCMeta
from collections import OrderedDict
from collections.abc import Iterable
from functools import partial
from typing import List, Union, Callable, Tuple

import PIL.Image as Image
import numpy as np
import pandas as pd
import torch


# <editor-fold desc='多态注册器'>
class Register(OrderedDict):

    def registry(self, *keys):
        def wrapper(_func):
            for key in keys:
                self[key] = _func
            return _func

        return wrapper


class ClassRegister(Register):
    def fetch(self, obj: object, strict: bool = True):
        if strict:
            return self[obj]
        else:
            for cls, _value in self.items():
                if isinstance(obj, cls):
                    return _value
            raise Exception('err')


class Convertable(metaclass=ABCMeta):
    REGISTER_COVERT = Register()

    @classmethod
    def convert(cls, obj, **kwargs):
        if obj.__class__ == cls:
            return obj
        for cls_, func in cls.REGISTER_COVERT.items():
            if obj.__class__ == cls_:
                return func(obj, **kwargs)
        for cls_, func in cls.REGISTER_COVERT.items():
            if isinstance(obj, cls_):
                return func(obj, **kwargs)

        raise Exception('err fmt ' + str(obj.__class__.__name__))


# </editor-fold>


# <editor-fold desc='随机种子'>
REGISTER_RANDDOM_SEED = Register()


@REGISTER_RANDDOM_SEED.registry(np)
def _set_random_seed_numpy(seed: int = 1):
    random.seed(seed)


@REGISTER_RANDDOM_SEED.registry(torch)
def _set_random_seed_torch(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_random_seed(seed: int = 1):
    random.seed(seed)
    for setter in REGISTER_RANDDOM_SEED.values():
        setter(seed)
    return None


# </editor-fold>

# <editor-fold desc='JSON序列化'>

REGISTER_JSON_ENC = Register()
REGISTER_JSON_DEC = Register()


@REGISTER_JSON_ENC.registry(np.ndarray)
def _ndarray2json_dct(arr: np.ndarray) -> dict:
    return {'value': tuple(arr.flatten()), 'shape': tuple(arr.shape)}


@REGISTER_JSON_DEC.registry(np.ndarray.__name__)
def _json_dct2ndarray(json_dct: dict) -> np.ndarray:
    return np.array(json_dct['value']).reshape(json_dct['shape'])


def obj2json_dct(obj: object, **kwargs) -> object:
    if isinstance(obj, (str, int, float)):
        return obj
    elif obj.__class__ in REGISTER_JSON_ENC.keys():
        func = REGISTER_JSON_ENC[obj.__class__]
        if func is not None:
            json_dct = func(obj, **kwargs)
            json_dct['type'] = obj.__class__.__name__
        else:
            json_dct = {}
        return json_dct
    elif isinstance(obj, list) or isinstance(obj, Tuple) or isinstance(obj, set):
        lst = []
        for v in obj:
            lst.append(obj2json_dct(v, **kwargs))
        return lst
    elif isinstance(obj, dict):
        json_dct = {}
        for key, value in obj.items():
            key = obj2json_dct(key, **kwargs)
            value = obj2json_dct(value, **kwargs)
            json_dct[key] = value
        return json_dct
    else:
        raise Exception('no impl')


def json_dct2obj(json_dct: object, **kwargs) -> object:
    if isinstance(json_dct, (str, int, float)):
        return json_dct

    elif isinstance(json_dct, list):
        lst = []
        for v in json_dct:
            lst.append(json_dct2obj(v))
        return lst

    elif isinstance(json_dct, dict):
        if 'type' in json_dct.keys():
            type_name = dict(json_dct).pop('type')
            if type_name in REGISTER_JSON_DEC.keys():
                func = REGISTER_JSON_DEC[type_name]
                return func(json_dct)
            else:
                raise Exception('err')
        else:
            dct = {}
            for key, value in json_dct.items():
                key = json_dct2obj(key, **kwargs)
                value = json_dct2obj(value, **kwargs)
                dct[key] = value
            return dct
    else:
        raise Exception('err')


def _getattrs(obj: object, attr_names: Tuple) -> dict:
    dct = {}
    for attr_name in attr_names:
        attr = getattr(obj, attr_name)
        dct[attr_name] = obj2json_dct(attr)
    return dct


def _obj_from_init(attr_dct: dict, cls: type, attr_names: Tuple) -> object:
    return cls(*[json_dct2obj(attr_dct[attr_name]) for attr_name in attr_names])


def REGISTRY_JSON_ENC_BY_ATTR(cls: type, attr_names: Tuple):
    REGISTER_JSON_ENC[cls] = partial(_getattrs, attr_names=attr_names)


def REGISTRY_JSON_ENC_BY_SLOTS(cls: type):
    REGISTER_JSON_ENC[cls] = partial(_getattrs, attr_names=cls.__slots__)


def _get_important_paras(init_func: Callable) -> Tuple[str]:
    paras = inspect.signature(init_func).parameters
    names = []
    for name, para in paras.items():
        if name == 'self':
            continue
        elif not para.kind == 1:
            continue
        names.append(name)
    return tuple(names)


def REGISTRY_JSON_DEC_BY_INIT(cls: type):
    names = _get_important_paras(cls.__init__)
    REGISTER_JSON_DEC[cls.__name__] = partial(_obj_from_init, cls=cls, attr_names=names)


def REGISTRY_JSON_ENCDEC_BY_INIT(cls: type):
    names = _get_important_paras(cls.__init__)
    REGISTER_JSON_DEC[cls.__name__] = partial(_obj_from_init, cls=cls, attr_names=names)
    REGISTER_JSON_ENC[cls] = partial(_getattrs, attr_names=names)


# </editor-fold>

# <editor-fold desc='计算对象占用空间'>
REGISTER_MEM_SIZE = Register()


@REGISTER_MEM_SIZE.registry(Image.Image)
def _mem_size_pil_img(img: Image.Image):
    memsize = sys.getsizeof(img)
    memsize += sys.getsizeof(img.tobytes())
    return memsize


@REGISTER_MEM_SIZE.registry(np.ndarray)
def _mem_size_numpy_array(arr: np.ndarray):
    memsize = sys.getsizeof(arr)
    memsize += arr.dtype.itemsize * arr.size
    return memsize


@REGISTER_MEM_SIZE.registry(torch.Tensor)
def _mem_size_torch_tensor(tensor: torch.Tensor):
    memsize = sys.getsizeof(tensor)
    memsize += tensor.element_size() * torch.numel(tensor)
    return memsize


# 计算内存占用
def memory_size(obj: object, seen: set = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if obj in REGISTER_MEM_SIZE.keys():
        func = REGISTER_MEM_SIZE[obj]
        memsize = func(obj)
    else:
        memsize = 0

    if isinstance(obj, dict):
        for v in obj.values():
            if not isinstance(v, (str, int, float, bytes, bytearray)):
                memsize += memory_size(v, seen)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        for v in obj:
            if not isinstance(v, (str, int, float, bytes, bytearray)):
                memsize += memory_size(v, seen)

    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                dct = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(dct) or inspect.ismemberdescriptor(dct):
                    memsize += memory_size(obj.__dict__, seen)
                break

    if hasattr(obj, '__slots__'):
        for s in obj.__slots__:
            memsize += memory_size(getattr(obj, s), seen)

    return memsize


# </editor-fold>

# <editor-fold desc='dataframe格式化'>
REGISTER_DATAFRAME_ELE2STR = Register()


@REGISTER_DATAFRAME_ELE2STR.registry(float)
def _ele2str_float(val: float):
    return '%.5f' % val


@REGISTER_DATAFRAME_ELE2STR.registry(np.int32, np.int64)
def _ele2str_numpy_int(val: Union[np.int32, np.int64]):
    return str(int(val))


@REGISTER_DATAFRAME_ELE2STR.registry(np.float32, np.float64)
def _ele2str_numpy_float(val: Union[np.float32, np.float64]):
    return '%.5f' % float(val)


def dataframe2strs(data: pd.DataFrame, inter_col: str = '\t', divider: int = 1) -> List[str]:
    num_row = len(data.index)
    num_col = len(data.columns)
    buffer = [[''] + list(data.columns)]
    for index, (_index, row) in enumerate(data.iterrows()):
        buffer_row = [str(index)]
        for v in row.values:
            if v is not None and v.__class__ in REGISTER_DATAFRAME_ELE2STR.keys():
                v = REGISTER_DATAFRAME_ELE2STR[v.__class__](v)
            else:
                v = str(v)
            buffer_row.append(v)
        buffer.append(buffer_row)
    for j in range(num_col + 1):
        max_len = 0
        for i in range(num_row + 1):
            max_len = max(max_len, len(buffer[i][j]))
        max_len = int(math.ceil(max_len / divider)) * divider
        for i in range(num_row + 1):
            buffer[i][j] = buffer[i][j].center(max_len)
    for i in range(num_row + 1):
        buffer[i] = inter_col.join(buffer[i])
    return buffer


# </editor-fold>

# <editor-fold desc='文本修饰'>
class MSG_FG_COLOR:
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLO = 33
    BLUE = 34
    PURPLE = 35
    CYAN = 36
    WHITE = 37
    DEFAULT = None


class MSG_BG_COLOR:
    BLACK = 40
    RED = 41
    GREEN = 42
    YELLO = 43
    BLUE = 44
    PURPLE = 45
    CYAN = 46
    WHITE = 47
    DEFAULT = None


class MSG_STYLE:
    NORMAL = 0  # 终端默认设置
    BOLD = 1  # 高亮显示
    UNDERLINE = 4  # 使用下划线
    BLINK = 5  # 闪烁
    INVERT = 7  # 反白显示
    HIDE = 8  # 不可见
    DEFAULT = None


def stylize_msg(msg: str, style: int = None, fg_color: int = None, bg_color: int = None) -> str:
    prefix = ';'.join(['%s' % s for s in [style, fg_color, bg_color] if s is not None])
    prefix = '\033[%sm' % prefix if prefix else ''
    appendix = '\033[%sm' % 0 if prefix else ''
    return '%s%s%s' % (prefix, msg, appendix)


def destylize_msg(msg: str) -> str:
    partten = '\033' + '\[' + '[\d;]*m'
    msg = re.sub(partten, '', msg, count=0, flags=0)
    return msg


# </editor-fold>

# <editor-fold desc='静态变量'>

class EXTENDS:
    MODEL_WEIGHT = 'pth'
    OPTIMIZER_WEIGHT = 'opt'
    TXT = 'txt'
    CHECKPOINT = 'ckpt'
    CACHE = 'pkl'
    DCT = 'json'
    EXCEL = 'xlsx'


class TIMENODE:
    BEFORE_TRAIN = 'before_train'
    AFTER_TRAIN = 'after_train'
    BEFORE_PROCESS = 'before_process'
    AFTER_PROCESS = 'after_process'
    BEFORE_EVAL = 'before_eval'
    AFTER_EVAL = 'after_eval'
    BEFORE_CALC = 'before_calc'
    AFTER_CALC = 'after_calc'
    BEFORE_INIT = 'before_init'
    AFTER_INIT = 'after_init'
    BEFORE_CYCLE = 'before_cycle'
    AFTER_CYCLE = 'after_cycle'
    BEFORE_EPOCH = 'before_epoch'
    AFTER_EPOCH = 'after_epoch'
    BEFORE_ITER = 'before_iter'
    AFTER_ITER = 'after_iter'
    BEFORE_INFER = 'before_infer'
    AFTER_INFER = 'after_infer'
    BEFORE_LOAD = 'before_load'
    AFTER_LOAD = 'after_load'
    BEFORE_TARGET = 'before_target'
    AFTER_TARGET = 'after_target'
    BEFORE_CORE = 'before_core'
    AFTER_CORE = 'after_core'
    BEFORE_FORWARD = 'before_foward'
    AFTER_FORWARD = 'after_foward'

    BEFORE_IMG_SAVE = 'before_img_save'
    AFTER_IMG_SAVE = 'after_img_save'

    BEFORE_IMG_RNDSAVE = 'before_img_rndsave'
    AFTER_IMG_RNDSAVE = 'after_img_rndsave'

    BEFORE_FORWARD_GEN = 'before_foward_gen'
    AFTER_FORWARD_GEN = 'after_foward_gen'

    BEFORE_FORWARD_DIS = 'before_foward_dis'
    AFTER_FORWARD_DIS = 'after_foward_dis'

    BEFORE_BACKWARD = 'before_backward'
    AFTER_BACKWARD = 'after_backward'

    BEFORE_BACKWARD_GEN = 'before_backward_gen'
    AFTER_BACKWARD_GEN = 'after_backward_gen'

    BEFORE_FORWARD_ENC = 'before_foward_enc'
    AFTER_FORWARD_ENC = 'after_foward_enc'

    BEFORE_FORWARD_DEC = 'before_foward_dec'
    AFTER_FORWARD_DEC = 'after_foward_dec'

    BEFORE_BACKWARD_DIS = 'before_backward_dis'
    AFTER_BACKWARD_DIS = 'after_backward_dis'

    BEFORE_OPTIMIZE = 'before_optimize'
    AFTER_OPTIMIZE = 'after_optimize'

    BEFORE_OPTIMIZE_DIS = 'before_optimize_dis'
    AFTER_OPTIMIZE_DIS = 'after_optimize_dis'

    BEFORE_OPTIMIZE_GEN = 'before_optimize_gen'
    AFTER_OPTIMIZE_GEN = 'after_optimize_gen'


class PERIOD:
    TRAIN = (TIMENODE.BEFORE_TRAIN, TIMENODE.AFTER_TRAIN)
    PROCESS = (TIMENODE.BEFORE_PROCESS, TIMENODE.AFTER_PROCESS)
    EVAL = (TIMENODE.BEFORE_EVAL, TIMENODE.AFTER_EVAL)
    CALC = (TIMENODE.BEFORE_CALC, TIMENODE.AFTER_CALC)
    INIT = (TIMENODE.BEFORE_INIT, TIMENODE.AFTER_INIT)
    CYCLE = (TIMENODE.BEFORE_CYCLE, TIMENODE.AFTER_CYCLE)
    EPOCH = (TIMENODE.BEFORE_EPOCH, TIMENODE.AFTER_EPOCH)
    ITER = (TIMENODE.BEFORE_ITER, TIMENODE.AFTER_ITER)
    INFER = (TIMENODE.BEFORE_INFER, TIMENODE.AFTER_INFER)
    LOAD = (TIMENODE.BEFORE_LOAD, TIMENODE.AFTER_LOAD)
    TARGET = (TIMENODE.BEFORE_TARGET, TIMENODE.AFTER_TARGET)
    CORE = (TIMENODE.BEFORE_CORE, TIMENODE.AFTER_CORE)
    FORWARD = (TIMENODE.BEFORE_FORWARD, TIMENODE.AFTER_FORWARD)
    FORWARD_GEN = (TIMENODE.BEFORE_FORWARD_GEN, TIMENODE.AFTER_FORWARD_GEN)
    FORWARD_ENC = (TIMENODE.BEFORE_FORWARD_ENC, TIMENODE.AFTER_FORWARD_ENC)
    FORWARD_DEC = (TIMENODE.BEFORE_FORWARD_DEC, TIMENODE.AFTER_FORWARD_DEC)
    FORWARD_DIS = (TIMENODE.BEFORE_FORWARD_DIS, TIMENODE.AFTER_FORWARD_DIS)
    BACKWARD = (TIMENODE.BEFORE_BACKWARD, TIMENODE.AFTER_BACKWARD)
    BACKWARD_GEN = (TIMENODE.BEFORE_BACKWARD_GEN, TIMENODE.AFTER_BACKWARD_GEN)
    BACKWARD_DIS = (TIMENODE.BEFORE_BACKWARD_DIS, TIMENODE.AFTER_BACKWARD_DIS)
    OPTIMIZE = (TIMENODE.BEFORE_OPTIMIZE, TIMENODE.AFTER_OPTIMIZE)
    OPTIMIZE_GEN = (TIMENODE.BEFORE_OPTIMIZE_GEN, TIMENODE.AFTER_OPTIMIZE_GEN)
    OPTIMIZE_DIS = (TIMENODE.BEFORE_OPTIMIZE_DIS, TIMENODE.AFTER_OPTIMIZE_DIS)
    IMG_SAVE = (TIMENODE.BEFORE_IMG_SAVE, TIMENODE.AFTER_IMG_SAVE)
    IMG_RNDSAVE = (TIMENODE.BEFORE_IMG_RNDSAVE, TIMENODE.AFTER_IMG_RNDSAVE)
# </editor-fold>
