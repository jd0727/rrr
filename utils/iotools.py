import copy
import json
import os
import pickle
import platform
import sys
import time
import types
from typing import List, Union, Tuple, Dict, Sequence, Iterable, Callable

import PIL.Image as Image
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

try:
    import yaml
except Exception as e:
    pass
from .define import stylize_msg, MSG_STYLE, MSG_FG_COLOR, dataframe2strs, destylize_msg
import torch.distributed as dist

# <editor-fold desc='格式化与输出'>
COLORFUL = False


class PLACEHOLDER:
    @staticmethod
    def FMT(place_holder: str, fmt: str):
        return place_holder.replace('}', ':' + fmt + '}')

    ETA = '{ETA}'
    IND = '{IND}'
    TOTAL = '{TOTAL}'
    TIME = '{TIME}'
    DATE = '{DATE}'
    MSG = '{MSG}'
    IMG_SIZE = '{IMAGE_SIZE}'
    BATCH_SIZE = '{BATCH_SIZE}'
    IND_EPOCH = '{IND_EPOCH}'
    IND_ITER = '{IND_ITER}'
    META = '{META}'
    MSIZE = '{MSIZE}'
    EMU = '{EMU}'
    MB = '{MB}'
    KB = '{KB}'
    GB = '{GB}'
    NAME = '{NAME}'
    SET_NAME = '{SET_NAME}'
    EXTEND = '{EXTEND}'
    FILE_NAME = '{FILE_NAME}'
    FILE_DIR = '{FILE_DIR}'

    METRIC = '{METRIC}'
    PERFORMANCE = '{PERFORMANCE}'

    CLS_NAME = '{CLS_NAME}'
    PREFIX = '{PREFIX}'
    APPENDIX = '{APPENDIX}'


class FORMATTER:
    EMPTY = ''
    PERIOD = '%H:%M:%S'
    DATE = '%Y-%m-%d %H:%M:%S'
    BROADCAST = PLACEHOLDER.DATE + ' : ' + PLACEHOLDER.MSG
    BROADCAST_COLORFUL = stylize_msg(PLACEHOLDER.DATE, style=MSG_STYLE.BOLD) + ' : ' + PLACEHOLDER.MSG
    CACHE_FILE_NAME = PLACEHOLDER.META
    MSIZE = PLACEHOLDER.MB

    SAVE_PTH_SIMPLE = PLACEHOLDER.FILE_DIR + PLACEHOLDER.PREFIX + PLACEHOLDER.FILE_NAME \
                      + PLACEHOLDER.APPENDIX + PLACEHOLDER.EXTEND
    SAVE_PTH_EPOCH = PLACEHOLDER.FILE_DIR + PLACEHOLDER.PREFIX + PLACEHOLDER.FILE_NAME \
                     + PLACEHOLDER.APPENDIX + '_ep' + PLACEHOLDER.IND_EPOCH + PLACEHOLDER.EXTEND
    SAVE_PTH_ITER = PLACEHOLDER.FILE_DIR + PLACEHOLDER.PREFIX + PLACEHOLDER.FILE_NAME \
                    + PLACEHOLDER.APPENDIX + '_it' + PLACEHOLDER.IND_ITER + PLACEHOLDER.EXTEND
    SAVE_PTH_PRFM = PLACEHOLDER.FILE_DIR + PLACEHOLDER.PREFIX + PLACEHOLDER.FILE_NAME \
                    + PLACEHOLDER.APPENDIX + '_pr' + PLACEHOLDER.PERFORMANCE + PLACEHOLDER.EXTEND
    SAVE_PTH_EPOCH_PRFM = PLACEHOLDER.FILE_DIR + PLACEHOLDER.FILE_NAME \
                          + PLACEHOLDER.APPENDIX + '_ep' + PLACEHOLDER.IND_EPOCH + '_pr' \
                          + PLACEHOLDER.PERFORMANCE + PLACEHOLDER.EXTEND


def format_save_pth(save_pth, ind_iter: int = 0, ind_epoch: int = 0, perfmce: float = 0, default_name: str = '',
                    appendix: str = '', prefix: str = '', extend: str = '', formatter=FORMATTER.SAVE_PTH_SIMPLE,
                    **kwargs):
    file_name_pure, _ = os.path.splitext(save_pth)
    extend = _stdlize_extend(extend)
    file_name = os.path.basename(file_name_pure)
    file_dir = os.path.dirname(file_name_pure)
    if len(file_dir) == 0:
        return ''
    else:
        file_dir = file_dir + os.path.sep
    if len(file_name) == 0:
        file_name = default_name

    formatter = formatter.replace(PLACEHOLDER.FILE_NAME, file_name)
    formatter = formatter.replace(PLACEHOLDER.PREFIX, prefix)
    formatter = formatter.replace(PLACEHOLDER.APPENDIX, appendix)

    formatter = formatter.replace(PLACEHOLDER.FILE_DIR, file_dir)
    formatter = formatter.replace(PLACEHOLDER.EXTEND, extend)

    formatter = formatter.replace(PLACEHOLDER.PERFORMANCE, ('%04d' % (perfmce * 10000)).replace('.', '_'))
    formatter = formatter.replace(PLACEHOLDER.IND_EPOCH, '%03d' % ind_epoch)
    formatter = formatter.replace(PLACEHOLDER.IND_ITER, '%03d' % ind_iter)
    return formatter


def replace_date(msg: str) -> str:
    if PLACEHOLDER.DATE in msg:
        date = time.strftime(FORMATTER.DATE, time.localtime(time.time()))
        msg = msg.replace(PLACEHOLDER.DATE, date)
    return msg


def format_msg(msg: str, formatter=FORMATTER.BROADCAST) -> str:
    formatter = replace_date(formatter)
    formatter = formatter.replace(PLACEHOLDER.MSG, msg)
    return formatter


def format_period(second: int, formatter=FORMATTER.PERIOD) -> str:
    second, millisecond = divmod(second, 1)
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    formatter = formatter.replace('%H', '%02d' % hour)
    formatter = formatter.replace('%M', '%02d' % minute)
    formatter = formatter.replace('%S', '%02d' % second)
    return formatter


def format_memsize(memsize: int, formatter=FORMATTER.MSIZE) -> str:
    if PLACEHOLDER.KB in formatter:
        formatter = formatter.replace(PLACEHOLDER.KB, '%.3f' % (memsize * 1e-3))
    if PLACEHOLDER.MB in formatter:
        formatter = formatter.replace(PLACEHOLDER.MB, '%.3f' % (memsize * 1e-6))
    if PLACEHOLDER.GB in formatter:
        formatter = formatter.replace(PLACEHOLDER.GB, '%.3f' % (memsize * 1e-9))
    return formatter


def format_set_folder(set_name: str, formatter=FORMATTER.BROADCAST) -> str:
    formatter = formatter.replace(PLACEHOLDER.SET_NAME, set_name)
    return formatter


def IS_MAIN_PROC() -> bool:
    return not (dist.is_initialized() and dist.get_rank() > 0)


class PrintBasedBroadcaster():

    def __init__(self, formatter: str = FORMATTER.BROADCAST):
        self.formatter = formatter

    def __call__(self, msg: str, end='\n', colorful=COLORFUL):
        if IS_MAIN_PROC():
            print(format_msg(msg, self.formatter), end=end, )


class LogBasedBroadcaster():

    def __init__(self, save_pth: str, new_log: bool = True, with_print: bool = True,
                 formatter: str = FORMATTER.BROADCAST):
        save_pth = format_save_pth(save_pth=save_pth, extend='log', default_name='output')
        self.with_print = with_print
        self.formatter = formatter
        self.new_log = new_log
        self.save_pth = save_pth
        self._inited = False

    def __call__(self, msg: str, end='\n', colorful=COLORFUL):
        if IS_MAIN_PROC():
            if not self._inited:
                if self.new_log and os.path.exists(self.save_pth):
                    os.remove(self.save_pth)
                self._inited = True

            msg = format_msg(msg, self.formatter)
            msg_plain = destylize_msg(msg)
            ensure_file_dir(self.save_pth)
            mode = 'a' if os.path.exists(self.save_pth) else 'w'
            with open(self.save_pth, mode, encoding='utf-8') as file:
                file.write(msg_plain + '\n')
            if self.with_print:
                print(msg)


BROADCAST = PrintBasedBroadcaster(formatter=FORMATTER.BROADCAST_COLORFUL if COLORFUL else FORMATTER.BROADCAST)


def BROADCAST_DATAFRAME(data: pd.DataFrame, inter_col: str = '\t', divider: int = 1, colorful=COLORFUL):
    lines = dataframe2strs(data, inter_col=inter_col, divider=divider)
    for line in lines:
        BROADCAST(line, colorful=colorful)


# </editor-fold>

# <editor-fold desc='平台自适应'>
PLATFORM = '|'.join(platform.uname())
PLATFORM_SEVA100 = 'Linux|star-SYS-740GP-TNRT|6.5.0-18-generic|#18~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Wed Feb  7 11:40:03 UTC 2|x86_64|x86_64'
PLATFORM_LAPTOP = 'Windows|JD-DESKTOP|10|10.0.22631|AMD64|Intel64 Family 6 Model 154 Stepping 3, GenuineIntel'
PLATFORM_DESTOPLAB = ''
PLATFORM_SEV3090 = 'Linux|sescomputer3090|5.4.0-26-generic|#30-Ubuntu SMP Mon Apr 20 16:58:30 UTC 2020|x86_64|x86_64'
PLATFORM_SEV4090 = 'Linux|star-SYS-7049GP-TRT|5.15.0-43-generic|#46-Ubuntu SMP Tue Jul 12 10:30:17 UTC 2022|x86_64|x86_64'
PLATFORM_SEVTAITAN = 'Linux|sescomputer|5.4.0-94-generic|#106-Ubuntu SMP Thu Jan 6 23:58:14 UTC 2022|x86_64|x86_64'
PLATFORM_BOARD = ''
PLATFORM_SEVWWH = 'Linux|user-PowerEdge-XE9680|5.15.0-101-generic|#111~20.04.1-Ubuntu SMP Mon Mar 11 15:44:43 UTC 2024|x86_64|x86_64'

BROADCAST_ENV = False
GPU_DRIVER_VERSION = ''
GPUS = []
DEVICE = torch.device('cpu')
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_DRIVER_VERSION = pynvml.nvmlSystemGetDriverVersion().decode()
    for gpu_id in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        GPUS.append(pynvml.nvmlDeviceGetName(handle).decode())
except Exception as e:
    pass


# 得到GPU占用
def cuda_usage():
    num_cuda = pynvml.nvmlDeviceGetCount()
    usages = []
    for ind in range(num_cuda):
        handle = pynvml.nvmlDeviceGetHandleByIndex(ind)  # 0表示第一块显卡
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = meminfo.used / meminfo.total
        usages.append(usage)
    return usages


# 小于min_thres的都会被占用
# 上述不匹配，取最小一台，若该台占用小于one_thres则占用
def ditri_gpu(min_thres: float = 0.1, one_thres: float = 0.3) -> List[Union[int, None]]:
    usages = cuda_usage()
    if np.min(usages) < min_thres:
        inds = [int(i) for i in range(len(usages)) if usages[i] < min_thres]
    elif np.min(usages) < one_thres:
        inds = [int(np.argmin(usages))]
    else:
        BROADCAST('No free GPU, use CPU')
        return [None]
    # 交换顺序device_ids[0]第一个出现
    inds = sorted(inds, key=lambda x: usages[x])
    return inds


# 自动确定device
def select_device(device=None, min_thres: float = 0.01, one_thres: float = 0.5) -> List[Union[int, None]]:
    if device.__class__.__name__ == 'device':
        return [device.index]
    elif isinstance(device, int):
        return [device]
    elif device is None or len(device) == 0:
        BROADCAST('Auto select device')
        if not torch.cuda.is_available():
            BROADCAST('No available GPU, use CPU')
            return [None]
        else:
            inds = ditri_gpu(min_thres=min_thres, one_thres=one_thres)
            return inds
    elif isinstance(device, list) or isinstance(device, tuple):
        return list(device)
    elif isinstance(device, str) and len(device) > 0:
        return [torch.device(device).index]
    else:
        raise Exception('err device ' + str(device))


def ENVIR_MSG(width: int = 12, colorful: bool = True) -> List[str]:
    msgs = []
    item_names = ['System', 'Name', 'Version', 'Machine', 'Processor']
    items = platform.uname()
    for item_name, item in zip(item_names, items):
        item_name = item_name.ljust(width)
        item_name = stylize_msg(item_name, MSG_STYLE.BOLD, MSG_FG_COLOR.RED) if colorful else item_name
        msgs.append(item_name + item)
    if len(GPUS) > 0:
        item_name = 'GPU Driver'.ljust(width)
        item_name = stylize_msg(item_name, MSG_STYLE.BOLD, MSG_FG_COLOR.PURPLE) if colorful else item_name
        msgs.append(item_name + GPU_DRIVER_VERSION)
        for i, gpu in enumerate(GPUS):
            item_name = ('GPU %d' % i).ljust(width)
            item_name = stylize_msg(item_name, MSG_STYLE.BOLD, MSG_FG_COLOR.PURPLE) if colorful else item_name
            msgs.append(item_name + gpu)

    item_names = ['Python', 'Torch', 'TorchVision']
    items = [sys.version.replace('\n', ' '), torch.__version__, torchvision.__version__]
    for item_name, item in zip(item_names, items):
        item_name = item_name.ljust(width)
        item_name = stylize_msg(item_name, MSG_STYLE.BOLD, MSG_FG_COLOR.BLUE) if colorful else item_name
        msgs.append(item_name + item)
    return msgs


if BROADCAST_ENV:
    for msg in ENVIR_MSG(width=15, colorful=COLORFUL):
        BROADCAST(msg)


# </editor-fold>


# <editor-fold desc='图像读取'>
def load_img_pil(img_pth: str) -> Image.Image:
    try:
        img = Image.open(img_pth).convert('RGB')
        return img
    except Exception as e:
        print('File Err ' + img_pth)
        print(e)


def save_img_cv2(img_pth: str, imgN: np.ndarray) -> bool:
    cv2.imwrite(img_pth, imgN[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])  # BGR-RGB
    return True


def load_img_cv2(img_pth: str) -> np.ndarray:
    try:
        img = cv2.imread(img_pth)[..., ::-1]  # BGR-RGB
        return img
    except Exception as e:
        print('File Err ' + img_pth)
        print(e)


def reload_img_pil(img_pth_src: str, img_pth_dst: str) -> Image.Image:
    try:
        img = Image.open(img_pth_src).convert('RGB')
        img.save(img_pth_dst, quality=100)
        return img
    except Exception as e:
        print('File Err ' + img_pth_src)
        print(e)


def reload_img_cv2(img_pth_src: str, img_pth_dst: str) -> np.ndarray:
    try:
        img = cv2.imread(img_pth_src)
        cv2.imwrite(img_pth_dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return img
    except Exception as e:
        print('File Err ' + img_pth_src)
        print(e)


# </editor-fold>


# <editor-fold desc='文件路径编辑'>
# 规范后缀类型
def _stdlize_extend(extend: str) -> str:
    if isinstance(extend, str) and len(extend) > 0:
        return '.' + extend.replace('.', '')
    return ''


def _stdlize_extends(extends: Union[List[str], Tuple[str]]) -> List[str]:
    if isinstance(extends, list) or isinstance(extends, tuple):
        extends_new = [_stdlize_extend(extend) for extend in extends]
        return extends_new
    return []


# 规范文件类型
def ensure_folder_pth(file_pth: str) -> str:
    os.makedirs(file_pth, exist_ok=True)
    return file_pth


def remove_pth_ifexist(file_pth: str) -> None:
    if os.path.exists(file_pth):
        os.remove(file_pth)
    return None


def remove_pths_ifexist(file_pths: Iterable[str]) -> None:
    for file_pth in file_pths:
        remove_pth_ifexist(file_pth)
    return None


def ensure_file_dir(file_pth: str) -> str:
    ensure_folder_pth(os.path.dirname(file_pth))
    return file_pth


def ensure_extend(file_pth: str, extend: str = '', overwrite: bool = False) -> str:
    extend = _stdlize_extend(extend)
    if len(file_pth) > 0:
        file_pth_pure, extend_ori = os.path.splitext(file_pth)
        if overwrite or (not overwrite and len(extend_ori) == 0):
            return file_pth_pure + extend
        else:
            return file_pth
    else:
        return file_pth


def listdir_extend(file_dir: str, extends: Union[Sequence[str], str, None] = None) -> List[str]:
    file_names = os.listdir(file_dir)
    if extends is None:
        return file_names
    elif isinstance(extends, str):
        extends = _stdlize_extend(extends)
        return list(filter(lambda file_name: os.path.splitext(file_name)[1] == extends, file_names))
    elif isinstance(extends, list) or isinstance(extends, tuple):
        extends = _stdlize_extends(extends)
        return list(filter(lambda file_name: os.path.splitext(file_name)[1] in extends, file_names))
    raise Exception('err fmt ' + extends.__class__.__name__)


def listdir_recursive(file_dir: str, extends: Union[List[str], Tuple[str], str, None] = None,
                      recursive: bool = True) -> List[str]:
    file_names = os.listdir(file_dir)
    if isinstance(extends, str):
        extends = _stdlize_extend(extends)
    elif isinstance(extends, list) or isinstance(extends, tuple):
        extends = _stdlize_extends(extends)
    file_pths = []
    for file_name in file_names:
        file_pth = os.path.join(file_dir, file_name)
        if os.path.isdir(file_pth):
            if recursive:
                file_pths.extend(listdir_recursive(file_pth, extends=extends))
        elif extends is None:
            file_pths.append(file_pth)
        else:
            _, extend_ori = os.path.splitext(file_name)
            if isinstance(extends, str) and extend_ori == extends:
                file_pths.append(file_pth)
            elif (isinstance(extends, list) or isinstance(extends, tuple)) and extend_ori in extends:
                file_pths.append(file_pth)
    return file_pths


# </editor-fold>


# <editor-fold desc='多种文件读写'>

def save_dfdct2xlsx(file_pth: str, datas: Dict[str, pd.DataFrame], extend: str = 'xlsx') -> str:
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    writer = pd.ExcelWriter(file_pth)
    for name, data in datas.items():
        data.to_excel(writer, sheet_name=name, startcol=0, index=False)
    writer.close()
    return file_pth


# 写xls
def save_np2xlsx(file_pth: str, data: np.ndarray, extend: str = 'xlsx') -> str:
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    df = pd.DataFrame(data, columns=None, index=None)
    df.to_excel(file_pth, index=False, header=False)
    return file_pth


# 读xls
def load_xlsx2np(file_pth: str, extend: str = 'xlsx') -> np.ndarray:
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    data = pd.read_excel(file_pth, header=None, sheet_name=None)
    data = np.array(data)
    return data


# 保存对象
def save_pkl(file_pth: str, obj: object, extend: str = 'pkl'):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    with open(file_pth, 'wb+') as f:
        pickle.dump(obj, f)
    return None


# 读取对象
def load_pkl(file_pth: str, extend: str = 'pkl'):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    with open(file_pth, 'rb+') as f:
        obj = pickle.load(f)
    return obj


# 保存当前工作区
IGNORE_NAMES = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__builtins__']


def save_space(file_pth: str, locals: dict, extend: str = 'pkl'):
    save_dict = {}
    for name, val in locals.items():
        if callable(val):
            continue
        if name in IGNORE_NAMES:
            continue
        if type(val) == types.ModuleType:
            continue
        save_dict[name] = copy.copy(val)
    # 保存
    save_pkl(file_pth, save_dict, extend)
    return None


# 恢复工作区
def load_space(file_pth: str, locals: dict, extend: str = 'pkl'):
    save_dict = load_pkl(file_pth, extend)
    for name, val in save_dict.items():
        locals[name] = val


TXT_ENCODING = ('utf-8', 'gbk', 'utf-8-sig', 'gb18030', 'utf-16', 'ISO-8859-1')


def load_txt(file_pth: str, extend: str = 'txt', encoding: Union[Sequence[str], str] = TXT_ENCODING):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    if isinstance(encoding, str):
        encoding = [encoding]
    assert os.path.exists(file_pth), file_pth
    for ecd in encoding:
        try:
            with open(file_pth, 'r', encoding=ecd) as file:
                lines = file.readlines()
                # lines = lines.split('\n')
            lines = [line.replace('\n', '').replace('\r', '') for line in lines]
            return lines
        except UnicodeError as e:
            pass
    raise Exception('no enc match')


def save_txt(file_pth: str, lines: List[str], extend: str = 'txt', encoding: str = 'utf-8', append_mode: bool = False):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    lines_enter = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and not line.endswith('\r\n'):
            line = line + '\r\n'
        lines_enter.append(line)
    mode = 'a' if append_mode and os.path.exists(file_pth) else 'w'
    with open(file_pth, mode, encoding=encoding) as file:
        file.writelines(lines_enter)
    return None


def load_json(file_pth: str, extend: str = 'json', encoding: Union[Sequence[str], str] = TXT_ENCODING):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    if isinstance(encoding, str):
        encoding = [encoding]
    assert os.path.exists(file_pth)
    for ecd in encoding:
        try:
            with open(file_pth, 'r', encoding=ecd) as file:
                dct = json.load(file)
            return dct
        except Exception as e:
            pass
    raise Exception('no enc match')


def save_json(file_pth: str, dct, extend: str = 'json', indent=None, encoding: str = 'utf-8'):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    with open(file_pth, 'w', encoding=encoding) as file:
        json.dump(dct, fp=file, indent=indent, ensure_ascii=False)
    return None


def load_yaml(file_pth: str, extend: str = 'yaml', encoding: str = 'utf-8'):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    with open(file_pth, 'r', encoding=encoding) as file:
        dct = yaml.safe_load(file)
    return dct


def save_yaml(file_pth: str, dct, extend: str = 'yaml', indent=None, encoding: str = 'utf-8'):
    file_pth = ensure_extend(file_pth, extend=extend, overwrite=False)
    with open(file_pth, 'w', encoding=encoding) as file:
        yaml.dump(dct, file, indent=indent)
    return None


# </editor-fold>


# <editor-fold desc='多种文件读写'>
class MATCH_TYPE:
    FULL_NAME = 1
    LAST_NAME = 2
    SIZE = 4


# 简单按序匹配算法
def _match_array(arr1, arr2, cert=None):
    if cert is None:
        cert = lambda x, y: x == y
    num1, num2 = len(arr1), len(arr2)
    match_mat = np.full(shape=(num1, num2), fill_value=False)
    for i in range(num1):
        for j in range(num2):
            match_mat[i, j] = cert(arr1[i], arr2[j])
    match_pairs = []
    for s in range(num1 + num2):
        for i in range(s, -1, -1):
            j = s - i
            if i >= num1 or j >= num2 or j < 0:
                continue
                # 查找匹配
            if match_mat[i, j]:
                match_pairs.append((i, j))
                match_mat[i, :] = False
                match_mat[:, j] = False
                # print('Fit ',i,' --- ', j)
    return match_pairs


# 根据character匹配state_dict
def match_state_dict(sd_tar: dict, sd_ori: dict, match_type=MATCH_TYPE.FULL_NAME, verbose: bool = False):
    names_tar = list(sd_tar.keys())
    names_ori = list(sd_ori.keys())
    tensors_tar = list(sd_tar.values())
    tensors_ori = list(sd_ori.values())
    arr_tar = [[] for _ in range(len(names_tar))]
    arr_ori = [[] for _ in range(len(names_ori))]
    cert = lambda x, y: x == y
    characters = ''
    if MATCH_TYPE.SIZE | match_type == match_type:
        characters += 'size '
        for i in range(len(arr_tar)):
            arr_tar[i].append(tensors_tar[i].size())
        for i in range(len(arr_ori)):
            arr_ori[i].append(tensors_ori[i].size())
    if MATCH_TYPE.FULL_NAME | match_type == match_type:
        characters += 'full_name '
        for i in range(len(arr_tar)):
            arr_tar[i].append(names_tar[i])
        for i in range(len(arr_ori)):
            arr_ori[i].append(names_ori[i])
    if MATCH_TYPE.LAST_NAME | match_type == match_type:
        characters += 'last_name '
        for i in range(len(arr_tar)):
            arr_tar[i].append(names_tar[i].split('.')[-1])
        for i in range(len(arr_ori)):
            arr_ori[i].append(names_ori[i].split('.')[-1])
    if len(characters) == 0:
        raise Exception('Unknown match type ' + str(match_type))
    BROADCAST('Try to match by  [ ' + characters + ']')
    match_pairs = _match_array(arr_tar, arr_ori, cert)
    if verbose:
        BROADCAST('[ Matched pairs ] ' + str(len(match_pairs)))
        unmatched_tar = np.full(fill_value=True, shape=len(sd_tar))
        unmatched_ori = np.full(fill_value=True, shape=len(sd_ori))
        if len(match_pairs) > 0:
            indent_ori = max([len(names_ori[ind_ori]) for ind_tar, ind_ori in match_pairs]) + 4
            indent_tar = max([len(names_tar[ind_tar]) for ind_tar, ind_ori in match_pairs]) + 4
        else:
            indent_ori, indent_tar = 0, 0
        for ind_tar, ind_ori in match_pairs:
            msg_ori = names_ori[ind_ori].ljust(indent_ori, ' ') + ' %-20s' % str(tuple(tensors_ori[ind_ori].size()))
            msg_tar = names_tar[ind_tar].ljust(indent_tar, ' ') + ' %-20s' % str(tuple(tensors_tar[ind_tar].size()))
            BROADCAST(msg_ori + ' -> ' + msg_tar)
            unmatched_tar[ind_tar] = False
            unmatched_ori[ind_ori] = False
        if np.any(unmatched_tar):
            inds_tar = np.nonzero(unmatched_tar)[0]
            BROADCAST('[ Unmatched target ] ' + str(len(inds_tar)))
            for ind_tar in inds_tar:
                BROADCAST(
                    names_tar[ind_tar].ljust(indent_tar, ' ') + ' %-20s' % str(tuple(tensors_tar[ind_tar].size())))
        if np.any(unmatched_ori):
            inds_ori = np.nonzero(unmatched_ori)[0]
            BROADCAST('[ Unmatched source ] ' + str(len(inds_ori)))
            for ind_ori in inds_ori:
                BROADCAST(
                    names_ori[ind_ori].ljust(indent_ori, ' ') + ' %-20s' % str(tuple(tensors_ori[ind_ori].size())))
    return match_pairs


# 自定义state_dict加载
def load_fmt(model: nn.Module, sd_ori: dict, match_type=MATCH_TYPE.FULL_NAME,
             only_fullmatch: bool = False, verbose: bool = False, power: float = 1.0, broadcast: Callable = BROADCAST):
    if isinstance(sd_ori, str):
        device = next(iter(model.parameters())).device
        sd_ori = torch.load(sd_ori, map_location=device)
    sd_tar = model.state_dict()
    names_tar = list(sd_tar.keys())
    tensors_ori = list(sd_ori.values())
    # #匹配
    fit_pairs = match_state_dict(sd_tar, sd_ori, match_type=match_type, verbose=verbose)
    # 检查匹配结果
    broadcast('Source %d' % len(sd_ori) + ' Target %d' % len(names_tar) +
              ' Match %d' % len(fit_pairs) + ' Power %.2f' % power)
    if only_fullmatch and len(fit_pairs) < len(names_tar):
        broadcast('Not enough match')
        return False
    fit_sd = {}
    for i in range(len(fit_pairs)):
        i_tar, i_ori = fit_pairs[i]
        fit_sd[names_tar[i_tar]] = tensors_ori[i_ori]
    for name, tensor in sd_tar.items():
        if name not in fit_sd.keys():
            fit_sd[name] = tensor
    # 匹配添加
    power = max(min(power, 1), 0)
    for name, tensor in fit_sd.items():
        names = str.split(name, '.')
        tar = model
        for n in names:
            tar = getattr(tar, n)
        if tar.data.type == torch.float32 and power < 1:
            tar.data = power * tensor + (1 - power) * tar.data
        else:
            tar.data = tensor
    return True


# 调整模型通道显示
def refine_chans(model: nn.Module):
    def refine(model):
        if len(list(model.children())) == 0:
            if isinstance(model, nn.Conv2d):
                wei = model.weight
                model.in_channels = wei.size()[1]
                model.out_channels = wei.size()[0]
            elif isinstance(model, nn.BatchNorm2d):
                wei = model.weight
                model.num_features = wei.size()[0]
                model.running_mean = model.running_mean[:wei.size()[0]]
                model.running_var = model.running_var[:wei.size()[0]]
            elif isinstance(model, nn.Linear):
                wei = model.weight
                model.in_features = wei.size()[1]
                model.out_features = wei.size()[0]
        else:
            for name, sub_model in model.named_children():
                refine(sub_model)

    refine(model)
    return None
# </editor-fold>
