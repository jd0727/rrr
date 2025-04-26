import collections
from abc import ABCMeta, abstractmethod
from collections.abc import Callable

try:
    import onnxruntime
except Exception as e:
    pass
from sklearn.cluster import KMeans
from torch.onnx import OperatorExportTypes

from .interface import HasImageSize
from .define import *
from .iotools import *
from .typings import *


# <editor-fold desc='模型统计'>
class RegisterCntFlop(OrderedDict):

    def registry(self, *tags, cnt_once: bool = True):
        def wrapper(_cnter):
            for tag in tags:
                self[tag] = (_cnter, cnt_once)
            return _cnter

        return wrapper

    def ignore(self, module):
        self[module] = None
        return module


REGISTER_CNTFLOP = RegisterCntFlop(
    {
        nn.SiLU: None,
        nn.Sigmoid: None,
        nn.AvgPool2d: None,
        nn.AdaptiveAvgPool2d: None,
        nn.ReLU: None,
        nn.MaxPool2d: None,
        nn.Dropout: None,
        nn.Dropout2d: None,
        nn.CrossEntropyLoss: None,
        nn.UpsamplingNearest2d: None,
        nn.LeakyReLU: None,
        nn.UpsamplingBilinear2d: None,
        nn.Identity: None,
        nn.GELU: None,
        nn.Flatten: None
    }
)


def _count_nonzero(data: torch.Tensor, dim: int = 0) -> int:
    if not dim == 0:
        data = data.transpose(dim, 0).contiguous()
    sum_val = torch.sum(torch.abs(data.reshape(data.shape[0], -1)), dim=1)
    num = torch.count_nonzero(sum_val).item()
    return num


class CntFlop(metaclass=ABCMeta):
    def __init__(self, module_name, type_name, msgs, **kwargs):
        self.module_name = module_name
        self.type_name = type_name
        self.msgs = msgs
        self.kwargs = {}

    @abstractmethod
    def __call__(self, module, data_input, data_output, ):
        pass

    def update(self, flop: int, **kwargs):
        msg = dict(FLOP=flop, Name=self.module_name, Class=self.type_name, **kwargs)
        self.msgs.append(msg)


@REGISTER_CNTFLOP.registry(nn.Conv2d)
class CntFlopConv2d(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output, ):
        _, ci, _, _ = list(data_input[0].size())
        _, co, ho, wo = list(data_output.size())
        _, _, hk, wk = list(module.weight.size())
        groups = module.groups
        if not self.ignore_zero:
            ci = _count_nonzero(data_input[0], dim=1)
            co = _count_nonzero(module.weight, dim=0)
        flop = hk * wk * ci * co * ho * wo // groups
        if module.bias is not None:
            flop += co * ho * wo
        size_str = '(%d' % co + '[%d]' % groups + ',%d' % ci + ',%d' % hk + ',%d' % wk + ')'
        self.update(flop=int(flop), Size=size_str, Output=str((wo, ho)))


@REGISTER_CNTFLOP.registry(nn.ConvTranspose2d)
class CntFlopConvTranspose2d(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output, ):
        _, ci, hi, wi = list(data_input[0].size())
        _, co, ho, wo = list(data_output.size())
        _, _, hk, wk = list(module.weight.size())
        groups = module.groups
        if not self.ignore_zero:
            ci = _count_nonzero(data_input[0], dim=1)
            co = _count_nonzero(module.weight, dim=0)
        flop = hk * wk * ci * co * hi * wi // groups
        if module.bias is not None:
            flop += co * hi * wi
        size_str = '(%d' % co + '[%d]' % groups + ',%d' % ci + ',%d' % hk + ',%d' % wk + ')'
        self.update(flop=int(flop), Size=size_str, Output=str((wo, ho)))


@REGISTER_CNTFLOP.registry(nn.BatchNorm2d)
class CntFlopBatchNorm2d(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output):
        _, c, ho, wo = data_input[0].shape
        if not self.ignore_zero:
            c = _count_nonzero(module.weight.data)
        flop = c * ho * wo
        self.update(Size=str((c,)), flop=flop, Output=str((wo, ho)))


@REGISTER_CNTFLOP.registry(nn.LayerNorm)
class CntFlopLayerNorm(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output):
        l = int(np.prod(data_input[0].size()[1:-1]))
        c = data_input[0].size(-1)
        if not self.ignore_zero:
            c = _count_nonzero(module.weight.data, dim=0)
        flop = c * l
        self.update(Size=str((c,)), flop=flop, Output=str((l,)))


@REGISTER_CNTFLOP.registry(nn.GroupNorm)
class CntFlopGroupNorm(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output):
        _, c, hf, wf = list(data_input[0].shape)
        if not self.ignore_zero:
            c = _count_nonzero(module.weight.data)
        flop = c * hf * wf
        self.update(Size=str((c,)), flop=flop, Output=str((wf, hf)))


@REGISTER_CNTFLOP.registry(nn.Linear)
class CntFlopLinear(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output):
        co, ci = module.weight.shape
        if not self.ignore_zero:
            ci = _count_nonzero(data_input[0], dim=-1)
            co = _count_nonzero(data_output, dim=-1)
        l = int(np.prod(data_input[0].size()[1:-1]))
        flop = l * ci * co
        if not module.bias is None:
            flop += l * co
        self.update(Size=str((co, ci)), flop=flop, Output=str((l,)))


@REGISTER_CNTFLOP.registry(nn.AdaptiveAvgPool2d)
class CntFlopAdaptiveAvgPool2d(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output):
        _, c, fhi, fwi = data_input[0].shape
        _, _, fho, fwo = data_output.shape
        if not self.ignore_zero:
            c = _count_nonzero(data_output, dim=1)
        flop = int(max(fhi, fho) * max(fwi, fwo) * c)
        self.update(Size=str((c,)), flop=flop, Output=str((fwo, fho)))


@REGISTER_CNTFLOP.registry(nn.AdaptiveAvgPool1d)
class CntFlopAdaptiveAvgPool1d(CntFlop):
    def __init__(self, module_name, type_name, msgs, ignore_zero=False, **kwargs):
        super().__init__(module_name, type_name, msgs, **kwargs)
        self.ignore_zero = ignore_zero

    def __call__(self, module, data_input, data_output):
        _, c, li = data_input[0].shape
        _, _, lo = data_output.shape
        if not self.ignore_zero:
            c = _count_nonzero(data_output, dim=1)
        flop = (1 + (li // lo)) * lo * c
        self.update(Size=str((c,)), flop=flop, Output=str((lo,)))


def count_delay(model: nn.Module, args: tuple, num_iter: int = 10, num_warmup: int = 1, device=DEVICE):
    time1 = 0
    model = model.eval()
    with torch.no_grad():
        for i in range(num_iter + num_warmup):
            _ = model(*args)
            if i == num_warmup - 1:
                if device.index is not None:
                    torch.cuda.synchronize(device)
                time1 = time.time()
    if device.index is not None:
        torch.cuda.synchronize(device)
    time2 = time.time()
    time_aver = (time2 - time1) / num_iter
    return time_aver


def count_func_delay(func: Callable, args: tuple, num_iter: int = 10, num_warmup: int = 1, device=DEVICE):
    time1 = 0
    with torch.no_grad():
        for i in range(num_iter + num_warmup):
            _ = func(*args)
            if i == num_warmup - 1:
                if device.index is not None:
                    torch.cuda.synchronize(device)
                time1 = time.time()
    if device.index is not None:
        torch.cuda.synchronize(device)
    time2 = time.time()
    time_aver = (time2 - time1) / num_iter
    return time_aver


def count_flop(model: nn.Module, args: tuple, ignore_zero: bool = True):
    assert isinstance(model, nn.Module), 'mdoel err'
    msgs = []
    handles = []

    # 添加hook
    def add_hook(module, module_name=None):
        if module.__class__ in REGISTER_CNTFLOP.keys():
            hook_type_once = REGISTER_CNTFLOP[module.__class__]
            if hook_type_once is not None:
                hook_type, cnt_once = hook_type_once
                assert issubclass(hook_type, CntFlop), 'hook err'
                hook = hook_type(module_name=module_name, type_name=module.__class__.__name__,
                                 ignore_zero=ignore_zero, msgs=msgs)
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                if cnt_once:
                    return None
            else:
                return None
        elif len(list(module.children())) == 0:
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
                pass
            else:
                print(module.__class__.__name__, 'not support')
            return None
        for name, sub_model in module.named_children():
            sub_model_name = name if module_name is None else module_name + '.' + name
            add_hook(sub_model, sub_model_name)

    # 规范输入
    model = model.eval()
    with torch.no_grad():
        add_hook(model, None)
        _ = model(*args)
    # 移除hook
    for handle in handles:
        handle.remove()
    order = ['Name', 'Class', 'Size', 'Output', 'FLOP']
    data = pd.DataFrame(columns=order)
    for i, msg in enumerate(msgs):
        data = pd.concat([data, pd.DataFrame(msg, index=[i])])
    return data


def count_para(model):
    assert isinstance(model, nn.Module), 'mdoel err ' + model.__class__.__name__
    count_para.data = pd.DataFrame(columns=['Name', 'Class', 'Para'])

    def stat_para(module, module_name=None):
        if len(list(module.children())) == 0:
            para_sum = 0
            for para in module.parameters():
                para_sum += para.numel()
            row = pd.DataFrame({
                'Name': module_name,
                'Class': module.__class__.__name__,
                'Para': para_sum
            }, index=[0])
            count_para.data = pd.concat([count_para.data, row])
        else:
            for name, sub_model in module.named_children():
                sub_model_name = name if module_name is None else module_name + '.' + name
                stat_para(sub_model, module_name=sub_model_name)

    stat_para(model, None)
    data = count_para.data
    return data


# </editor-fold>

# <editor-fold desc='数据分析'>


def analyse_cens(whs, centers, whr_thres=4):
    n_clusters = centers.shape[0]
    ratios = whs[:, None, :] / centers[None, :, :]
    ratios = np.max(np.maximum(ratios, 1 / ratios), axis=2)
    markers = ratios < whr_thres
    matched = np.sum(np.any(markers, axis=1))
    aver_mtch = np.mean(np.sum(markers, axis=1))
    # 输出
    print('* Centers --------------')
    for i in range(n_clusters):
        width, height = centers[i, :]
        print('[ %5d' % int(width) + ' , %5d' % int(height) + ' ] --- ' + str(np.sum(markers[:, i])))
    print('* Boxes --------------')
    print('Matched ' + '%5d' % int(matched) + ' / %5d' % int(whs.shape[0]))
    print('Average ' + '%.2f' % aver_mtch + ' box per obj')
    print('* -----------------------')
    return True


def cluster_wh(whs, n_clusters=9, log_metric=True):
    if log_metric:
        whs_log = np.log(whs)
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(whs_log)
        centers_log = kmeans_model.cluster_centers_
        centers = np.exp(centers_log)
    else:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=1).fit(whs)
        centers = kmeans_model.cluster_centers_
    areas = centers[:, 0] * centers[:, 1]
    order = np.argsort(areas)
    centers_sorted = centers[order]
    return centers_sorted


# </editor-fold>


# 得到device
def get_device(model):
    if hasattr(model, 'device'):
        return model.device
    else:
        if len(model.state_dict()) > 0:
            return next(iter(model.parameters())).device
        else:
            return torch.device('cpu')


# 规范4维输入
def _get_input_size(img_size: TV_Int2 = (32, 32), in_channels: int = 3, batch_size: int = 1):
    img_size = ps_int2_repeat(img_size)
    input_size = (batch_size, in_channels, img_size[1], img_size[0])
    return input_size


def count_delay_image(model: nn.Module, img_size: TV_Int2, num_iter: int = 10, num_warmup: int = 1, in_channels=3,
                      batch_size=1, device=DEVICE):
    input_size = _get_input_size(img_size=img_size, in_channels=in_channels, batch_size=batch_size)
    img = torch.rand(input_size, device=device)
    delay = count_delay(model, (img,), num_iter=num_iter, num_warmup=num_warmup, device=device)
    return delay / batch_size


def count_flop_image(model: nn.Module, img_size: TV_Int2, in_channels=3,
                     batch_size=1, device=DEVICE, ignore_zero: bool = True):
    input_size = _get_input_size(img_size=img_size, in_channels=in_channels, batch_size=batch_size)
    img = torch.rand(input_size, device=device)
    flop = count_flop(model, (img,), ignore_zero=ignore_zero)
    return flop


def model2onnx(onnx_pth, model, img_size: TV_Int2, in_channels=3, batch_size=1, **kwargs):
    onnx_dir = os.path.dirname(onnx_pth)
    if not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir)
    onnx_pth = onnx_pth + '.onnx' if not str.endswith(onnx_pth, '.onnx') else onnx_pth
    input_size = _get_input_size(img_size=img_size, in_channels=in_channels, batch_size=batch_size)
    # 仅支持单输入单输出
    input_names = ['input']
    output_names = ['output']
    dynamic_batch = input_size[0] is None or input_size[0] < 0
    if dynamic_batch:
        input_size = list(input_size)
        input_size[0] = 1
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        print('Using dynamic batch size')
    else:
        dynamic_axes = None
        print('Exporting static batch size')
    test_input = (torch.rand(size=input_size) - 0.5) * 4
    test_input = test_input.to(get_device(model))
    model.eval()
    print('Exporting onnx to ' + onnx_pth)
    torch.onnx.export(model, test_input, onnx_pth, verbose=True, opset_version=11,
                      operator_export_type=OperatorExportTypes.ONNX, do_constant_folding=True,
                      input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    return True


# <editor-fold desc='计数'>

# </editor-fold>

# <editor-fold desc='规范化导出接口'>

def _onnx_sim(onnx_pth):
    import onnx
    from onnxsim import simplify
    model_onnx = onnx.load(onnx_pth)  # load onnx model

    model_onnx, check = simplify(model_onnx)
    onnx.save(model_onnx, onnx_pth)
    return True


def _onnx_slim(onnx_pth):
    import onnx
    import onnxslim
    model_onnx = onnx.load(onnx_pth)  # load onnx model
    model_onnx = onnxslim.slim(model_onnx)

    onnx.save(model_onnx, onnx_pth)
    return True


class ONNXExportable(nn.Module):

    @property
    @abstractmethod
    def input_names(self):
        pass

    @property
    @abstractmethod
    def output_names(self):
        pass

    @property
    @abstractmethod
    def input_sizes(self):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def export_onnx(self, onnx_pth, dynamic_batch=False, opset_version=17, with_slim=False):
        onnx_pth = ensure_extend(onnx_pth, 'onnx')
        ensure_file_dir(onnx_pth)
        input_names = self.input_names
        output_names = self.output_names
        input_sizes = self.input_sizes
        device = self.device
        input_sizes = [[1] + list(input_size) for input_size in input_sizes]
        input_tens = tuple([torch.rand(input_size, device=device) for input_size in input_sizes])

        msg = 'Exporting onnx to ' + onnx_pth + ' |'
        if dynamic_batch:
            dynamic_axes = {}
            for input_name in input_names:
                dynamic_axes[input_name] = {0: 'batch_size'}

            for output_name in output_names:
                dynamic_axes[output_name] = {0: 'batch_size'}
            msg += ' <dynamic> batch |'
        else:
            dynamic_axes = None
            msg += ' <static> batch |'

        msg += ' ' + str(input_sizes) + ' |'
        print(msg)
        torch.onnx.export(self, input_tens, onnx_pth, verbose=False, opset_version=opset_version,
                          operator_export_type=OperatorExportTypes.ONNX, do_constant_folding=True,
                          input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
        if with_slim:
            print('Simplify Onnx')
            # _onnx_sim(onnx_pth)
            _onnx_slim(onnx_pth)
        return self

    def count_flop(self, ignore_zero=True, batch_size: int = 1):
        device = self.device
        input_sizes = [[batch_size] + list(input_size) for input_size in self.input_sizes]
        input_tens = tuple([torch.rand(input_size, device=device) for input_size in input_sizes])
        data = count_flop(self, input_tens, ignore_zero=ignore_zero)
        return data

    def count_delay(self, num_iter: int = 10, num_warmup: int = 1, batch_size: int = 1):
        device = self.device
        input_sizes = [[batch_size] + list(input_size) for input_size in self.input_sizes]
        input_tens = tuple([torch.rand(input_size, device=device) for input_size in input_sizes])
        delay = count_delay(self, input_tens, num_iter=num_iter, num_warmup=num_warmup, device=device) / batch_size
        return delay

    def count_para(self):
        return count_para(self)


class SISOONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['input']

    @property
    def output_names(self):
        return ['output']


class SequenceONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['sequence']

    @property
    def output_names(self):
        return ['output']

    @property
    @abstractmethod
    def length(self):
        pass

    @property
    @abstractmethod
    def in_features(self):
        pass

    @property
    def input_sizes(self):
        return [(self.length, self.in_features)]


class ImageONNXExportable(ONNXExportable, HasImageSize):

    @property
    def input_names(self):
        return ['image']

    @property
    def output_names(self):
        return ['output']

    @property
    @abstractmethod
    def in_channels(self) -> int:
        pass

    @property
    def input_sizes(self) -> Tuple[Tuple[int, ...]]:
        return ((self.in_channels, self.img_size[1], self.img_size[0]),)


class GeneratorONNXExportable(ONNXExportable):

    @property
    def input_names(self):
        return ['latvecs']

    @property
    def output_names(self):
        return ['fimages']

    @property
    def input_sizes(self):
        return [(self.in_features,)]

    @property
    @abstractmethod
    def in_features(self):
        pass


# </editor-fold>


# <editor-fold desc='onnx读取工具'>

class ONNXModule():
    def __init__(self, onnx_pth, device=None):
        onnx_pth = onnx_pth + '.onnx' if not str.endswith(onnx_pth, '.onnx') else onnx_pth
        device_ids = select_device(device)
        if device_ids[0] is None:
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_pth, providers=['CPUExecutionProvider'])
        else:
            self.onnx_session = onnxruntime.InferenceSession(
                onnx_pth, providers=['CUDAExecutionProvider'], provider_options=[{'device_id': str(device_ids[0])}])

    @property
    def output_names(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    @property
    def input_names(self):
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name


class OneStageONNXModule(ONNXModule):
    def __init__(self, onnx_pth, device=None):
        super().__init__(onnx_pth=onnx_pth, device=device)
        inputs = self.onnx_session.get_inputs()
        outputs = self.onnx_session.get_outputs()
        assert len(inputs) == 1, 'fmt err'
        assert len(outputs) == 1, 'fmt err'
        self.input_size = inputs[0].shape
        self.output_size = outputs[0].shape
        print('ONNXModule from ' + onnx_pth + ' * input ' + str(inputs[0].shape) + ' * output ' + str(outputs[0].shape))

    def __call__(self, input, **kwargs):
        input_feed = {self.input_names[0]: input}
        outputs = self.onnx_session.run(self.output_names, input_feed=input_feed)
        output = outputs[0]
        return output
# </editor-fold>
