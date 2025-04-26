from models.modules import *
from models.modules import conv_auto_pad
from models.template import *
from utils import ps_int2_repeat


def kwargs_dispense(index: int, **kwargs) -> Dict:
    dct_ret = {}
    for k, v in kwargs.items():
        if isinstance(v, List):
            dct_ret[k] = v[index]
        else:
            dct_ret[k] = v
    return dct_ret


# <editor-fold desc='Drop'>
@REGISTER_CNTFLOP.ignore
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.p = p
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


# </editor-fold>

class ReShape(nn.Module):
    def __init__(self, size: Sequence[int]):
        nn.Module.__init__(self)
        self.size = tuple(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.size)

# <editor-fold desc='MLP'>


class MLP(nn.Sequential):

    def __init__(self, in_features: int, out_features: int, inner_featuress: Sequence[int] = (),
                 act=ACT.RELU, dropout: TV_Flt = 0.0, norm=NORM.LAYER, **kwargs):
        super().__init__()
        last_features = in_features
        for i, inner_features in enumerate(inner_featuress):
            self.add_module(str(i) + '_lin', nn.Linear(
                in_features=last_features, out_features=inner_features, bias=norm is not None))
            self.add_module(str(i) + '_norm', NORM_LN.build(features=inner_features, norm=norm, **kwargs))
            dropout_i = dropout[i] if isinstance(dropout, tuple) else dropout
            if dropout_i > 0.0:
                self.add_module(str(i) + '_drop', nn.Dropout(dropout_i))
            self.add_module(str(i) + '_act', ACT.build(act, **kwargs))
            last_features = inner_features
        self.add_module(str(len(inner_featuress)) + '_lin',
                        nn.Linear(in_features=last_features, out_features=out_features))


class MLC2d(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, inner_channelss: LTV_Int = (),
                 act=ACT.RELU, norm=NORM.BATCH, dilation: TV_Int2 = 1, ):
        super().__init__()
        inner_channelss = [inner_channelss] if isinstance(inner_channelss, int) else inner_channelss
        in_channelss = [in_channels] + list(inner_channelss)
        out_channelss = list(inner_channelss) + [out_channels]
        dilation = ps_int2_repeat(dilation)
        for i, (inc, outc) in enumerate(zip(in_channelss, out_channelss)):
            self.add_module(str(i) + '_conv', nn.Conv2d(
                in_channels=inc, out_channels=outc, kernel_size=(1, 1), stride=(1, 1), padding=0))
            if norm is not None and i < len(inner_channelss):
                self.add_module(str(i) + '_norm', NORM.build(channels=outc, norm=norm))
            if act is not None and i < len(inner_channelss):
                self.add_module(str(i) + '_str', ACT.build(act))


class MLC1d(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, inner_featuress: LTV_Int = (),
                 act=ACT.RELU, dropout=0.0):
        super().__init__()
        inner_featuress = [inner_featuress] if isinstance(inner_featuress, int) else inner_featuress
        in_featuress = [in_features] + list(inner_featuress)
        out_featuress = list(inner_featuress) + [out_features]
        for i, (inc, outc) in enumerate(zip(in_featuress, out_featuress)):
            self.add_module(str(i) + '_conv', nn.Conv1d(
                in_channels=inc, out_channels=outc, kernel_size=(1,), stride=(1,), padding=0))
            if i < len(inner_featuress):
                self.add_module(str(i) + '_norm', nn.BatchNorm1d(num_features=outc))
            if dropout > 0.0:
                self.add_module(str(i) + '_drop', nn.Dropout(dropout))
            if act is not None and i < len(inner_featuress):
                self.add_module(str(i) + '_act', ACT.build(act))


# </editor-fold>


# <editor-fold desc='注意力'>
class SEModule(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25, act=ACT.SIG):
        super(SEModule, self).__init__()
        inner_channels = int(ratio * channels)
        self.se_pth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=channels, out_features=inner_channels),
            nn.ReLU(),
            nn.Linear(in_features=inner_channels, out_features=channels),
            ACT.build(act),
        )

    def forward(self, x):
        return x * (self.se_pth(x)[..., None, None])


class SEModuleHalf(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25, act=ACT.HSIG):
        super(SEModuleHalf, self).__init__()

        inner_channels = int(ratio * channels)
        self.se_pth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=channels, out_features=inner_channels),
            ACT.build(act),
        )

    def forward(self, x):
        return x * (self.se_pth(x)[..., None, None])


# </editor-fold>

# <editor-fold desc='池化'>

class SPP(nn.Module):
    def __init__(self, kernels=(13, 9, 5), stride=1, shortcut=True):
        super(SPP, self).__init__()
        self.pools = nn.ModuleList()
        for kernel in kernels:
            padding = (kernel - 1) // 2
            self.pools.append(nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding))
        self.shortcut = shortcut

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outs = []
            for pool in self.pools:
                outs.append(pool(x))
            if self.shortcut:
                outs.append(x)
            outs = torch.cat(outs, dim=1)
        return outs


class SPPF(nn.Module):
    '''Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher'''

    def __init__(self, in_channels, out_channels, kernel_size=5, act=ACT.RELU,
                 norm=NORM.BATCH):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        inner_channels = in_channels // 2  # hidden channels
        self.redcr = Ck1s1NA(in_channels, inner_channels, act=act, norm=norm)
        self.concatr = Ck1s1NA(inner_channels * 4, out_channels, act=act, norm=norm)
        self.kernel_size = kernel_size
        self.pooler = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.redcr(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.pooler(x)
            y2 = self.pooler(y1)
            y3 = self.pooler(y2)
            return self.concatr(torch.cat((x, y1, y2, y3), dim=1))


class Focus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)


class PSP(nn.Module):
    def __init__(self, in_channels, out_channels, strides=(1, 2, 4, 8), act=ACT.RELU, norm=NORM.BATCH):
        super(PSP, self).__init__()
        num_stride = len(strides)
        self.strides = strides
        self.cvters = nn.ModuleList([
            Ck1s1NA(in_channels=in_channels, out_channels=out_channels // len(strides), act=act, norm=norm)
            for _ in range(num_stride)])
        self.mixor = Ck1s1A(in_channels=in_channels + out_channels, out_channels=out_channels, act=act)

    def forward(self, x):
        out = [x]
        for stride, cvter in zip(self.strides, self.cvters):
            x_i = F.max_pool2d(x, stride=stride, kernel_size=stride)
            x_i = cvter(x_i)
            x_i = F.interpolate(x_i, scale_factor=stride)
            out.append(x_i)
        out = torch.cat(out, dim=1)
        out = self.mixor(out)
        return out


# </editor-fold>

# <editor-fold desc='多特征处理'>
def _prase_lev_kwargs(lev, kwargs):
    kwargs_lev = {}
    for name, val in kwargs.items():
        if isinstance(val, list) or isinstance(val, tuple):
            kwargs_lev[name] = val[lev]
        else:
            kwargs_lev[name] = val
    return kwargs_lev


class BranchModule(nn.Module):
    def __init__(self, in_channels, out_channels, branch_num, **kwargs):
        super(BranchModule, self).__init__()
        assert branch_num >= 1, 'len err'
        self.modules = nn.ModuleList()
        for i in enumerate(range(branch_num)):
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            self.modules.append(self._build_module(i, in_channels, out_channels, **kwargs_i))

    @abstractmethod
    def _build_module(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass


class BranchModuleAdd(BranchModule):
    def __init__(self, in_channels, out_channels, branch_num, **kwargs):
        super(BranchModuleAdd, self).__init__(in_channels, out_channels, branch_num, **kwargs)

    def forward(self, feat):
        feat0 = self.modules[0](feat)
        for i in range(1, len(self.modules)):
            feat0 = feat0 + self.modules[i](feat)
        return feat0


class BranchModuleConcat(BranchModule):
    def __init__(self, in_channels, out_channels, branch_num, **kwargs):
        super(BranchModuleConcat, self).__init__(in_channels, out_channels, branch_num, **kwargs)

    def forward(self, feat):
        feats = []
        for i in range(len(self.modules)):
            feats.append(self.modules[i](feat))
        feats = torch.cat(feats, dim=1)
        return feats


class ParallelModule(nn.Module):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super(ParallelModule, self).__init__()
        self.modules = nn.ModuleList()
        for i, in_channels, out_channels in enumerate(zip(in_channelss, out_channelss)):
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            self.modules.append(self._build_module(i, in_channelss, out_channelss, **kwargs_i))

    @abstractmethod
    def _build_module(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feats):
        for i in range(len(feats)):
            feats[i] = self.modules[i](feats[i])
        return feats


class ParallelCpaBA(ParallelModule):

    def __init__(self, in_channelss, out_channelss, kernel_size=3, stride=1, dilation=1, groups=1, act=ACT.RELU,
                 norm=NORM.BATCH):
        super(ParallelCpaBA, self).__init__(
            in_channelss, out_channelss, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, act=act, norm=norm)

    def _build_module(self, lev, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                      act=ACT.RELU, norm=NORM.BATCH):
        return CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     dilation=dilation, groups=groups, act=act, norm=norm)


class CascadeModule(nn.Module):
    def __init__(self, in_channels, out_channelss, **kwargs):
        super(CascadeModule, self).__init__()
        self.modules = nn.ModuleList()
        last_channels = in_channels
        for i in range(len(out_channelss)):
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            self.modules.append(self._build_module(i, last_channels, out_channelss[i], **kwargs_i))

    @abstractmethod
    def _build_module(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feat):
        feats_out = []
        for i in range(len(self.modules)):
            feat = self.modules[i](feat)
            feats_out.append(feat)
        return feats_out


class CascadeCpaBA(ParallelModule):

    def __init__(self, in_channels, out_channelss, kernel_size=3, stride=1, dilation=1, groups=1, act=ACT.RELU,
                 norm=NORM.BATCH,
                 cross_c1=True):
        super(CascadeCpaBA, self).__init__(
            in_channels, out_channelss, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, act=act, norm=norm, cross_c1=cross_c1)

    def _build_module(self, lev, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                      act=ACT.RELU, norm=NORM.BATCH, cross_c1=True):
        kernel_size_lev = 1 if cross_c1 and lev % 2 == 0 else kernel_size
        return CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size_lev, stride=stride,
                     dilation=dilation, groups=groups, act=act, norm=norm)


class ParallelCpaBARepeat(ParallelModule):
    def __init__(self, in_channelss, out_channelss, kernel_size=3, stride=1, dilation=1, groups=1, act=ACT.RELU,
                 norm=NORM.BATCH,
                 num_repeat=1, cross_c1=True):
        super(ParallelCpaBARepeat, self).__init__(
            in_channelss, out_channelss, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, act=act, norm=norm, num_repeat=num_repeat, cross_c1=cross_c1)

    def _build_module(self, lev, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                      act=ACT.RELU, norm=NORM.BATCH, num_repeat=1, cross_c1=True):
        if num_repeat == 1:
            return CpaNA(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         dilation=dilation, groups=groups, act=act, norm=norm)
        else:
            convs = []
            for i in range(num_repeat):
                kernel_size_i = 1 if cross_c1 and i % 2 == 0 else kernel_size
                convs.append(CpaNA(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels,
                                   kernel_size=kernel_size_i, act=act, norm=norm))
                return nn.Sequential(*convs)


class MixStreamConcat(nn.Module):
    # feats_in[n]|-mixrs[n]->feats_out[n]-|
    #                                adprs[n-1]
    #                                   |
    # feats_in[n-1]|------------------[+]>-mixrs[n-1]->feats_out[n-1]

    def __init__(self, in_channelss, out_channelss, revsd=True, **kwargs):
        super(MixStreamConcat, self).__init__()
        num_lev = len(in_channelss)
        self.mixrs = nn.ModuleList([nn.Identity()] * num_lev)
        self.adprs = nn.ModuleList([nn.Identity()] * num_lev)
        self.revsd = revsd
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        out_channels_last = -1
        for i in iterator:
            in_channels = in_channelss[i]
            out_channels = out_channelss[i]
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            if out_channels_last == -1:
                adpr_channels = 0
            else:
                adpr_channels = self._adpr_channels(i, in_channels, out_channels, out_channels_last, **kwargs_i)
                self.adprs[i] = self._build_mixr(i, out_channels_last, adpr_channels, **kwargs_i)
            self.mixrs[i] = self._build_mixr(i, in_channels + adpr_channels, out_channels, **kwargs_i)
            out_channels_last = out_channels

    @abstractmethod
    def _adpr_channels(self, lev, in_channels, out_channels, out_channels_last, **kwargs) -> int:
        pass

    @abstractmethod
    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](torch.cat([feats[i], self.adprs[i](feat_buff)], dim=1))
            feats_out[i] = feat_buff
        return feats_out


class MixStreamAdd(nn.Module):
    # feats_in[n]|-mixrs[n]->feats_out[n]-|
    #                                adprs[n-1]
    #                                   |
    # feats_in[n-1]|------------------[+]>-mixrs[n-1]->feats_out[n-1]

    def __init__(self, in_channelss, out_channelss, revsd=True, **kwargs):
        super(MixStreamAdd, self).__init__()
        num_lev = len(in_channelss)
        self.mixrs = nn.ModuleList([nn.Identity()] * num_lev)
        self.adprs = nn.ModuleList([nn.Identity()] * num_lev)
        self.revsd = revsd
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        out_channels_last = -1
        for i in iterator:
            in_channels = in_channelss[i]
            out_channels = out_channelss[i]
            kwargs_i = _prase_lev_kwargs(i, kwargs)
            if not out_channels_last == -1:
                self.adprs[i] = self._build_mixr(i, out_channels_last, in_channels, **kwargs_i)
            self.mixrs[i] = self._build_mixr(i, in_channels, out_channels, **kwargs_i)
            out_channels_last = out_channels

    @abstractmethod
    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        pass

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = self.mixrs[i](feats[i] + self.adprs[i](feat_buff))
            feats_out[i] = feat_buff
        return feats_out


class DownStreamConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)


class DownStreamConcatSamp(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, mode='nearest', scale_factor=2, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = F.upsample(self.adprs[i](feat_buff), scale_factor=self.scale_factor, mode=self.mode)
                feat_buff = self.mixrs[i](torch.cat([feats[i], feat_buff], dim=1))
            feats_out[i] = feat_buff
        return feats_out


class UpStreamConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)


class DownStreamCk1s1BAConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)

    def _adpr_channels(self, lev, in_channels, out_channels, out_channels_last, **kwargs) -> int:
        return out_channels_last

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


class UpStreamCk1s1BAConcat(MixStreamConcat):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)

    def _adpr_channels(self, lev, in_channels, out_channels, out_channels_last, **kwargs) -> int:
        return out_channels_last

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


class DownStreamAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)


class DownStreamAddSamp(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, mode='nearest', scale_factor=2, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, feats):
        num_lev = len(feats)
        feat_buff = None
        feats_out = [None] * num_lev
        iterator = range(num_lev)
        iterator = reversed(iterator) if self.revsd else iterator
        for i in iterator:
            if feat_buff is None:
                feat_buff = self.mixrs[i](feats[i])
            else:
                feat_buff = F.upsample(self.adprs[i](feat_buff), scale_factor=self.scale_factor, mode=self.mode)
                feat_buff = self.mixrs[i](feats[i] + feat_buff)
            feats_out[i] = feat_buff
        return feats_out


class UpStreamAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)


class DownStreamCk1s1BAAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=True, **kwargs)

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


class UpStreamCk1s1BAAdd(MixStreamAdd):
    def __init__(self, in_channelss, out_channelss, **kwargs):
        super().__init__(in_channelss, out_channelss, revsd=False, **kwargs)

    def _build_adpr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)

    def _build_mixr(self, lev, in_channels, out_channels, **kwargs) -> nn.Module:
        return Ck1s1NA(in_channels=in_channels, out_channels=out_channels)


# </editor-fold>


if __name__ == '__main__':
    x1 = torch.zeros(size=(1, 5, 6, 6))
    x2 = torch.zeros(size=(1, 4, 6, 6))
    x3 = torch.zeros(size=(1, 3, 6, 6))
    feats = (x1, x2, x3)

    model = DownStreamAdd((5, 4, 3), (2, 4, 6))

    torch.onnx.export(model, (feats,), './test.onnx', opset_version=11)
    # y = model(feats)
