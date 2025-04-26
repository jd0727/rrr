from models.baseline.define import *


# CBA+CBA+Res
class DarkNetResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, inner_channels: int, stride: int, act=ACT.LK,
                 norm=NORM.BATCH, **kwargs):
        nn.Module.__init__(self)
        self.conv1 = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, act=act, norm=norm)
        self.conv2 = Ck3NA(in_channels=inner_channels, out_channels=out_channels, stride=stride, act=act, norm=norm)
        if stride > 1:
            self.shortcut = None
        elif in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Ck1s1(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class DarkNetResidualSE(DarkNetResidual):
    def __init__(self, in_channels: int, out_channels: int, inner_channels: int, stride: int, act=ACT.LK,
                 norm=NORM.BATCH, **kwargs):
        DarkNetResidual.__init__(self, in_channels=in_channels, out_channels=out_channels,
                                 inner_channels=inner_channels, stride=stride, act=act, norm=norm, **kwargs)
        self.se = SEModule(channels=inner_channels, ratio=0.25)

    def forward(self, x):
        out = self.conv2(self.se(self.conv1(x)))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class DarkNetBkbn(nn.Module):
    MODEULE = DarkNetResidual
    SAMP = Ck3NA

    # SAMP = None

    def __init__(self, channelss: Sequence[int], nums_repeat: Sequence[int], strides: Sequence[int],
                 act=ACT.LK, norm=NORM.BATCH, in_channels: int = 3, **kwargs):
        nn.Module.__init__(self)
        self.pre = Ck3s1NA(in_channels=in_channels, out_channels=channelss[0] // 2, act=act, norm=norm)
        self.stages = nn.ModuleList([])
        self.samps = nn.ModuleList([])
        for i in range(len(channelss)):
            if i == 0:
                in_channels_cur = channelss[0] // 2
            else:
                in_channels_cur = channelss[i - 1]
            stride = strides[i]
            if self.SAMP is None:
                smap = nn.Identity()
            else:
                smap = self.SAMP(in_channels=in_channels_cur, out_channels=channelss[i], stride=stride,
                                 act=act, norm=norm, **kwargs_dispense(index=i, **kwargs))
                in_channels_cur = channelss[i]
                stride = 1
            self.samps.append(smap)
            self.stages.append(self.ModuleRepeat(
                Module=self.MODEULE, in_channels=in_channels_cur, out_channels=channelss[i],
                num_repeat=nums_repeat[i], stride=stride, act=act, norm=norm, **kwargs_dispense(index=i, **kwargs)))

    @staticmethod
    def ModuleRepeat(Module: nn.Module, in_channels: int, out_channels: int, num_repeat: int = 1, stride: int = 2,
                     act=ACT.LK, norm=NORM.BATCH, **kwargs):
        if num_repeat == 0:
            return nn.Identity()
        bkbn = []
        for i in range(num_repeat):
            if i == 0:
                last_channels = in_channels
            else:
                last_channels = out_channels
                stride = 1
            bkbn.append(Module(in_channels=last_channels, out_channels=out_channels, stride=stride,
                               inner_channels=out_channels // 2, act=act, norm=norm,
                               **kwargs_dispense(index=i, **kwargs)))
        return nn.Sequential(*bkbn)

    def forward(self, imgs):
        feats = self.pre(imgs)
        for i in range(len(self.stages)):
            feats = self.samps[i](feats)
            feats = self.stages[i](feats)
        return feats

    PARA_R53 = dict(channelss=(64, 128, 256, 512, 1024),
                    nums_repeat=(1, 2, 8, 8, 4), strides=(2, 2, 2, 2, 2))

    @classmethod
    def R53(cls, in_channels: int = 3, act=ACT.RELU, norm=NORM.BATCH):
        return cls(**cls.PARA_R53, act=act, norm=norm, in_channels=in_channels)


# ConvResidualRepeat+CBA+Res
class CSPBlockV4(nn.Module):

    def __init__(self, Module: nn.Module, in_channels: int, out_channels: int, shortcut_channels: int,
                 backbone_channels: int, backbone_inner_channels: int,
                 num_repeat: int, stride: int = 1, act=ACT.LK, norm=NORM.BATCH, **kwargs):
        nn.Module.__init__(self)
        assert stride == 1
        self.shortcut = Ck1s1NA(in_channels=in_channels, out_channels=shortcut_channels, act=act, norm=norm)
        backbone = [Ck1s1NA(in_channels=in_channels, out_channels=backbone_channels, act=act, norm=norm)]
        for i in range(num_repeat):
            backbone.append(Module(
                in_channels=backbone_channels, out_channels=backbone_channels, stride=1,
                inner_channels=backbone_inner_channels, act=act, norm=norm))
        backbone.append(Ck1s1NA(in_channels=backbone_channels, out_channels=backbone_channels, act=act, norm=norm))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1NA(in_channels=shortcut_channels + backbone_channels, out_channels=out_channels, act=act,
                                norm=norm)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


# ConvResidualRepeat+CBA+Res
class CSPBlockV5(nn.Module):

    def __init__(self, Module: nn.Module, in_channels: int, out_channels: int, shortcut_channels: int,
                 backbone_channels: int, backbone_inner_channels: int,
                 num_repeat: int, stride: int = 1, act=ACT.LK, norm=NORM.BATCH, **kwargs):
        nn.Module.__init__(self)
        assert stride == 1
        self.shortcut = Ck1s1NA(in_channels=in_channels, out_channels=shortcut_channels, act=act, norm=norm)
        backbone = [Ck1s1NA(in_channels=in_channels, out_channels=backbone_channels, act=act, norm=norm)]
        for i in range(num_repeat):
            backbone.append(Module(
                in_channels=backbone_channels, out_channels=backbone_channels, stride=1,
                inner_channels=backbone_inner_channels, act=act, norm=norm))
        self.backbone = nn.Sequential(*backbone)
        self.concater = Ck1s1NA(in_channels=shortcut_channels + backbone_channels, out_channels=out_channels, act=act,
                                norm=norm)

    def forward(self, x):
        xc = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        xc = self.concater(xc)
        return xc


class DarkNetV4Bkbn(nn.Module):
    MODULE = DarkNetResidual
    CSPBLOCK = CSPBlockV4
    SAMP = Ck3NA
    PRE = Ck3s1NA

    def __init__(self, channelss: Sequence[int], nums_repeat: Sequence[int], strides: Sequence[int],
                 act=ACT.LK, norm=NORM.BATCH, in_channels: int = 3, **kwargs):
        nn.Module.__init__(self)
        if self.PRE is not None:
            self.pre = self.PRE(in_channels=in_channels, out_channels=channelss[0] // 2, act=act, norm=norm)
        else:
            self.pre = nn.Identity()
        self.stages = nn.ModuleList([])
        self.samps = nn.ModuleList([])
        for i in range(len(channelss)):
            if i == 0:
                in_channels_cur = channelss[0] // 2
            else:
                in_channels_cur = channelss[i - 1]
            stride = strides[i]
            if self.SAMP is None:
                smap = nn.Identity()
            else:
                smap = self.SAMP(in_channels=in_channels_cur, out_channels=channelss[i], stride=stride,
                                 act=act, norm=norm, **kwargs_dispense(index=i, **kwargs))
                in_channels_cur = channelss[i]
                stride = 1
            self.samps.append(smap)
            out_channels_cur = channelss[i]
            self.stages.append(self.CSPBLOCK(
                Module=self.MODULE, in_channels=in_channels_cur, out_channels=out_channels_cur,
                shortcut_channels=out_channels_cur // 2, backbone_inner_channels=out_channels_cur // 2,
                num_repeat=nums_repeat[i], backbone_channels=out_channels_cur // 2,
                stride=stride, act=act, norm=norm, **kwargs_dispense(index=i, **kwargs)))

    def forward(self, imgs):
        feats = self.pre(imgs)
        for i in range(len(self.stages)):
            feats = self.samps[i](feats)
            feats = self.stages[i](feats)
        return feats

    PARA_R53 = dict(channelss=(64, 128, 256, 512, 1024),
                    nums_repeat=(1, 2, 8, 8, 4), strides=(2, 2, 2, 2, 2))

    PARA_NANO = dict(channelss=(32, 64, 128, 256), nums_repeat=(1, 2, 3, 1), strides=(2, 2, 2, 2))
    PARA_SMALL = dict(channelss=(64, 128, 256, 512), nums_repeat=(1, 2, 3, 1), strides=(2, 2, 2, 2))
    PARA_MEDIUM = dict(channelss=(96, 192, 384, 768), nums_repeat=(2, 4, 6, 2), strides=(2, 2, 2, 2))
    PARA_LARGE = dict(channelss=(128, 256, 512, 1024), nums_repeat=(3, 6, 9, 3), strides=(2, 2, 2, 2))
    PARA_XLARGE = dict(channelss=(160, 320, 640, 1280), nums_repeat=(4, 8, 12, 4), strides=(2, 2, 2, 2))

    @classmethod
    def R53(cls, in_channels: int = 3, act=ACT.RELU, norm=NORM.BATCH):
        return cls(**cls.PARA_R53, act=act, norm=norm, in_channels=in_channels)

    @classmethod
    def Nano(cls, act=ACT.RELU, norm=NORM.BATCH, in_channels: int = 3):
        return cls(**cls.PARA_NANO, act=act, norm=norm, in_channels=in_channels)

    @classmethod
    def Small(cls, act=ACT.RELU, norm=NORM.BATCH, in_channels: int = 3):
        return cls(**cls.PARA_SMALL, act=act, norm=norm, in_channels=in_channels)

    @classmethod
    def Medium(cls, act=ACT.RELU, norm=NORM.BATCH, in_channels: int = 3):
        return cls(**cls.PARA_MEDIUM, act=act, norm=norm, in_channels=in_channels)

    @classmethod
    def Large(cls, act=ACT.RELU, norm=NORM.BATCH, in_channels: int = 3):
        return cls(**cls.PARA_LARGE, act=act, norm=norm, in_channels=in_channels)

    @classmethod
    def XLarge(cls, act=ACT.RELU, norm=NORM.BATCH, in_channels: int = 3):
        return cls(**cls.PARA_XLARGE, act=act, norm=norm, in_channels=in_channels)


class DarkNetV5Bkbn(DarkNetV4Bkbn):
    CSPBLOCK = CSPBlockV5
    SAMP = Ck3NA
    PRE = partial(CpaNA, kernel_size=7, stride=2, )


class DarkNetResidualV8(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, act=ACT.LK, norm=NORM.BATCH, **kwargs):
        nn.Module.__init__(self)
        self.conv2 = Ck3NA(in_channels=in_channels, out_channels=out_channels, stride=stride, act=act, norm=norm)
        self.conv1 = Ck3s1NA(in_channels=out_channels, out_channels=out_channels, act=act, norm=norm)
        if stride > 1:
            self.shortcut = None
        elif in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Ck1s1(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class DarkNetResidualV8SE(DarkNetResidualV8):
    def __init__(self, in_channels: int, out_channels: int, stride: int, act=ACT.LK, norm=NORM.BATCH, **kwargs):
        DarkNetResidualV8.__init__(self, in_channels=in_channels, out_channels=out_channels,
                                   stride=stride, act=act, norm=norm, **kwargs)
        self.se = SEModule(channels=out_channels, ratio=0.25)

    def forward(self, x):
        out = self.conv2(self.se(self.conv1(x)))
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


class CSPBlockV8(nn.Module):
    def __init__(self, Module: nn.Module, in_channels: int, out_channels: int, num_repeat: int, stride: int = 1,
                 act=ACT.LK, norm=NORM.BATCH, **kwargs):
        nn.Module.__init__(self)
        assert stride == 1
        self.inter = Ck1s1NA(in_channels=in_channels, out_channels=out_channels, act=act, norm=norm)
        inner_channels = out_channels // 2
        self.inner_channels = inner_channels
        self.backbone = nn.ModuleList([])
        for i in range(num_repeat):
            self.backbone.append(Module(in_channels=inner_channels, out_channels=inner_channels, stride=1,
                                        act=act, norm=norm))
        self.outer = Ck1s1NA(in_channels=inner_channels * (num_repeat + 2), out_channels=out_channels, act=act,
                             norm=norm)

    def forward(self, x):
        x = self.inter(x)
        _, buffer = x.split(self.inner_channels, dim=1)
        xs = [x]
        for module in self.backbone:
            buffer = module(buffer)
            xs.append(buffer)
        xs = torch.cat(xs, dim=1)
        xs = self.outer(xs)
        return xs


class DarkNetV8Bkbn(DarkNetV4Bkbn):
    MODULE = DarkNetResidualV8
    CSPBLOCK = CSPBlockV8
    SAMP = Ck3NA
    PRE = partial(Ck3NA, stride=2, )

    PARA_NANO = dict(channelss=(32, 64, 128, 256), nums_repeat=(1, 2, 2, 1), strides=(2, 2, 2, 2))
    PARA_SMALL = dict(channelss=(64, 128, 256, 512), nums_repeat=(1, 2, 2, 1), strides=(2, 2, 2, 2))
    PARA_MEDIUM = dict(channelss=(96, 192, 384, 576), nums_repeat=(2, 4, 4, 2), strides=(2, 2, 2, 2))
    PARA_LARGE = dict(channelss=(128, 256, 512, 512), nums_repeat=(3, 6, 6, 3), strides=(2, 2, 2, 2))
    PARA_XLARGE = dict(channelss=(160, 320, 640, 640), nums_repeat=(4, 8, 8, 4), strides=(2, 2, 2, 2))


class RepNCSP(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, Module: nn.Module, in_channels: int, out_channels: int, num_repeat: int = 1, norm=NORM.BATCH,
                 act=ACT.SILU, ):
        nn.Module.__init__(self)
        inner_channels = out_channels // 2  # hidden channels
        self.conv1 = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, norm=norm, act=act)
        self.conv2 = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, norm=norm, act=act)
        self.conv3 = Ck1s1NA(in_channels=2 * inner_channels, out_channels=out_channels, norm=norm, act=act)
        self.stem = nn.Sequential(
            *(Module(in_channels=inner_channels, out_channels=inner_channels, stride=1, norm=norm, act=act)
              for _ in range(num_repeat)))

    def forward(self, x):
        return self.conv3(torch.cat((self.stem(self.conv1(x)), self.conv2(x)), dim=1))


class ADown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, norm=NORM.BATCH, act=ACT.SILU, **kwargs):
        nn.Module.__init__(self)
        self.stride = stride
        self.conv1 = CpaNA(in_channels=in_channels // 2, out_channels=out_channels // 2, stride=stride,
                           kernel_size=stride, norm=norm, act=act)
        self.conv2 = Ck1s1NA(in_channels=in_channels // 2, out_channels=out_channels // 2, norm=norm, act=act)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = F.max_pool2d(x2, kernel_size=self.stride, stride=self.stride, padding=0)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class CSPBlockV9(nn.Module):
    # csp-elan
    def __init__(self, Module: nn.Module, in_channels: int, out_channels: int, inner_channels: int, num_repeat: int = 1,
                 norm=NORM.BATCH, act=ACT.SILU, **kwargs):
        super().__init__()
        self.inter = Ck1s1NA(in_channels=in_channels, out_channels=inner_channels, norm=norm, act=act)
        stem_channels = inner_channels // 2
        self.stem1 = nn.Sequential(
            RepNCSP(Module=Module, in_channels=stem_channels, out_channels=stem_channels,
                    num_repeat=num_repeat, norm=norm, act=act),
            Ck3s1NA(in_channels=stem_channels, out_channels=stem_channels, norm=norm, act=act))
        self.stem2 = nn.Sequential(
            RepNCSP(Module=Module, in_channels=stem_channels, out_channels=stem_channels,
                    num_repeat=num_repeat, norm=norm, act=act),
            Ck3s1NA(in_channels=stem_channels, out_channels=stem_channels, norm=norm, act=act))
        self.outer = Ck1s1NA(in_channels=inner_channels + inner_channels, out_channels=out_channels, norm=norm, act=act)

    def forward(self, x):
        x = self.inter(x)
        x1, x2 = x.chunk(2, dim=1)
        x3 = self.stem1(x2)
        x4 = self.stem2(x3)
        xc = torch.cat([x, x3, x4], dim=1)
        return self.outer(xc)


class CBModule(nn.Module):
    def __init__(self, in_channelss: Sequence[int], out_channelss: Sequence[int], norm=NORM.BATCH, act=ACT.SILU, ):
        nn.Module.__init__(self)
        self.cvtors = nn.ModuleList()
        self.out_channelss = out_channelss
        for i, in_channels in enumerate(in_channelss):
            self.cvtors.append(
                Ck1s1NA(in_channels=in_channels, out_channels=sum(out_channelss[:i + 1]), act=act, norm=norm))

    def forward(self, feats):
        featss_tmp = [[] for _ in range(len(feats))]
        for i, (feat, cvtor) in enumerate(zip(feats, self.cvtors)):
            feat = cvtor(feat)
            feats_prcd = feat.split(self.out_channelss[:i + 1], dim=1)
            for j, feat_prcd in enumerate(feats_prcd):
                feat_prcd = F.interpolate(feat_prcd, size=feats[j].size()[-2:], mode='nearest')
                featss_tmp[j].append(feat_prcd)
        feats_out = [sum(feats_tmp) for feats_tmp in featss_tmp]
        return feats_out


class DarkNetV9Bkbn(nn.Module):
    MODULE = DarkNetResidualV8
    CSPBLOCK = CSPBlockV9
    SAMP = ADown
    PRE = partial(Ck3NA, stride=2, )

    def __init__(self, channelss: Sequence[int], channelss_samp: Sequence[int], nums_repeat: Sequence[int],
                 strides: Sequence[int], act=ACT.LK, norm=NORM.BATCH, in_channels: int = 3, **kwargs):
        nn.Module.__init__(self)
        self.pre = self.PRE(in_channels=in_channels, out_channels=channelss[0] // 2, act=act, norm=norm)
        self.stages = nn.ModuleList([])
        self.samps = nn.ModuleList([])
        for i in range(len(channelss)):
            if i == 0:
                in_channels_cur = channelss[0] // 2
            else:
                in_channels_cur = channelss[i - 1]
            stride = strides[i]
            if self.SAMP is None:
                smap = nn.Identity()
            else:
                smap = self.SAMP(in_channels=in_channels_cur, out_channels=channelss_samp[i], stride=stride,
                                 act=act, norm=norm, **kwargs_dispense(index=i, **kwargs))
                in_channels_cur = channelss_samp[i]
                stride = 1
            self.samps.append(smap)
            out_channels_cur = channelss[i]
            self.stages.append(self.CSPBLOCK(
                Module=self.MODULE, in_channels=in_channels_cur, out_channels=out_channels_cur,
                inner_channels=out_channels_cur // 2, num_repeat=nums_repeat[i],
                stride=stride, act=act, norm=norm, **kwargs_dispense(index=i, **kwargs)))

    def forward(self, imgs):
        feats = self.pre(imgs)
        for i in range(len(self.stages)):
            feats = self.samps[i](feats)
            feats = self.stages[i](feats)
        return feats

    PARA_LARGE = dict(channelss=(256, 512, 1024, 1024), channelss_samp=(128, 256, 512, 1024),
                      nums_repeat=(2, 2, 2, 2), strides=(2, 2, 2, 2))

    @classmethod
    def Large(cls, act=ACT.RELU, norm=NORM.BATCH, in_channels: int = 3):
        return cls(**cls.PARA_LARGE, act=act, norm=norm, in_channels=in_channels)


class DarkNetV9AUXBkbn(DarkNetV9Bkbn):
    def __init__(self, channelss: Sequence[int], channelss_samp: Sequence[int],
                 nums_repeat: Sequence[int], strides: Sequence[int],
                 act=ACT.LK, norm=NORM.BATCH, in_channels: int = 3, **kwargs):
        DarkNetV9Bkbn.__init__(self, channelss=channelss, channelss_samp=channelss_samp, nums_repeat=nums_repeat,
                               strides=strides,
                               act=act, norm=norm, in_channels=in_channels, **kwargs)
        in_channelss = [channelss[0] // 2] + list(channelss)
        out_channelss = [channelss[0] // 2] + list(channelss_samp)
        self.cb = CBModule(in_channelss=in_channelss, out_channelss=out_channelss)
        self.pre_aux = self.PRE(in_channels=in_channels, out_channels=channelss[0] // 2, act=act, norm=norm)
        self.stages_aux = nn.ModuleList([])
        self.samps_aux = nn.ModuleList([])
        for i in range(len(channelss)):
            if i == 0:
                in_channels_cur = channelss[0] // 2
            else:
                in_channels_cur = channelss[i - 1]
            stride = strides[i]
            if self.SAMP is None:
                smap = nn.Identity()
            else:
                smap = self.SAMP(in_channels=in_channels_cur, out_channels=channelss_samp[i], stride=stride,
                                 act=act, norm=norm, **kwargs_dispense(index=i, **kwargs))
                in_channels_cur = channelss_samp[i]
                stride = 1
            self.samps_aux.append(smap)
            out_channels_cur = channelss[i]
            self.stages_aux.append(self.CSPBLOCK(
                Module=self.MODULE, in_channels=in_channels_cur, out_channels=out_channels_cur,
                inner_channels=out_channels_cur // 2, num_repeat=nums_repeat[i],
                stride=stride, act=act, norm=norm, **kwargs_dispense(index=i, **kwargs)))

    def forward(self, imgs):
        feats_aux = self.pre_aux(imgs)
        festss_aux = [feats_aux]
        for i in range(len(self.stages)):
            feats_aux = self.samps_aux[i](feats_aux)
            feats_aux = self.stages_aux[i](feats_aux)
            festss_aux.append(feats_aux)
        festss_aux = self.cb(festss_aux)
        feats = self.pre(imgs) + festss_aux[0]
        for i in range(len(self.stages)):
            feats = self.samps[i](feats) + festss_aux[i + 1]
            feats = self.stages[i](feats)
        return feats


if __name__ == '__main__':
    model = DarkNetV9AUXBkbn.Large()
    # model = DarkNetV4Bkbn.R53()
    imgs = torch.rand(2, 3, 512, 512)
    y = model(imgs)
    print(y.size())
