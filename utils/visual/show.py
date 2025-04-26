from .axplt import *
from .axplt import _axplt_items
from .axplt3d import _axplt3d_items
from .pilrnd import *


def show_label(*items, axis=None, tick=True, **kwargs):
    # if use_pil:
    #     img = _pilrnd_label(img, label, **kwargs)
    #     axis.imshow(img)
    #     return axis

    return _axplt_items(*items, axis=axis, **kwargs)


def show_label3d(*items, axis=None, tick=True, **kwargs):
    # if use_pil:
    #     img = _pilrnd_label(img, label, **kwargs)
    #     axis.imshow(img)
    #     return axis

    return _axplt3d_items(*items, axis=axis, **kwargs)


# 检查数据分布
def show_distribute(data, low=None, high=None, num_quant=100, axis=None, color='k', bar=True, uni=False):
    if axis is None:
        fig, axis = plt.subplots()
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data.reshape(-1)
    if len(data) == 0:
        return axis
    low = np.min(data) if low is None else low
    high = np.max(data) if high is None else high

    vals = np.linspace(low, high, num_quant + 1)
    inds = np.round((data - low) / (high - low) * num_quant).astype(np.int32)

    fltr_in = (inds >= 0) * (inds <= num_quant)
    inds = inds[fltr_in]
    nums = np.zeros(shape=num_quant + 1)
    np.add.at(nums, inds, 1)
    if uni:
        nums = nums / np.sum(nums) * num_quant
    if bar:
        axis.bar(vals, nums, width=(high - low) / num_quant * 0.8, color=color)
    else:
        axis.plot(vals, nums, color=color)
    return axis


# 显示图片
def show_img(img, axis=None, title=None, tick=True, **kwargs):
    img = img2imgN(img)

    axis = get_axis(axis=axis, tick=tick, **kwargs)

    axis.imshow(img, extent=(0, img.shape[1], img.shape[0], 0), **kwargs)
    if title is not None:
        axis.set_title(title, fontdict=FONT_DICT_LARGE)
    # axis.axis('off')
    return axis


# 显示曲线
def show_curve(data, axis=None, tick=True, color='k', **kwargs):
    axis = get_axis(axis=axis, tick=tick, **kwargs)
    axis.plot(data, color=color, **kwargs)
    return axis


# </editor-fold>

# <editor-fold desc='数据封装'>

class ShowDatas(metaclass=ABCMeta):

    @abstractmethod
    def size(self) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def get_kwargs(self, *index) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_args(self, *index) -> tuple:
        raise NotImplementedError


class ShowDatas1dArrs(ShowDatas):

    def size(self) -> tuple:
        return self._size

    def __init__(self, *items, **kwargs):
        self.items = items
        size = 1
        for itml in self.items:
            if hasattr(itml, '__len__'):
                size = max(size, len(itml))
        self._size = (size,)
        self.kwargs = kwargs

    def get_args(self, index) -> tuple:
        ret = []
        for itml in self.items:
            try:
                ret.append(itml[index])
            except Exception as e:
                ret.append(itml)
        return tuple(ret)

    def get_kwargs(self, index) -> dict:
        sub_dct = {}
        for key, valuel in self.kwargs.items():
            try:
                sub_dct[key] = valuel[index]
            except Exception as e:
                sub_dct[key] = valuel
        return sub_dct


class ShowDatas2dArrs(ShowDatas):
    def size(self) -> tuple:
        return self._size

    def __init__(self, *items, size: tuple = (2, 2), **kwargs):
        self.items = items
        self._size = size
        self.kwargs = kwargs

    def get_args(self, index1, index2) -> tuple:
        ret = []
        for itml in self.items:
            if hasattr(itml, '__get_item__'):
                ret.append(itml[index1][index2])
            else:
                ret.append(itml)
        return tuple(ret)

    def get_kwargs(self, index1, index2) -> dict:
        sub_dct = {}
        for key, valuel in self.kwargs.items():
            if hasattr(valuel, '__get_item__'):
                sub_dct[key] = valuel[index1][index2]
            else:
                sub_dct[key] = valuel
        return sub_dct


# </editor-fold>

# <editor-fold desc='子图排列'>

# 按数列展示
def arrange_arr(datas, shower: Callable, projection=None):
    # 确定长宽
    s0 = datas.size()[0]
    area = 8 * 8
    num_wid = int(np.ceil(np.sqrt(s0)))
    num_hei = int(np.ceil(s0 / num_wid))
    rate = num_wid / num_hei
    hei = round(np.sqrt(area / rate))
    wid = round(area / hei)
    fig = plt.figure(figsize=(wid, hei))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, )
    # 画图
    ind = 0
    for i in range(num_wid):
        for j in range(num_hei):
            if ind == s0:
                break
            axis = fig.add_subplot(num_hei, num_wid, ind + 1, projection=projection)
            shower(*datas.get_args(ind), axis=axis, index=ind, **datas.get_kwargs(ind))
            ind += 1
    fig.subplots_adjust(wspace=0.15, hspace=0.2)
    return fig


# 按矩阵展示
def arrange_mat(datas, shower: Callable, ):
    s0, s1 = datas.size()
    rate = s0 / s1
    area = 8 * 8
    wid = round(np.sqrt(area / rate))
    hei = round(area / wid)
    fig = plt.figure(figsize=(wid, hei))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, )
    # 画图
    ind = 1
    for i in range(s0):
        for j in range(s1):
            axis = fig.add_subplot(s0, s1, ind)
            shower(*datas.get_args(i, j), axis=axis, **datas.get_kwargs(i, j))
            ind += 1
    # fig.subplots_adjust(wspace=0.3, hspace=0.6)
    return fig


# </editor-fold>


# <editor-fold desc='数据展示接口'>

def show_arrs(arrs, as_1d=True, gol_uni=False, titles=None, **kwargs):
    if isinstance(arrs, torch.Tensor):
        arrs = arrs.detach().cpu().numpy()
    if gol_uni:
        kwargs['vmin'] = np.min(arrs)
        kwargs['vmax'] = np.max(arrs)
    if len(arrs.shape) == 2:
        datas = ShowDatas1dArrs(size=arrs.shape[0], data=arrs, title=titles, **kwargs)
        fig = arrange_arr(datas, shower=show_curve)
    elif as_1d:
        datas = ShowDatas1dArrs(size=arrs.shape[0], img=arrs, title=titles, **kwargs)
        fig = arrange_arr(datas, shower=show_img)
    elif len(arrs.shape) == 4:
        datas = ShowDatas2dArrs(size=arrs.shape[0:2], img=arrs, title=titles, **kwargs)
        fig = arrange_mat(datas, shower=show_img)
    else:
        raise Exception('err shape')
    return fig


def show_labels(*items, **kwargs):
    datas = ShowDatas1dArrs(*items, **kwargs)
    fig = arrange_arr(datas, shower=show_label, projection=None)
    return fig


def show_labels3d(*items, **kwargs):
    datas = ShowDatas1dArrs(*items, **kwargs)
    fig = arrange_arr(datas, shower=show_label3d, projection='3d')
    return fig

# </editor-fold>
