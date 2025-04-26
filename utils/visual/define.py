from typing import Optional

from ..iotools import *
from ..label import Category
from ..label import IndexCategory

os.environ["DISPLAY"] = "localhost:11.0"

REGISTER_MATPLOTLIB_BACKEND = {
    PLATFORM_LAPTOP: 'Qt5Agg',
    PLATFORM_SEV4090: 'tkagg',
    PLATFORM_SEV3090: 'tkagg'
}

import matplotlib

matplotlib.use('tkagg')
# matplotlib.use(REGISTER_MATPLOTLIB_BACKEND.get(PLATFORM, 'agg'))
import matplotlib.pyplot as plt

plt.interactive(True)
from ..define import *

try:
    from ..fonts import FONT_MAPPER

    PILRND_FONT_PTH = FONT_MAPPER['times']
except Exception as e:
    PILRND_FONT_PTH = ''

matplotlib.rcParams['font.sans-serif'] = ['times']
matplotlib.rcParams['axes.unicode_minus'] = False

FONT_DICT_SMALL = dict(fontfamily='Times New Roman', fontsize='small', weight='bold')
FONT_DICT_XLARGE = dict(fontfamily='Times New Roman', fontsize='x-large', weight='bold')
FONT_DICT_LARGE = dict(fontfamily='Times New Roman', fontsize='large', weight='bold')

IMAGE_SIZE_DEFAULT = (256, 256)
CMAP = plt.get_cmap('jet')

REGISTER_AXPLT = Register()
REGISTER_AXPLT3D = Register()

PLT_COLOR = Union[str, tuple]
PLT_AXIS = object
PLT_AXIS3D = object
PIL_COLOR = tuple


def cate_cont2str(cate_cont):
    cate = IndexCategory.convert(cate_cont.category)
    name = cate_cont['name'] if 'name' in cate_cont.keys() else '<%d>' % cate._cindN
    cate_str = name + ' %.2f' % cate.confN if cate.confN < 1 else name
    return cate_str


def _item_text(category: Category, name: Optional[str] = None, ) -> str:
    if name is None:
        name = '<%d>' % category.cindN
    if category.confN < 1:
        name += ' %.2f' % category.confN
    return name


from matplotlib import colors as plt_colors


# <editor-fold desc='颜色处理'>
def random_color(index: int, low: int = 30, high: int = 200, unit: bool = False) -> tuple:
    radius = (high - low) / 2
    color = np.cos([index * 7, index * 8 + np.pi, index * 9 - np.pi]) * radius + radius + low
    color = tuple(color / 255) if unit else tuple(color.astype(np.int32))
    return color


def _ensure_rgb(color: PLT_COLOR, unit: bool = True):
    if isinstance(color, str):
        color = plt_colors.to_rgb(color)
        if not unit:
            color = _unit_col2int_col(color)
        return color
    elif isinstance(color, tuple):
        return color
    else:
        raise Exception('color err')


def _determine_color(color: Optional[PLT_COLOR] = None, cind2col: Optional[Callable] = None,
                     unit: bool = False, index: int = 0) -> tuple:
    if color is not None:
        return _ensure_rgb(color, unit=unit)
    elif cind2col is not None:
        return cind2col(index)
    else:
        return random_color(index, unit=unit)


def _unit_col2int_col(color: tuple) -> tuple:
    color = matplotlib.colors.to_rgb(color)
    return tuple((np.array(color) * 255).astype(np.int32))


def _int_col2unit_col(color: tuple) -> tuple:
    return tuple(np.array(color) / 255)

# </editor-fold>
