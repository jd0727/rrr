import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_DIR)
from models import *
from tools import *
from datas import *
from datas.resh3d import Resh3DDataSource
from utils import *
from pytorch3d.renderer import PointLights, DirectionalLights

from datas.base.inplabel import InpPKLWriter

if __name__ == '__main__':
    root = 'VOC dataset root'
    ds = VOCCommon(root=root, img_folder='JPEGImages', set_folder='',
                   anno_folder='Annotations_tst', cls_names=Resh3DDataSource.CLASS_NAMES_TST)
    lb_writer = InpPKLWriter(label_dir=os.path.join(ds.root, 'labels_tst'), set_pth=None)

    for set_name in ['test', 'val']:
        lb_writer.save_labels_of(ds.dataset(set_name))
