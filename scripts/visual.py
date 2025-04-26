import os
import sys

from datas.resh3d import Resh3DDataSource
from models.upsv3d.manager import ColSurfManager

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from datas import *
from utils import *

if __name__ == '__main__':
    img_size = (640, 640)
    batch_size = 4
    num_workers = 0

    root = os.path.join(PROJECT_DIR, 'dataset-example')
    set_name = 'all'
    img_folder = 'images'
    label_folder = 'labels_pred'

    ds = Resh3DDataSource(root=root, data_mode=DATA_MODE.FULL, )
    manager = ColSurfManager.from_dir(
        info_pth=ds.info_pth, names=ds.CLASS_NAMES_ZN, num_samp=1000, z_min=300, z_max=1800, focal=(5000, 5000),
        img_size=img_size, resol=2.0, color_noise=0.05)

    loader = ds.loader(set_name=set_name, label_folder=label_folder, img_folder=img_folder,
                       batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=num_workers,
                       # aug_seq=LargestMaxSize(max_size=img_size),
                       # aug_seq=AugNorm(img_size=img_size, p=1, thres=5),
                       # aug_seq=AugV3R(img_size=img_size, p=1, thres=1),
                       # aug_seq=Mosaic(num_repeat=1.0, img_size=img_size)
                       )

    imgs, labels = next(iter(loader))

    show_labels(imgs)
    show_labels(imgs, labels)
    # show_labels(labels)
    plt.pause(1e5)
