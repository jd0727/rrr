import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from datas.base.inplabel import InpPKLWriter
from datas.resh3d import Resh3DDataSource
from models.upsv3d.manager import ColSurfManager
from models.upsv3d.postproc import UpdetLabelWriteProcActor
from models.upsv3d.updetD import UpDetD
from models.upsv3d.updetD_mm import UpDetDMM
from models.upsv3d.updetD_yv import UpDetDYOLO
from models.upsv3d.uprender import UpRender
from models import *
from tools import *
from datas import *

if __name__ == '__main__':
    local_rank, world_size = init_if_dist(0, init_random=False)
    img_size = (640, 640)
    num_workers = 0
    batch_size = 4
    enable_half = False

    root = os.path.join(PROJECT_DIR, 'dataset-example')
    set_name = 'all'
    img_folder = 'images'
    label_folder = 'labels_pred'

    ds = Resh3DDataSource(root=root, data_mode=DATA_MODE.FULL, )
    # ds = Resh3DDataSource(root='D:\Datasets\Resh3D', data_mode=DATA_MODE.FULL, )

    anno_loader = ds.loader(set_name=set_name, img_folder=img_folder, label_folder=label_folder,
                            batch_size=batch_size, pin_memory=False, shuffle=True,
                            drop_last=False, num_workers=num_workers,
                            # aug_seq=AugV1(img_size=img_size)
                            )

    # info_pth = os.path.join('/ses-data/JD/Resh3DX/', 'wobjs/info.json')
    manager = ColSurfManager.from_dir(
        info_pth=ds.info_pth, names=ds.CLASS_NAMES_ZN, img_size=img_size,
        num_samp=1000, z_min=300, z_max=1800, z_pow=0.5, focal=(5000, 5000), resol=2.0, color_noise=0.1)

    model = UpDetDMM.R50(device=local_rank, manager=manager, )
    # model = UpDetDMM.SwinTiny(device=local_rank, manager=manager, )
    # model = UpDetDYOLO.V11L(device=local_rank, manager=manager, )

    save_pth = os.path.join(PROJECT_DIR, 'weights/rrr_res50fpn_pretrain.pth')
    model.load(save_pth)

    kwargs_infer = dict(conf_thres=0.001, iou_thres=0.4, ntop_cls=1, num_presv=200, cluster_index=CLUSTER_INDEX.NONE,
                        cind2name=anno_loader.cind2name, num_limt=10, last_ratio=0.7)
    render = UpRender(device=local_rank, manager=manager, num_buf_max=1024)  # 生成类别无先验
    broadcast = PrintBasedBroadcastActor()

    annor_inf = Annotator(anno_loader, total_epoch=1, kwargs_annotate=kwargs_infer, enable_half=enable_half)
    annor_inf.add_actor(AnnotatorCollectActor(step=50, title='Infer'))
    annor_inf.add_actor(broadcast)
    lb_writer = InpPKLWriter(label_dir=os.path.join(ds.root, label_folder), set_pth=None)
    annor_inf.add_actor(UpdetLabelWriteProcActor(
        writer=lb_writer, nums_priori=ds.NUMS_PRIORI_TST * 10,
        cluster_index=ds.CLUSTER_INDEX, cls_names=ds.CLASS_NAMES_TST))

    annor_inf.start(model)

    # plt.pause(1e5)
