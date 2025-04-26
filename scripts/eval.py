import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    local_rank, world_size = init_if_dist(3, init_random=False)
    img_size = (640, 640)
    num_workers = 4
    batch_size = 4
    enable_half = False

    root = os.path.join(PROJECT_DIR, 'dataset-example')
    set_name_train = 'train'
    set_name_test = 'test'
    img_folder = 'images'
    img_folder_gen = 'img_folder_gen'
    label_folder_gen = 'labels_gen'

    ds = Resh3DDataSource(root=root, data_mode=DATA_MODE.FULL, )
    # ds = Resh3DDataSource(root='D:\Datasets\Resh3D', data_mode=DATA_MODE.FULL, )

    train_loader = ds.loader(set_name=set_name_train, img_folder=img_folder_gen, label_folder=label_folder_gen,
                             batch_size=batch_size, pin_memory=False, shuffle=True,
                             drop_last=True, num_workers=num_workers,
                             # aug_seq=AugV5R(img_size=img_size, thres=10),
                             aug_seq=AugNorm(img_size=img_size, thres=10)
                             )
    test_loader = ds.loader(set_name=set_name_test, img_folder=img_folder, label_folder='labels_tst',
                            cls_names=ds.CLASS_NAMES_TST,
                            batch_size=batch_size, pin_memory=False, shuffle=True,
                            drop_last=False, num_workers=num_workers,
                            aug_seq=AugNorm(img_size=img_size, thres=20))
    anno_loader = ds.loader(set_name=set_name_train, img_folder=img_folder, label_folder=label_folder_gen,
                            batch_size=batch_size, pin_memory=False, shuffle=True,
                            drop_last=False, num_workers=num_workers,
                            # aug_seq=AugV1(img_size=img_size)
                            )

    manager = ColSurfManager.from_dir(
        info_pth=ds.info_pth, names=ds.CLASS_NAMES_ZN, img_size=img_size,
        num_samp=1000, z_min=300, z_max=1800, z_pow=0.5, focal=(5000, 5000), resol=2.0, color_noise=0.1)

    model = UpDetDMM.R50(device=local_rank, manager=manager, )
    # model = UpDetDMM.SwinTiny(device=local_rank, manager=manager, )
    # model = UpDetDYOLO.V11L(device=local_rank, manager=manager, )

    save_pth = os.path.join(PROJECT_DIR, 'ckpt/any_weight.pth')
    model.load(save_pth)

    kwargs_eval = dict(conf_thres=0.001, iou_thres=0.4, ntop_cls=None, num_presv=2000, cluster_index=ds.CLUSTER_INDEX,
                       label_mode='xyxy', cind2name=ds.cind2name_tst)
    kwargs_infer = dict(conf_thres=0.01, iou_thres=0.4, ntop_cls=1, num_presv=200, cluster_index=CLUSTER_INDEX.NONE,
                        cind2name=anno_loader.cind2name, num_limt=10, last_ratio=0.7)
    render = UpRender(device=local_rank, manager=manager, num_buf_max=1024)  # 生成类别无先验
    broadcast = PrintBasedBroadcastActor()

    evaler = BoxCOCOSTDEvaler(loader=test_loader, total_epoch=1, kwargs_eval=kwargs_eval, enable_half=enable_half,
                              ignore_class=False)
    evaler.add_actor(EvalerCollectActor(step=50))
    evaler.add_actor(EvalerCacheActor(save_pth=save_pth))
    evaler.add_actor(EvalerReportActor(save_pth=save_pth))
    evaler.add_actor(broadcast)

    annor_inf = Annotator(anno_loader, total_epoch=1, kwargs_annotate=kwargs_infer, enable_half=enable_half)
    annor_inf.add_actor(AnnotatorCollectActor(step=50, title='Infer'))
    annor_inf.add_actor(broadcast)
    lb_writer = InpPKLWriter(label_dir=os.path.join(ds.root, label_folder_gen), set_pth=None)
    annor_inf.add_actor(UpdetLabelWriteProcActor(
        writer=lb_writer, nums_priori=ds.NUMS_PRIORI_TST,
        cluster_index=ds.CLUSTER_INDEX, cls_names=ds.CLASS_NAMES_TST))

    # imgs, labels = next(iter(train_loader))
    # show_labels(imgs, labels)
    # labels_pd = model.imgs2labels(imgs, **kwargs_infer)
    # # # show_labels(imgs)
    # show_labels(imgs, labels)
    # show_labels(imgs, labels_pd)
    evaler.start(model)
    # annor_inf.start(model)

    # plt.pause(1e5)
