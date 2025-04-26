import os
import sys

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from models import *
from tools import *
from datas import *
from datas.resh3d import Resh3DDataSource
from datas.base.inplabel import InpPKLWriter
from models.upsv3d.manager import ColSurfManager
from models.upsv3d.updetD_yv import UpDetDYOLO
from models.upsv3d.postproc import UpdetLabelWriteProcActor, EpochBasedRendActor
from models.upsv3d.uprender import UpRender
from models.upsv3d.updetD_mm import UpDetDMM

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    local_rank, world_size = init_if_dist(0)

    img_size = (640, 640)
    num_epoch = 30
    num_workers = 4
    batch_size = 1
    enable_half = False
    root = os.path.join(PROJECT_DIR, 'dataset-example')
    set_name_train = 'train'
    set_name_test = 'test'
    img_folder = 'images'
    img_folder_gen = 'img_folder_gen'
    label_folder_gen = 'labels_gen'

    ds = Resh3DDataSource(root=root, data_mode=DATA_MODE.FULL, )
    train_loader = ds.loader(set_name=set_name_train, img_folder=img_folder_gen, label_folder=label_folder_gen,
                             batch_size=batch_size, pin_memory=False, shuffle=True,
                             drop_last=True, num_workers=num_workers,
                             aug_seq=AugV3R(img_size=img_size, thres=1))
    test_loader = ds.loader(set_name=set_name_test, img_folder=img_folder, label_folder='labels_tst',
                            cls_names=ds.CLASS_NAMES_TST,
                            batch_size=batch_size, pin_memory=False, shuffle=False,
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
    # render = UpRender(device=local_rank, manager=manager, num_buf_max=1024, nums_priori=ds.NUMS_PRIORI)
    render = UpRender(device=local_rank, manager=manager, num_buf_max=1024)  # 生成类别无先验

    # model = UpDetDMM.SwinTiny(device=local_rank, manager=manager, )
    # model = UpDetDMM.R101(device=local_rank, manager=manager, )
    model = UpDetDMM.R50(device=local_rank, manager=manager, )
    # model = UpDetDYOLO.V11L(device=local_rank, manager=manager, )
    # model.load(os.path.join(PROJECT_DIR, 'ckpt/pretrained/resnet50_in1k'), transfer=True)
    save_pth = os.path.join(PROJECT_DIR, 'ckpt/rrr_res50fpn')

    broadcast = LogBasedBroadcastActor(save_pth=save_pth, new_log=True)

    kwargs_rend = dict(num_obj=4, cind2name=anno_loader.cind2name)
    lb_writer = InpPKLWriter(label_dir=os.path.join(ds.root, label_folder_gen), set_pth=None)
    annor_rnd = Annotator(anno_loader, total_epoch=1, kwargs_annotate=kwargs_rend, enable_half=enable_half)
    annor_rnd.add_actor(AnnotatorCollectActor(step=50, title='Rend'))
    annor_rnd.add_actor(broadcast)
    annor_rnd.add_actor(LabelWriteActor(writer=lb_writer))
    annor_rnd.add_actor(ImageSaveActor(save_dir=os.path.join(ds.root, img_folder_gen)))

    kwargs_infer = dict(conf_thres=0.01, iou_thres=0.4, ntop_cls=1, num_presv=200, cluster_index=CLUSTER_INDEX.NONE,
                        cind2name=anno_loader.cind2name, num_limt=10, last_ratio=0.7)
    annor_inf = Annotator(anno_loader, total_epoch=1, kwargs_annotate=kwargs_infer, enable_half=enable_half)
    annor_inf.add_actor(AnnotatorCollectActor(step=50, title='Infer'))
    annor_inf.add_actor(broadcast)
    annor_inf.add_actor(UpdetLabelWriteProcActor(
        writer=lb_writer, nums_priori=ds.NUMS_PRIORI_TST * 0.6,
        cluster_index=ds.CLUSTER_INDEX, cls_names=ds.CLASS_NAMES_TST))

    kwargs_eval = dict(conf_thres=0.01, iou_thres=0.4, ntop_cls=3, num_presv=2000, cluster_index=ds.CLUSTER_INDEX,
                       label_mode='xyxy', cind2name=ds.cind2name_tst)
    evaler = BoxVOCEvaler(loader=test_loader, total_epoch=1, kwargs_eval=kwargs_eval, enable_half=enable_half,
                          ignore_class=False)
    evaler.add_actor(EvalerCollectActor(step=50))
    evaler.add_actor(broadcast)

    trainer = Trainer(loader=train_loader, enable_half=enable_half, total_epoch=None, total_iter=None, accu_step=1,
                      kwargs_train=dict(cluster_index=ds.CLUSTER_INDEX))
    trainer.add_actor(TrainerCollectActor(step=50))
    trainer.add_actor(broadcast)
    trainer.add_actor(SGDBuilder(lr=0.01, momentum=0.95, dampening=0, weight_decay=5e-4))
    trainer.add_actor(GradClipActor(grad_norm=10))
    trainer.add_actor(IterBasedEMAActor(ema_ratio=0.99, step=1, lev=-1))
    trainer.add_actor(EpochBasedLRScheduler.Cos(lr_init=0.01, lr_end=1e-6, num_epoch=num_epoch))
    trainer.add_actor(EpochBasedSaveActor(save_pth=save_pth, step=1, last=True, offset=0, num_keep=1, lev=0))
    # trainer.add_actor(
    #     EpochBasedSaveActor(save_pth=save_pth, step=1, num_keep=1, lev=0, formatter=FORMATTER.SAVE_PTH_SIMPLE))
    trainer.add_actor(EpochBasedEvalActor(evaler, step=1, save_pth=save_pth, lev=1))
    trainer.add_actor(EpochBasedEvalActor(evaler, step=1, save_pth=save_pth, lev=1, select_ema=True))
    trainer.add_actor(EpochBasedRendActor(annor_rnd, render=render, step=3, lev=2, first=False))
    trainer.add_actor(EpochBasedAnnotateActor(annor_inf, step=3, lev=3, select_ema=True, first=False))

    # if local_rank == 0:
    #     train_loader.dataset.delete()
    annor_rnd.start(render)
    # annor_inf.start(model)

    # evaler.start(model)
    trainer.start(model)

    sys.exit(0)
