from datas import LabelWriter
from .base.container import _set_model_state
from .evaler import EvalerCollectActor
from utils.visual.pilrnd import _pilrnd_items
from .base import *


# <editor-fold desc='标注器原型'>
class Annotator(IterBasedTemplate, VariableManager):
    def __init__(self,
                 loader: IMDataLoader,
                 total_epoch: int = 1,
                 device: torch.device = DEVICE,
                 enable_half: bool = True,
                 kwargs_annotate: Optional[Dict] = None,
                 ):
        IterBasedTemplate.__init__(self, total_epoch=total_epoch, total_iter=total_epoch * len(loader),
                                   loader=loader, device=device, processor=None)
        VariableManager.__init__(self)
        self.cind2name = loader.cind2name
        self.kwargs_annotate = {} if kwargs_annotate is None else kwargs_annotate
        self.labels_cmb = None
        self.enable_half = enable_half
        self.return_obj = None

    def set_state(self, model):
        return _set_model_state(model, train_mode=False, enable_half=self.enable_half)

    def act_init(self, model, *args, kwargs_annotate: Optional[Dict] = None, **kwargs):
        if kwargs_annotate is not None:
            self.kwargs_annotate.update(kwargs_annotate)
        assert isinstance(model, AnnotatableModel)
        self.model = model
        if isinstance(model, HasDevice):
            self.device = model.device
        else:
            self.device = DEVICE
        self.set_state(model)
        self.running = True
        self.labels_cmb = []
        self.ind_iter = 0
        self.ind_epoch = 0
        self.add_actor(model)
        self.model.act_init_annotate(self)
        dist_barrier()
        return None

    def act_return(self):
        return self.return_obj

    def act_ending(self):
        dist_barrier()
        return None

    def act_iter(self):
        imgs_ds, labels_ds = self.batch_data

        with torch.no_grad():
            self.update_time(TIMENODE.BEFORE_INFER)
            imgs_md, labels_md = self.model.act_iter_annotate(self, imgs_ds, labels_ds, **self.kwargs_annotate)
            self.update_time(TIMENODE.AFTER_INFER)
        self.infered_data = (imgs_md, labels_md)

        return None


class EpochBasedAnnotateActor(IntervalTrigger, AfterEpochActor):

    def __init__(self, annotator: Annotator, step: int, offset: int = 0, first: bool = False,
                 last: bool = False, select_ema: bool = False, lev: int = 1):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.annotator = annotator
        self.select_ema = select_ema
        self.lev = lev

    def act_after_epoch(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_epoch, total=trainer.total_epoch):
            if self.select_ema and trainer.model_ema is not None:
                model = trainer.model_ema
            else:
                model = trainer.model
            self.annotator.start(model)

    @property
    def lev_after_epoch(self):
        return self.lev


class IterBasedAnnotateActor(IntervalTrigger, AfterIterActor):

    def __init__(self, annotator: Annotator, step: int, offset: int = 0, first: bool = False,
                 last: bool = False, select_ema: bool = False, lev: int = 1):
        IntervalTrigger.__init__(self, step=step, offset=offset, first=first, last=last)
        self.annotator = annotator
        self.select_ema = select_ema
        self.lev = lev

    def act_after_iter(self, trainer, **kwargs):
        if self.trigger(ind=trainer.ind_iter, total=trainer.total_iter):
            if self.select_ema:
                model = trainer.model_ema
            else:
                model = trainer.model
            self.annotator.start(model)

    @property
    def lev_after_iter(self):
        return self.lev


# </editor-fold>

# <editor-fold desc='标注器可选'>

class AnnotatorCacheActor(EndingActor):
    def __init__(self, save_pth: str, formatter=FORMATTER.SAVE_PTH_SIMPLE):
        self.save_pth = format_save_pth(
            save_pth=save_pth, formatter=formatter, extend=EXTENDS.CACHE,
            default_name='eval', appendix='_cache')

    def act_ending(self, annoer: Annotator, **kwargs):
        if annoer.main_proc and len(self.save_pth) > 0:
            annoer.broadcast('Save test cache at ' + self.save_pth)
            save_pkl(file_pth=self.save_pth, obj=annoer.labels_cmb)


class AnnotatorLabelCollectActor(EndingActor, AfterIterActor):
    def __init__(self, with_recover: bool = False):
        self.with_recover = with_recover

    def act_ending(self, container, **kwargs):
        container.labels_cmb = all_extend_object_list(container.labels_cmb)
        container.broadcast('Cluster %d labels' % len(container.labels_cmb))
        labels_dct = {}
        for label_cmb in container.labels_cmb:
            label_ds, label_md = label_cmb
            if label_ds.meta not in labels_dct.keys():
                labels_dct[label_ds.meta] = [label_cmb]
            else:
                labels_dct[label_ds.meta].append(label_cmb)
        container.return_obj = labels_dct
        return None

    def act_after_iter(self, container, **kwargs):
        imgs_ds, labels_ds = container.batch_data
        imgs_md, labels_md = container.infered_data
        for label_ds, label_md in zip(labels_ds, labels_md):
            if label_md is not None and label_ds is not None:
                label_md.info_from(label_ds)
            if label_md is not None and self.with_recover:
                label_md.recover()
            if label_ds is not None and self.with_recover:
                label_ds.recover()
            container.labels_cmb.append((label_md, label_ds))
        return None


class AnnotatorCollectActor(EvalerCollectActor):

    def __init__(self, step: int = 50, offset: int = 0, first: bool = False, last: bool = False,
                 title: str = 'Annotate'):
        EvalerCollectActor.__init__(self, step=step, offset=offset, first=first, last=last, title=title)

    def act_ending(self, evaler, **kwargs):
        pass


# </editor-fold>

# <editor-fold desc='标注器扩展'>
class LabelWriteActor(AfterIterActor, InitialActor, EndingActor):

    def __init__(self, writer: LabelWriter, with_recover: bool = False):
        self.writer = writer
        self.with_recover = with_recover
        self.caches = []

    def act_init(self, container, **kwargs):
        self.caches = []

    def act_after_iter(self, container, **kwargs):
        imgs_ds, labels_ds = container.batch_data
        imgs_md, labels_md = container.infered_data
        for label_ds, label_md in zip(labels_ds, labels_md):
            if label_md is not None and label_ds is not None:
                label_md.info_from(label_ds)
            if label_md is not None and self.with_recover:
                label_md.recover()
            if label_ds is not None and self.with_recover:
                label_ds.recover()
            self.caches.append(self.writer.save_label(label_md))

    def act_ending(self, container, **kwargs):
        self.caches = all_extend_object_list(self.caches)
        if container.main_proc:
            self.writer.save_all(self.caches)


class ImageRendSaveActor(AfterIterActor, InitialActor):
    def act_init(self, container, **kwargs):
        container.update_period(name='imgrndsave', period_pair=PERIOD.IMG_RNDSAVE)

    def __init__(self, save_dir: dir, from_model: bool = True, formatter=FORMATTER.SAVE_PTH_SIMPLE, **kwargs):
        self.save_dir = save_dir
        self.from_model = from_model
        self.formatter = formatter
        ensure_folder_pth(save_dir)
        self.kwargs = kwargs

    def act_after_iter(self, container, **kwargs):
        if self.from_model:
            imgs, labels = container.infered_data
        else:
            imgs, labels = container.batch_data
        container.update_time(TIMENODE.BEFORE_IMG_RNDSAVE)
        for img, label in zip(imgs, labels):
            imgP_md = _pilrnd_items(img, label)
            save_pth = format_save_pth(save_pth=os.path.join(self.save_dir, label.meta), extend='jpg',
                                       formatter=self.formatter)
            imgP_md.save(save_pth, quality=100)
        container.update_time(TIMENODE.AFTER_IMG_RNDSAVE)


class ImageSaveActor(AfterIterActor, InitialActor):
    def act_init(self, container, **kwargs):
        container.regist_period_auto(name='imgsave', period_pair=PERIOD.IMG_SAVE)

    def __init__(self, save_dir: dir, from_model: bool = True, formatter=FORMATTER.SAVE_PTH_SIMPLE, **kwargs):
        self.save_dir = save_dir
        self.from_model = from_model
        self.formatter = formatter
        ensure_folder_pth(save_dir)
        self.kwargs = kwargs

    def act_after_iter(self, container, **kwargs):
        if self.from_model:
            imgs, labels = container.infered_data
        else:
            imgs, labels = container.batch_data

        container.update_time(TIMENODE.BEFORE_IMG_SAVE)
        for img, label in zip(imgs, labels):
            imgP_md = img2imgP(img)
            save_pth = format_save_pth(save_pth=os.path.join(self.save_dir, label.meta), extend='jpg',
                                       formatter=self.formatter)
            imgP_md.save(save_pth, quality=100)
        container.update_time(TIMENODE.AFTER_IMG_SAVE)


# class SimpleAnnotator(Annotator):
#
#     def act_return(self):
#         labels_dct = super(SimpleAnnotator, self).act_return()
#         labels_anno = []
#         for meta in labels_dct.keys():
#             labels_cmb = labels_dct[meta]
#             labels_anno.appendx(labels_cmb[0][1])
#         return labels_anno
#
#
# class CombineAnnotator(Annotator):
#
#     def act_return(self):
#         labels_dct = super(CombineAnnotator, self).act_return()
#         labels_cmb_full = []
#         for meta in labels_dct.keys():
#             labels_cmb = labels_dct[meta]
#             labels_cmb_full += labels_cmb
#         return labels_cmb_full


# </editor-fold>


class LabelMixor(metaclass=ABCMeta):
    @abstractmethod
    def mix(self, labels, **kwargs):
        pass


class NMSBoxesMixor(LabelMixor):
    def __init__(self, iou_thres: float = 0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU,
                 num_presv: int = 10000, by_cls: bool = True):
        self.iou_thres = iou_thres
        self.nms_type = nms_type
        self.iou_type = iou_type
        self.num_presv = num_presv
        self.by_cls = by_cls

    def mix(self, labels, **kwargs):
        label_sum = copy.deepcopy(labels[0])
        for i in range(1, len(labels)):
            label_sum += labels[i]
        xyxysN = label_sum.export_xyxysN()
        confsN = label_sum.export_confsN()
        presv_inds = xyxysN_nms(
            xyxysN, confsN, cindsN=label_sum.export_cindsN() if self.by_cls else None,
            iou_thres=self.iou_thres, nms_type=self.nms_type, iou_type=self.iou_type,
            num_presv=self.num_presv)
        return label_sum[presv_inds]


if __name__ == '__main__':
    pass
