from .cutting import *
from ..voc import *


# <editor-fold desc='数据集转化'>


# 逐图像地生成分割数据集
def vocD2vocS(dataset: IMDataset, colors, root: str, mask_folder: str = 'SegmentationClass', mask_extend: str = 'png',
              prefix: str = 'Convert Segmentation', broadcast=BROADCAST, ):
    (mask_dir,) = ensure_folders(root, [mask_folder])
    broadcast(dsmsgfmtr_create(root, dataset.set_name, [mask_folder], prefix=prefix))
    for i, (img, boxes) in MEnumerate(dataset, broadcast=broadcast):
        mask_pth = os.path.join(mask_dir, ensure_extend(boxes.meta, mask_extend))
        segs = SegsLabel.convert(boxes)
        VOCSegmentationWriter.save_mask(mask_pth=mask_pth, segs=segs, colors=colors)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return True


# 逐图像地生成分割数据集
def vocD2vocI(dataset: IMDataset, colors, root: str, inst_folder: str = 'SegmentationClass', inst_extend: str = 'png',
              prefix: str = 'Convert Instance', broadcast=BROADCAST, ):
    (inst_dir,) = ensure_folders(root, [inst_folder])
    broadcast(dsmsgfmtr_create(root, dataset.set_name, [inst_folder], prefix=prefix))
    for i, (img, boxes) in MEnumerate(dataset, broadcast=broadcast):
        inst_pth = os.path.join(inst_dir, ensure_extend(boxes.meta, inst_extend))
        insts = InstsLabel.convert(boxes)
        VOCInstanceWriter.save_inst(inst_pth=inst_pth, insts=insts, colors=colors)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return True


# 逐图像地生成分割数据集
def datasetI2vocI(dataset, set_name, colors, root: str, set_folder: str = 'ImageSets/Main',
                  fltr: Optional[Callable] = None,
                  inst_folder: str = 'SegmentationObject', inst_extend: str = 'png', empty_prob: float = 0.0,
                  anno_folder: str = 'Annotations', anno_extend: str = 'xml', img_folder: str = 'JPEGImages',
                  img_extend: str = 'jpg',
                  prefix: str = 'Convert Instance', broadcast=BROADCAST, ):
    folders = [inst_folder, anno_folder, img_folder, set_folder]
    inst_dir, anno_dir, img_dir, set_dir = ensure_folders(root, folders)
    metas = []
    broadcast(dsmsgfmtr_create(root, set_name, folders, prefix=prefix))
    for i, (img, label) in MEnumerate(dataset, broadcast=broadcast):
        label.filt_(fltr=fltr)
        if len(label) == 0 and np.random.rand() < empty_prob:
            continue
        piece_pth = os.path.join(img_dir, ensure_extend(label.meta, img_extend))
        anno_pth = os.path.join(anno_dir, ensure_extend(label.meta, anno_extend))
        inst_pth = os.path.join(inst_dir, ensure_extend(label.meta, inst_extend))
        img2imgP(img).save(piece_pth)
        VOCInstanceWriter.save_anno_inst(anno_pth=anno_pth, inst_pth=inst_pth, colors=colors, insts=label, )
        metas.append(label.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return True


def datasetD2vocD(dataset, set_name, root, set_folder: str = 'ImageSets/Main', fltr: Optional[Callable] = None,
                  empty_prob: float = 0.0, anno_folder: str = 'Annotations', anno_extend: str = 'xml',
                  img_folder: str = 'JPEGImages', img_extend: str = 'jpg', prefix: str = 'Convert Detection',
                  broadcast=BROADCAST, ):
    folders = [anno_folder, img_folder, set_folder]
    anno_dir, img_dir, set_dir = ensure_folders(root, folders)
    metas = []
    broadcast(dsmsgfmtr_create(root, set_name, folders, prefix=prefix))
    for i, (img, label) in MEnumerate(dataset, broadcast=broadcast):
        label.filt_(fltr=fltr)
        if len(label) == 0 and np.random.rand() < empty_prob:
            continue
        piece_pth = os.path.join(img_dir, ensure_extend(label.meta, img_extend))
        anno_pth = os.path.join(anno_dir, ensure_extend(label.meta, anno_extend))
        img2imgP(img).save(piece_pth)
        VOCDetectionWriter.save_anno(anno_pth=anno_pth, boxes=label)
        metas.append(label.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return True


def _datasetI2vocI_cutter(dataset, set_name, colors, root: str, cutter: ImageDataCutter,
                          set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                          inst_folder: Optional[str] = 'SegmentationClass', anno_folder: Optional[str] = 'Annotations',
                          img_extend: str = 'jpg', anno_extend: str = 'xml', inst_extend: str = 'png',
                          prefix: str = 'Crop Instance', broadcast=BROADCAST,
                          fltr: Optional[Callable] = None, func: Optional[Callable] = None, ):
    folders = [inst_folder, anno_folder, img_folder, set_folder]
    inst_dir, anno_dir, img_dir, set_dir = ensure_folders(root, folders)
    broadcast(dsmsgfmtr_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, meta in MEnumerate(dataset.metas, broadcast=broadcast):
        label = dataset._meta2label(meta)
        label.filt_(fltr=fltr)
        if img_dir is not None:
            img = dataset._meta2img(meta)
        else:
            img = None
        pieces, plabels = cutter.cut_data(img, label, )
        for piece, plabel in zip(pieces, plabels):
            if func is not None:
                piece, plabel = func(piece, plabel)
            if img_dir is not None:
                img_pth = os.path.join(img_dir, ensure_extend(plabel.meta, img_extend))
                piece.save(img_pth)
            if anno_dir is not None:
                anno_pth = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
                inst_pth = os.path.join(inst_dir, ensure_extend(plabel.meta, inst_extend))
                VOCInstanceWriter.save_anno_inst(anno_pth=anno_pth, inst_pth=inst_pth, colors=colors, insts=plabel, )
            metas.append(plabel.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return metas


def _datasetD2vocD_cutter(dataset, set_name, root: str, cutter: ImageDataCutter,
                          set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                          anno_folder: Optional[str] = 'Annotations',
                          img_extend: str = 'jpg', anno_extend: str = 'xml',
                          prefix: str = 'Crop Box', broadcast=BROADCAST,
                          fltr: Optional[Callable] = None, func: Optional[Callable] = None, ):
    folders = [anno_folder, img_folder, set_folder]
    anno_dir, img_dir, set_dir = ensure_folders(root, folders)
    broadcast(dsmsgfmtr_create(root, set_name, folders, prefix=prefix))
    metas = []
    for i, meta in MEnumerate(dataset.metas, broadcast=broadcast, step=50):
        label = dataset._meta2label(meta)
        label.filt_(fltr=fltr)
        if img_dir is not None:
            img = dataset._meta2img(meta)
        else:
            img = None
        pieces, plabels = cutter.cut_data(img, label, )
        for piece, plabel in zip(pieces, plabels):
            if func is not None:
                piece, plabel = func(piece, plabel)
            if img_dir is not None:
                img_pth = os.path.join(img_dir, ensure_extend(plabel.meta, img_extend))
                img2imgP(piece).save(img_pth)
            if anno_dir is not None:
                anno_pth = os.path.join(anno_dir, ensure_extend(plabel.meta, anno_extend))
                VOCDetectionWriter.save_anno(anno_pth, boxes=plabel, )
            metas.append(plabel.meta)
    save_txt(os.path.join(set_dir, set_name + '.txt'), lines=metas)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return metas


def datasetI2vocI_persize(dataset, set_name, colors, root: str,
                          set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                          inst_folder: Optional[str] = 'SegmentationClass', anno_folder: Optional[str] = 'Annotations',
                          img_extend: str = 'jpg', anno_extend: str = 'xml', inst_extend: str = 'png',
                          prefix: str = 'Crop Instance', broadcast=BROADCAST, fltr: Optional[Callable] = None,
                          func: Optional[Callable] = None,
                          piece_size: tuple = (640, 640), over_lap: tuple = (100, 100), offset: tuple = (0, 0),
                          empty_prob: float = 0.0, align_border: bool = False, box_protect: bool = False,
                          unique_thres: float = 0.8):
    cutter = ImageDataCutterPerSize(
        piece_size=piece_size, over_lap=over_lap, offset=offset,
        empty_prob=empty_prob, align_border=align_border, box_protect=box_protect, unique_thres=unique_thres)
    metas = _datasetI2vocI_cutter(dataset, set_name, colors, root, cutter, set_folder, img_folder, inst_folder,
                                  anno_folder, img_extend, anno_extend, inst_extend, prefix, broadcast, fltr, func)
    return metas


# 逐检测框地生成分割数据集
def datasetI2vocI_perbox(dataset, set_name, colors, root: str,
                         set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                         inst_folder: Optional[str] = 'SegmentationClass', anno_folder: Optional[str] = 'Annotations',
                         img_extend: str = 'jpg', anno_extend: str = 'xml', inst_extend: str = 'png',
                         prefix: str = 'Crop Instance', broadcast=BROADCAST, fltr: Optional[Callable] = None,
                         func: Optional[Callable] = None,
                         expend_ratio: float = 1.0, expand_min: int = 0, as_square: bool = True,
                         align_border: bool = False, box_protect: bool = False,
                         unique_thres: float = 0.8):
    cutter = ImageDataCutterPerBox(
        expend_ratio=expend_ratio, expand_min=expand_min, as_square=as_square,
        align_border=align_border, box_protect=box_protect, unique_thres=unique_thres)
    metas = _datasetI2vocI_cutter(dataset, set_name, colors, root, cutter, set_folder, img_folder, inst_folder,
                                  anno_folder, img_extend, anno_extend, inst_extend, prefix, broadcast, fltr, func)
    return metas


# 根据标注文件随机裁剪背景
def datasetD2vocI_background(dataset, set_name, colors, root: str,
                             set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                             inst_folder: Optional[str] = 'SegmentationClass',
                             anno_folder: Optional[str] = 'Annotations',
                             img_extend: str = 'jpg', anno_extend: str = 'xml', inst_extend: str = 'png',
                             prefix: str = 'Crop Instance', broadcast=BROADCAST, fltr: Optional[Callable] = None,
                             func: Optional[Callable] = None,
                             min_size: int = 0, max_size: int = 16, num_repeat: int = 1, as_square: bool = True,
                             unique_thres: float = 0.8):
    cutter = ImageDataCutterBackground(
        min_size=min_size, max_size=max_size, num_repeat=num_repeat, as_square=as_square, unique_thres=unique_thres)
    metas = _datasetI2vocI_cutter(dataset, set_name, colors, root, cutter, set_folder, img_folder, inst_folder,
                                  anno_folder, img_extend, anno_extend, inst_extend, prefix, broadcast, fltr, func)
    return metas


def datasetD2vocD_persize(dataset, set_name, root: str,
                          set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                          anno_folder: Optional[str] = 'Annotations', img_extend: str = 'jpg', anno_extend: str = 'xml',
                          prefix: str = 'Crop Box', broadcast=BROADCAST,
                          fltr: Optional[Callable] = None, func: Optional[Callable] = None,
                          piece_size: tuple = (640, 640), over_lap: tuple = (100, 100), offset: tuple = (0, 0),
                          empty_prob: float = 0.0, align_border: bool = False, box_protect: bool = False,
                          unique_thres: float = 0.8):
    cutter = ImageDataCutterPerSize(
        piece_size=piece_size, over_lap=over_lap, offset=offset,
        empty_prob=empty_prob, align_border=align_border, box_protect=box_protect, unique_thres=unique_thres)
    metas = _datasetD2vocD_cutter(dataset, set_name, root, cutter, set_folder, img_folder,
                                  anno_folder, img_extend, anno_extend, prefix, broadcast, fltr, func)
    return metas


def datasetD2vocD_background(dataset, set_name, root: str,
                             set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                             anno_folder: Optional[str] = 'Annotations', img_extend: str = 'jpg',
                             anno_extend: str = 'xml',
                             prefix: str = 'Crop Box', broadcast=BROADCAST,
                             fltr: Optional[Callable] = None, func: Optional[Callable] = None,
                             min_size: int = 0, max_size: int = 16, num_repeat: int = 1, as_square: bool = True,
                             unique_thres: float = 0.8):
    cutter = ImageDataCutterBackground(
        min_size=min_size, max_size=max_size, num_repeat=num_repeat, as_square=as_square, unique_thres=unique_thres)
    metas = _datasetD2vocD_cutter(dataset, set_name, root, cutter, set_folder, img_folder,
                                  anno_folder, img_extend, anno_extend, prefix, broadcast, fltr, func)
    return metas


def datasetD2vocD_perbox(dataset, set_name, root: str,
                         set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                         anno_folder: Optional[str] = 'Annotations', img_extend: str = 'jpg', anno_extend: str = 'xml',
                         prefix: str = 'Crop Box', broadcast=BROADCAST, fltr: Optional[Callable] = None,
                         func: Optional[Callable] = None,
                         expend_ratio: float = 1.0, expand_min: int = 0, as_square: bool = True,
                         align_border: bool = True, box_protect: bool = False,
                         unique_thres: float = 0.8):
    cutter = ImageDataCutterPerBox(
        expend_ratio=expend_ratio, expand_min=expand_min, as_square=as_square,
        align_border=align_border, box_protect=box_protect, unique_thres=unique_thres)
    metas = _datasetD2vocD_cutter(dataset, set_name, root, cutter, set_folder, img_folder,
                                  anno_folder, img_extend, anno_extend, prefix, broadcast, fltr, func)
    return metas


def datasetD2vocD_pyramid(dataset, set_name, root: str,
                          set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                          anno_folder: Optional[str] = 'Annotations', img_extend: str = 'jpg', anno_extend: str = 'xml',
                          prefix: str = 'Crop Box', broadcast=BROADCAST, fltr: Optional[Callable] = None,
                          func: Optional[Callable] = None,
                          piece_sizes: Tuple[Tuple[int, ...], ...] = ((640, 640),),
                          over_laps: Tuple[Tuple[int, ...], ...] = ((100, 100),),
                          offsets: Optional[Tuple[Tuple[int, ...], ...]] = ((0, 0),), empty_prob: float = 0.0,
                          align_border: bool = True,
                          box_protect: bool = False, unique_thres: float = 0.8):
    cutter = ImageDataCutterPyramid(
        piece_sizes=piece_sizes, over_laps=over_laps, offsets=offsets,
        empty_prob=empty_prob, align_border=align_border, box_protect=box_protect, unique_thres=unique_thres)
    metas = _datasetD2vocD_cutter(dataset, set_name, root, cutter, set_folder, img_folder,
                                  anno_folder, img_extend, anno_extend, prefix, broadcast, fltr, func)
    return metas


def datasetD2vocD_from_dir(dataset, set_name, root: str,
                           set_folder: str = 'ImageSets/Main', img_folder: Optional[str] = 'JPEGImages',
                           anno_folder: Optional[str] = 'Annotations', img_extend: str = 'jpg',
                           anno_extend: str = 'xml',
                           prefix: str = 'Crop Box', broadcast=BROADCAST, fltr: Optional[Callable] = None,
                           func: Optional[Callable] = None,
                           ref_dir: str = ''):
    cutter = ImageDataCutterDefine.imitate_from_dir(ref_dir=ref_dir)
    metas = _datasetD2vocD_cutter(dataset, set_name, root, cutter, set_folder, img_folder,
                                  anno_folder, img_extend, anno_extend, prefix, broadcast, fltr, func)
    return metas
# </editor-fold>
