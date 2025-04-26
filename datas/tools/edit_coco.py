from .cutting import *
from ..coco import *


# <editor-fold desc='数据集转化'>

def datasetI2cocoI(dataset: IMDataset, root: str, json_name: str,
                   img_folder: str = 'images', json_folder: str = 'annotation', img_extend: str = 'jpg',
                   name2id: Optional[Callable] = None, prefix: str = 'Create', broadcast=BROADCAST,
                   fltr: Optional[Callable] = None, empty_prob: float = 0.0, ):
    folders = [img_folder, json_folder]
    img_dir, json_dir = ensure_folders(root, folders)
    json_pth = ensure_extend(os.path.join(json_dir, json_name), 'json')
    broadcast(dsmsgfmtr_create(root, set_name=json_name, folders=folders, prefix=prefix))
    labels_all = []
    for i, (img, label) in MEnumerate(dataset, broadcast=broadcast):
        label.filt_(fltr)
        if len(label) == 0 and np.random.rand() < empty_prob:
            continue
        img_pth = os.path.join(img_dir, ensure_extend(label.meta, img_extend))
        img.save(img_pth)
        labels_all.append(label)

    json_dct = COCOWriter.labels2json_dct(labels=labels_all, name2id=name2id, img_extend=img_extend)
    save_json(json_pth, json_dct)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return labels_all


# 逐检测框地生成分割数据集
def _datasetI2cocoI_cutter(dataset: IMDataset, root: str, json_name: str, cutter: ImageDataCutter,
                           img_folder: str = 'images', json_folder: str = 'annotation', img_extend: str = 'jpg',
                           name2id: Optional[Callable] = None, prefix: str = 'Create', broadcast=BROADCAST,
                           fltr: Optional[Callable] = None, func: Optional[Callable] = None, ):
    folders = [img_folder, json_folder]
    img_dir, json_dir = ensure_folders(root, folders)
    json_pth = ensure_extend(os.path.join(json_dir, json_name), 'json')
    broadcast(dsmsgfmtr_create(root, set_name=json_name, folders=folders, prefix=prefix))
    plabels_all = []
    metas = []
    for i, (img, label) in MEnumerate(dataset, broadcast=broadcast):
        label.filt_(fltr=fltr)
        pieces, plabels = cutter.cut_data(img, label, )
        for piece, plabel in zip(pieces, plabels):
            patch_pth = os.path.join(img_dir, ensure_extend(plabel.meta, img_extend))
            piece.save(patch_pth)
            plabels_all.append(plabel)
            metas.append(plabel.meta)
    json_dct = COCOWriter.labels2json_dct(labels=plabels_all, name2id=name2id, img_extend=img_extend)
    save_json(json_pth, json_dct)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return metas


# 按尺寸分割数据集
def datasetI2cocoI_persize(dataset: IMDataset, root: str, json_name: str,
                           img_folder: str = 'images', json_folder: str = 'annotation', img_extend: str = 'jpg',
                           name2cind_remapper: Optional[Callable] = None, prefix: str = 'Create', broadcast=BROADCAST,
                           fltr: Optional[Callable] = None, func: Optional[Callable] = None,
                           piece_size: tuple = (640, 640), over_lap: tuple = (100, 100), offset: tuple = (0, 0),
                           empty_prob: float = 0.0, align_border: bool = False, box_protect: bool = False,
                           unique_thres: float = 0.8):
    cutter = ImageDataCutterPerSize(
        piece_size=piece_size, over_lap=over_lap, offset=offset,
        empty_prob=empty_prob, align_border=align_border, box_protect=box_protect, unique_thres=unique_thres)
    metas = _datasetI2cocoI_cutter(dataset, root, json_name, cutter, img_folder, json_folder, img_extend,
                                   name2cind_remapper, prefix, broadcast, fltr, func)
    return metas

# 按提取子类


# </editor-fold>
