from .cutting import *
from ..base import *


def _datasetD2folderC_cutter(dataset: IMDataset, root: str, cutter: ImageDataCutter, img_extend: str = 'jpg',
                             prefix: str = 'Crop Box', broadcast=BROADCAST,
                             fltr: Optional[Callable] = None, func: Optional[Callable] = None, ):
    ensure_folder_pth(root)
    broadcast(dsmsgfmtr_create(root, dataset.set_name, tuple(), prefix=prefix))
    metas = []
    for i, (img, label) in MEnumerate(dataset, broadcast=broadcast, step=50):
        label.filt_(fltr=fltr)
        pieces, plabels = cutter.cut_data(img, label, )
        for piece, plabel in zip(pieces, plabels):
            if len(plabel) > 0:
                idx = np.argmax(xyxyN2areaN(plabel.xyxysN))
                name = plabel[idx]['name']
            else:
                name = 'background'
            ensure_folder_pth(os.path.join(root, name))
            patch_pth = os.path.join(root, name, ensure_extend(plabel.meta, img_extend))
            if func is not None:
                piece, plabel = func(piece, plabel)
            metas.append(plabel.meta)
            img2imgP(piece).save(patch_pth)
    broadcast(dsmsgfmtr_end(prefix=prefix))
    return metas


def datasetD2folderC_perboxsingle(dataset: IMDataset, root: str, img_extend: str = 'jpg',
                                  prefix: str = 'Crop Box',
                                  broadcast=BROADCAST,
                                  fltr: Optional[Callable] = None, func: Optional[Callable] = None,
                                  expend_ratio: float = 1.0, expand_min: int = 0, as_square: bool = True,
                                  align_border: bool = True, ):
    cutter = ImageDataCutterPerBoxSingle(
        expend_ratio=expend_ratio, expand_min=expand_min, as_square=as_square, align_border=align_border, )
    metas = _datasetD2folderC_cutter(dataset, root, cutter=cutter,
                                     img_extend=img_extend, prefix=prefix, fltr=fltr, func=func, broadcast=broadcast)
    return metas


def datasetD2folderC_background(dataset: IMDataset, root: str, img_extend: str = 'jpg',
                                prefix: str = 'Crop Background',
                                broadcast=BROADCAST,
                                fltr: Optional[Callable] = None, func: Optional[Callable] = None,
                                min_size: int = 0, max_size: int = 16, num_repeat: int = 1, as_square: bool = True,
                                unique_thres: float = 0.8, ):
    cutter = ImageDataCutterBackground(
        min_size=min_size, max_size=max_size, num_repeat=num_repeat, as_square=as_square, unique_thres=unique_thres)
    metas = _datasetD2folderC_cutter(dataset, root, cutter=cutter,
                                     img_extend=img_extend, prefix=prefix, fltr=fltr, func=func, broadcast=broadcast)
    return metas
