from typing import Sequence

import numpy as np

from utils import *


# <editor-fold desc='分割任务'>

def eval_pair_miou(label_md, label_ds, num_cls: int):
    tp = np.zeros(shape=(num_cls))
    fp = np.zeros(shape=(num_cls))
    tn = np.zeros(shape=(num_cls))
    fn = np.zeros(shape=(num_cls))
    masks_md = label_md.export_masksN_enc(num_cls=num_cls, img_size=label_md.img_size)
    masks_ds = label_ds.export_masksN_enc(num_cls=num_cls, img_size=label_ds.img_size)
    for cind in range(num_cls):
        mask_ds_cind = masks_ds == cind
        mask_md_cind = masks_md == cind
        tp[cind] = np.sum(mask_ds_cind * mask_md_cind)
        tn[cind] = np.sum(~mask_ds_cind * ~mask_md_cind)
        fp[cind] = np.sum(~mask_ds_cind * mask_md_cind)
        fn[cind] = np.sum(mask_ds_cind * ~mask_md_cind)
    return tp, fp, tn, fn


# 省内存版本
def eval_sum_miou_eff(tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                     scale: float = 1000):
    num_cls = tp.shape[-1]
    nlbs_ds = np.zeros(num_cls, dtype=np.int32)
    nlbs_md = np.zeros(num_cls, dtype=np.int32)
    precs = np.zeros(num_cls)
    recls = np.zeros(num_cls)
    f1s = np.zeros(num_cls)
    accs = np.zeros(num_cls)
    ious = np.zeros(num_cls)
    dices = np.zeros(num_cls)

    for cind in range(num_cls):
        tp_cind = np.sum(tp[:, cind] / scale)
        tn_cind = np.sum(tn[:, cind] / scale)
        fp_cind = np.sum(fp[:, cind] / scale)
        fn_cind = np.sum(fn[:, cind] / scale)
        vol = tp_cind + tn_cind + fp_cind + fn_cind
        prec = tp_cind / (tp_cind + fp_cind) if (tp_cind + fp_cind) > 0 else 0
        recl = tp_cind / (tp_cind + fn_cind) if (tp_cind + fn_cind) > 0 else 0
        f1 = 2 / (1 / prec + 1 / recl) if prec > 0 and recl > 0 else 0
        acc = (tp_cind + tn_cind) / vol if vol > 0 else 0
        iou = tp_cind / (tp_cind + fp_cind + fn_cind) if (tp_cind + fp_cind + fn_cind) > 0 else 0
        precs[cind] = prec
        recls[cind] = recl
        f1s[cind] = f1
        accs[cind] = acc
        ious[cind] = iou
        nlbs_ds[cind] = tp_cind + fn_cind
        nlbs_md[cind] = tp_cind + fp_cind
        dices[cind] = 2 * tp / (nlbs_ds[cind] + nlbs_md[cind])
    return nlbs_ds, nlbs_md, precs, recls, f1s, accs, ious, dices

# </editor-fold>
