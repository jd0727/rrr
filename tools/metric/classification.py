from typing import Sequence

from utils import *


# <editor-fold desc='分类任务'>


# 分类别计算AUC
def auc_per_class(chots_md: np.ndarray, cinds_ds: np.ndarray, num_cls: int) -> np.ndarray:
    num_samp = len(cinds_ds)
    aucs = np.zeros(num_cls)
    for i in range(num_cls):
        mask_i_ds = cinds_ds == i
        n_ds = (cinds_ds == i).sum()
        n_ds_neg = num_samp - n_ds
        if n_ds == 0 or n_ds_neg == 0:
            aucs[i] = 0
        else:
            # 置信度降序
            confs_cind = chots_md[:, i]
            order = np.argsort(-confs_cind)
            mask_i_ds = mask_i_ds[order]
            # 计算曲线
            tpr_curve = (mask_i_ds).cumsum() / n_ds
            fpr_curve = (~mask_i_ds).cumsum() / n_ds_neg
            # 计算面积
            fpr_curve = np.concatenate(([0.0], fpr_curve))
            fpr_dt = fpr_curve[1:] - fpr_curve[:-1]
            aucs[i] = np.sum(fpr_dt * tpr_curve)
    aucs = np.array(aucs)
    return aucs


def confusion_per_class(cinds_md: np.ndarray, cinds_ds: np.ndarray, num_cls: int):
    tp = np.zeros(num_cls, dtype=np.int32)
    tn = np.zeros(num_cls, dtype=np.int32)
    fp = np.zeros(num_cls, dtype=np.int32)
    fn = np.zeros(num_cls, dtype=np.int32)
    for cind in range(num_cls):
        mask_i_ds = cinds_ds == cind
        mask_i_md = cinds_md == cind
        tp[cind] = np.sum(mask_i_ds * mask_i_md)
        tn[cind] = np.sum(~mask_i_ds * ~mask_i_md)
        fp[cind] = np.sum(~mask_i_ds * mask_i_md)
        fn[cind] = np.sum(mask_i_ds * ~mask_i_md)
    return tp, tn, fp, fn


def f1_from_confusion(tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray):
    prec = tp / np.clip(tp + fp, a_min=1, a_max=None)
    recl = tp / np.clip(tp + fn, a_min=1, a_max=None)
    f1 = 2 * prec * recl / np.clip(prec + recl, a_min=1e-7, a_max=None)
    acc = (tp + tn) / np.clip(tp + tn + fn + fp, a_min=1, a_max=None)
    return prec, recl, f1, acc


# 分类计算prec
def prec_recl_per_class(cinds_md: np.ndarray, cinds_ds: np.ndarray, num_cls: int):
    nlbs_ds = np.zeros(num_cls, dtype=np.int32)
    nlbs_md = np.zeros(num_cls, dtype=np.int32)
    tps = np.zeros(num_cls, dtype=np.int32)
    precs = np.zeros(num_cls)
    recls = np.zeros(num_cls)
    f1s = np.zeros(num_cls)
    accs = np.zeros(num_cls)
    for cind in range(num_cls):
        mask_i_ds = cinds_ds == cind
        mask_i_md = cinds_md == cind
        nlbs_ds[cind] = np.sum(mask_i_ds)
        nlbs_md[cind] = np.sum(mask_i_md)
        tp = np.sum(mask_i_ds * mask_i_md)
        tn = np.sum(~mask_i_ds * ~mask_i_md)
        fp = np.sum(~mask_i_ds * mask_i_md)
        fn = np.sum(mask_i_ds * ~mask_i_md)

        prec = tp / max(tp + fp, 1)
        recl = tp / max(tp + fn, 1)
        f1 = 2 * prec * recl / max(prec + recl, 1e-7)
        acc = (tp + tn) / max(len(cinds_ds), 1)
        precs[cind] = prec
        recls[cind] = recl
        f1s[cind] = f1
        accs[cind] = acc
        tps[cind] = tp
    return nlbs_ds, nlbs_md, tps, precs, recls, f1s, accs


def eval_pair_tpacc(label_md, label_ds, top_nums: Tuple[int, ...] = (1, 5)):
    chot_md = OneHotCategory.convert(label_md.category)._chotN
    cind_ds = label_ds.category.cindN
    max_top = max(top_nums)
    cinds_ct_md = np.argsort(-chot_md, axis=0)[:max_top]
    return cinds_ct_md, cind_ds


# top精度
def acc_top_nums(cinds_ct_md: np.ndarray, cinds_ds: np.ndarray, num_cls: int,
                 top_nums: Tuple[int, ...] = (1, 5)) -> (np.ndarray, np.ndarray):
    ns_ds = np.zeros(shape=num_cls, dtype=np.int32)
    tps = np.zeros(shape=(num_cls, len(top_nums)), dtype=np.int32)

    for j, cind_ds in enumerate(cinds_ds):
        ns_ds[cind_ds] += 1
        for i, num in enumerate(top_nums):
            if cind_ds in cinds_ct_md[j, :num]:
                tps[cind_ds, i] += 1

    accs = tps / ns_ds[:, None]
    return ns_ds, accs

# </editor-fold>
