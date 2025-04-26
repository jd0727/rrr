from typing import Sequence

from utils import *


def label_ropr_mat_inst(label_md, label_ds, opr_type=OPR_TYPE.IOU):
    xyxys_md, cinds_md, masks_md = label_md.export_rgn_xyxysN_maskNs_ref()
    xyxys_ds, cinds_ds, masks_ds = label_ds.export_rgn_xyxysN_maskNs_ref()
    iou_mat = xyxysN_maskNs_ropr(xyxysN1=xyxys_md[:, None, :], xyxysN2=xyxys_ds[None, :, :],
                                 maskNs1=masks_md[:, None, :, :], maskNs2=masks_ds[None, :, :, :], opr_type=opr_type)
    return iou_mat


def label_ropr_mat_box(label_md, label_ds, opr_type=OPR_TYPE.IOU):
    if np.any([isinstance(box.border, XYWHABorder) for box in label_ds]) or \
            np.any([isinstance(box.border, XYWHABorder) for box in label_md]):
        xywhas_ds = label_ds.export_xywhasN()
        xywhas_md = label_md.export_xywhasN()
        iou_mat = xywhaN_ropr(xywhas_md[:, None, :], xywhas_ds[None, :, :], opr_type=opr_type)
    else:
        xyxys_ds = label_ds.export_xyxysN()
        xyxys_md = label_md.export_xyxysN()
        iou_mat = xyxyN_ropr(xyxys_md[:, None, :], xyxys_ds[None, :, :], opr_type=opr_type)
    return iou_mat


def eval_pair_vocap(label_md, label_ds, label_ropr_mat: Callable, iou_thres: float = 0.5, ignore_class: bool = False):
    confs_md = label_md.export_confsN()
    order = np.argsort(-confs_md)
    label_md = label_md[order]
    confs_md = confs_md[order]
    cinds_md = label_md.export_cindsN()
    cinds_ds = label_ds.export_cindsN()
    ignores_ds = label_ds.export_ignoresN()
    iou_mat = label_ropr_mat(label_md, label_ds)

    mask_pos_i, mask_neg_i = ap_match_core(
        cinds_md, cinds_ds, ignores_ds, iou_mat, iou_thres=iou_thres, ignore_class=ignore_class)
    return cinds_ds[~ignores_ds], cinds_md, confs_md, mask_pos_i, mask_neg_i


def eval_pair_cocoap(label_md, label_ds, label_ropr_mat: Callable, iou_thress: Sequence[float] = (0.5, 0.55, 0.6),
                     ignore_class: bool = False):
    confs_md = label_md.export_confsN()
    order = np.argsort(-confs_md)
    label_md = label_md[order]
    confs_md = confs_md[order]
    cinds_md = label_md.export_cindsN()
    cinds_ds = label_ds.export_cindsN()
    ignores_ds = label_ds.export_ignoresN()
    iou_mat = label_ropr_mat(label_md, label_ds)
    mask_md_pos = []
    mask_md_neg = []

    for j, iou_thres in enumerate(iou_thress):
        iou_mat_j = copy.deepcopy(iou_mat)
        mask_pos_i, mask_neg_i = ap_match_core(
            cinds_md, cinds_ds, ignores_ds, iou_mat_j, iou_thres=iou_thres, ignore_class=ignore_class)
        mask_md_pos.append(mask_pos_i)
        mask_md_neg.append(mask_neg_i)
    mask_md_pos = np.stack(mask_md_pos, axis=-1)
    mask_md_neg = np.stack(mask_md_neg, axis=-1)
    return cinds_ds[~ignores_ds], cinds_md, confs_md, mask_md_pos, mask_md_neg


def confusion_per_class_det(cinds_md: np.ndarray, masks_md_pos: np.ndarray,
                            masks_md_neg: np.ndarray, num_cls: int):
    tp = np.zeros(num_cls, dtype=np.int32)
    tn = np.zeros(num_cls, dtype=np.int32)
    fp = np.zeros(num_cls, dtype=np.int32)
    fn = np.zeros(num_cls, dtype=np.int32)
    for cind in range(num_cls):
        mask_i_md = cinds_md == cind
        tp[cind] = np.sum(masks_md_pos[mask_i_md])
        tn[cind] = np.sum(masks_md_pos[~mask_i_md])
        fp[cind] = np.sum(masks_md_neg[mask_i_md])
        fn[cind] = np.sum(masks_md_neg[~mask_i_md])
    return tp, tn, fp, fn


def precrecl_from_confusion_det(tp, n_md, n_ds):
    prec = tp / np.clip(n_md, a_min=1, a_max=None)
    recl = tp / np.clip(n_ds, a_min=1, a_max=None)
    f1 = 2 * prec * recl / np.clip(prec + recl, a_min=1e-7, a_max=None)
    return prec, recl, f1


# <editor-fold desc='检测任务'>
def ap_match_core(cinds_md: np.ndarray, cinds_ds: np.ndarray, ignores_ds: np.ndarray, iou_mat: np.ndarray,
                  iou_thres: float = 0.5, ignore_class: bool = False) -> (np.ndarray, np.ndarray):
    mask_pos = np.full(shape=len(cinds_md), fill_value=False)
    mask_neg = np.full(shape=len(cinds_md), fill_value=False)
    if len(cinds_ds) == 0:
        return mask_pos, ~mask_neg
    elif len(cinds_md) == 0:
        return mask_pos, mask_neg
    else:
        for k in range(len(cinds_md)):
            if not ignore_class:  # 不同分类之间不能匹配比较
                iou_mat[k, ~(cinds_md[k] == cinds_ds)] = 0
                ind_ds = np.argmax(iou_mat[k, :])
            else:  # 动态改变类别
                ind_ds = np.argmax(iou_mat[k, :])
                cinds_md[k] = cinds_ds[ind_ds]
            if iou_mat[k, ind_ds] > iou_thres:
                if not ignores_ds[ind_ds]:
                    mask_pos[k] = True
                iou_mat[:, ind_ds] = 0  # 防止重复匹配
            else:
                mask_neg[k] = True
        return mask_pos, mask_neg


def ap_match_core_v2(cinds_md: np.ndarray, cinds_ds: np.ndarray, ignores_ds: np.ndarray, iou_mat: np.ndarray,
                     iou_thres: float = 0.5, ignore_class: bool = False) -> (np.ndarray, np.ndarray):
    mask_pos = np.full(shape=len(cinds_md), fill_value=False)
    mask_neg = np.full(shape=len(cinds_md), fill_value=False)
    if len(cinds_ds) == 0:
        return mask_pos, ~mask_neg
    elif len(cinds_md) == 0:
        return mask_pos, mask_neg
    else:
        fltr_iou = (iou_mat >= iou_thres)
        if not ignore_class:
            fltr_cls = cinds_md[:, None] == cinds_ds[None, :]
            fltr_iou *= fltr_cls
        ids_md, ids_ds = np.nonzero(fltr_iou)
        ious = iou_mat[ids_md, ids_ds]
        order = np.argsort(-ious)
        ids_md, ids_ds, ious = ids_md[order], ids_ds[order], ious[order]
        fltr_repeat = np.unique(ids_ds, return_index=True)[1]
        ids_md, ids_ds, ious = ids_md[fltr_repeat], ids_ds[fltr_repeat], ious[fltr_repeat]
        fltr_repeat = np.unique(ids_md, return_index=True)[1]
        ids_md, ids_ds, ious = ids_md[fltr_repeat], ids_ds[fltr_repeat], ious[fltr_repeat]
        mask_pos[ids_md] = True
        mask_neg = ~mask_pos * ~ignores_ds
        return mask_pos, mask_neg


# 分类别计算AP
def ap_per_class(cinds_ds: np.ndarray, cinds_md: np.ndarray, confs_md: np.ndarray,
                 mask_pos: np.ndarray, mask_neg: np.ndarray, num_cls: int = 20, interp: bool = False) -> np.ndarray:
    order = np.argsort(-confs_md)
    mask_pos, mask_neg, confs_md, cinds_md = mask_pos[order], mask_neg[order], confs_md[order], cinds_md[order]
    aps = np.zeros(num_cls)
    for cind in range(num_cls):
        mask_pred_pos = cinds_md == cind
        num_gt = (cinds_ds == cind).sum()
        num_pred = mask_pred_pos.sum()
        if num_pred == 0 or num_gt == 0:
            aps[cind] = 0
        else:
            fp_nums = (mask_neg[mask_pred_pos]).cumsum()  # 累加和列表
            tp_nums = (mask_pos[mask_pred_pos]).cumsum()
            # 计算曲线
            recall_curve = tp_nums / (num_gt + 1e-16)
            precision_curve = tp_nums / (tp_nums + fp_nums + 1e-16)
            # 计算面积
            recall_curve = np.concatenate(([0.0], recall_curve, [1.0]))
            precision_curve = np.concatenate(([1.0], precision_curve, [0.0]))

            precision_curve = np.flip(np.maximum.accumulate(np.flip(precision_curve)))
            if interp:
                x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
                ap = np.trapz(np.interp(x, recall_curve, precision_curve), x)  # integrate
            else:  # 'continuous'
                ids = np.where(recall_curve[1:] != recall_curve[:-1])[0]  # points where x axis (recall) changes
                ap = np.sum((recall_curve[ids + 1] - recall_curve[ids]) * precision_curve[ids + 1])  # area under curve
            aps[cind] = ap

            # for i in range(precision_curve.size - 1, 0, -1):
            #     precision_curve[i - 1] = np.maximum(precision_curve[i - 1], precision_curve[i])
            # aps[cind] = np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:])
    aps = np.array(aps)
    return aps

# </editor-fold>
