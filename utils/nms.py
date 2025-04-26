import torchvision

from .label import *


# <editor-fold desc='numpy nms原型'>

class CLUSTER_INDEX:
    NONE = 'none'
    CLASS = 'class'


class NMS_TYPE:
    HARD = 'hard'
    NONE = 'none'


def remap_cindsN(cindsN: np.ndarray, cluster_index: Union[np.ndarray, str]) -> Union[np.ndarray, None]:
    if cluster_index is None or (isinstance(cluster_index,str) and cluster_index == CLUSTER_INDEX.NONE):
        return None
    elif isinstance(cluster_index,str) and cluster_index == CLUSTER_INDEX.CLASS:
        return cindsN.astype(np.int32)
    else:
        cluster_index = np.array(cluster_index).astype(np.int32)
        return cluster_index[cindsN.astype(np.int32)]


def _nmsN(bordersN: np.ndarray, confsN: np.ndarray, roprN: Callable, cindsN: Optional[np.ndarray] = None,
          iou_thres: float = 0.45, iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD,
          num_presv: int = 10000) -> np.ndarray:
    if bordersN.shape[0] == 0:
        return np.zeros(shape=0, dtype=np.int32)
    if nms_type == NMS_TYPE.NONE:
        prsv_inds = np.arange(bordersN.shape[0], dtype=np.int32)
        return prsv_inds
    if cindsN is None:
        order = np.argsort(-confsN)[:num_presv]
        bordersN = bordersN[order]
        flags = confsN[order].astype(np.float32)
        prsv_inds = []
        for i in range(bordersN.shape[0]):
            if np.isnan(flags[i]):
                continue
            prsv_inds.append(order[i])
            res_inds = i + 1 + np.nonzero(~np.isnan(flags[i + 1:]))[0]
            if len(res_inds) == 0:
                break
            bordersN1 = np.repeat(bordersN[i:i + 1], repeats=len(res_inds), axis=0)
            ious = roprN(bordersN1, bordersN[res_inds], opr_type=iou_type)
            flags[res_inds[ious > iou_thres]] = np.nan
        prsv_inds = np.array(prsv_inds)
        return prsv_inds
    else:
        prsv_inds = []
        num_cls = int(np.max(cindsN))
        for i in range(num_cls + 1):
            inds = cindsN == i
            if np.any(inds):
                prsv_inds_cls = _nmsN(bordersN[inds], confsN[inds], roprN, cindsN=None, iou_thres=iou_thres,
                                      iou_type=iou_type, num_presv=num_presv)
                inds = np.nonzero(inds)[0]
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = np.concatenate(prsv_inds, axis=0)
    return prsv_inds


def xyxysN_nms(xyxysN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
               iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    return _nmsN(xyxysN, confsN, xyxyN_ropr, cindsN, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xyzxyzsN_nms(xyzxyzsN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
                 iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    return _nmsN(xyzxyzsN, confsN, xyzxyzN_ropr, cindsN, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xywhsN_nms(xywhsN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
               iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    return _nmsN(xywhsN, confsN, xywhN_ropr, cindsN, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xyzwhlsN_nms(xyzwhlsN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
                 iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    return _nmsN(xyzwhlsN, confsN, xyzwhlN_ropr, cindsN, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xywhasN_nms(xywhasN: np.ndarray, confsN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
                iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    return _nmsN(xywhasN, confsN, xywhaN_ropr, cindsN, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xypNs_nms(xypNs: List[np.ndarray], confsN: np.ndarray, cindsN: np.ndarray = None, iou_thres=0.45,
              iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    if len(xypNs) == 0:
        return np.zeros(shape=0, dtype=np.int32)
    if nms_type == NMS_TYPE.NONE:
        prsv_inds = np.arange(len(xypNs), dtype=np.int32)
        return prsv_inds

    if cindsN is None:
        order = np.argsort(-confsN)[:num_presv]
        xypNs = [xypNs[ind] for ind in order]
        flags = confsN[order].astype(np.float32)
        prsv_inds = []
        for i in range(len(xypNs)):
            if np.isnan(flags[i]):
                continue
            prsv_inds.append(order[i])
            res_inds = i + 1 + np.nonzero(~np.isnan(flags[i + 1:]))[0]
            if len(res_inds) == 0:
                break
            xyps1 = xypNs[i:i + 1] * len(res_inds)
            xyps2 = [xypNs[ind] for ind in res_inds]
            ious = xypNs_ropr(xyps1, xyps2, opr_type=iou_type)
            flags[res_inds[ious > iou_thres]] = np.nan
        prsv_inds = np.array(prsv_inds)
    else:
        prsv_inds = []
        num_cls = int(np.max(cindsN))
        for i in range(num_cls + 1):
            inds = np.nonzero(cindsN == i)[0]
            if len(inds) > 0:
                xyps_cls = [xypNs[ind] for ind in inds]
                prsv_inds_cls = xypNs_nms(
                    xyps_cls, confsN[inds], cindsN=None, iou_thres=iou_thres, iou_type=iou_type, num_presv=num_presv)
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = np.concatenate(prsv_inds, axis=0)
    return prsv_inds
    # boxes2


def xysN_nms(xysN: np.ndarray, confsN: np.ndarray, radius: float = 1.0, num_presv=10000) -> np.ndarray:
    if xysN.shape[0] == 0:
        return np.zeros(shape=0, dtype=np.int32)
    order = np.argsort(-confsN)[:num_presv]
    xysN, confsN = xysN[order], confsN[order]
    prsv_inds = []
    for i in range(len(xysN)):
        if confsN[i] == 0:
            continue
        prsv_inds.append(order[i])
        res_inds = i + 1 + np.nonzero(confsN[i + 1:] > 0)[0]
        xys_prsv = xysN[res_inds]
        dists = np.linalg.norm(xysN[i:i + 1] - xys_prsv, axis=1)
        confsN[res_inds[dists < radius]] = 0
    prsv_inds = np.array(prsv_inds)
    return prsv_inds


def xyxysN_nms_byarea(xyxysN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
                      iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    areas = xyxyN2areaN(xyxysN)
    return _nmsN(xyxysN, confsN=areas, roprN=xyxyN_ropr, cindsN=cindsN, iou_thres=iou_thres, iou_type=iou_type,
                 nms_type=nms_type, num_presv=num_presv)


def xywhsN_nms_byarea(xywhsN: np.ndarray, cindsN: np.ndarray = None, iou_thres: float = 0.45,
                      iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> np.ndarray:
    areas = xywhN2areaN(xywhsN)
    return _nmsN(xywhsN, confsN=areas, roprN=xywhN_ropr, cindsN=cindsN, iou_thres=iou_thres, iou_type=iou_type,
                 nms_type=nms_type, num_presv=num_presv)


# </editor-fold>

# <editor-fold desc='label nms原型'>
class NMS_ORDERBY:
    CONF = 'conf'
    AREA = 'area'


_BORDER_CMPLX = {
    XYXYBorder: 0,
    XYWHBorder: 0,
    XYWHABorder: 1,
    XYPBorder: 2,
}


def _select_border_type(boxes: BoxesLabel):
    border_type = XYXYBorder
    cmplx = _BORDER_CMPLX[border_type]
    for box in boxes:
        border_type_cur = box.border.__class__
        cmplx_cur = _BORDER_CMPLX[border_type_cur]
        if cmplx_cur > cmplx:
            border_type = border_type_cur
            cmplx = cmplx_cur
    return border_type


def items_nms(items: ImageItemsLabel, border_type=None, iou_thres: float = 0.0, cluster_index=CLUSTER_INDEX.CLASS,
              iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, nms_orderby=NMS_ORDERBY.CONF,
              num_presv: int = 10000) -> ImageItemsLabel:
    if iou_thres <= 0:
        return items.empty()
    boxes = BoxesLabel.convert(items)
    border_type = border_type if border_type is not None else _select_border_type(boxes)
    cindsN = boxes.export_cindsN()
    cindsN = remap_cindsN(cindsN, cluster_index=cluster_index)
    if nms_orderby == NMS_ORDERBY.CONF:
        confsN = boxes.export_confsN()
        if border_type == XYXYBorder or border_type == XYWHBorder:
            xyxysN = boxes.export_xyxysN()
            prsv_inds = xyxysN_nms(xyxysN, confsN, cindsN=cindsN, iou_thres=iou_thres,
                                   nms_type=nms_type, iou_type=iou_type, num_presv=num_presv)
            return items[prsv_inds]
        else:
            xypNs = boxes.export_xypNs()
            prsv_inds = xypNs_nms(xypNs, confsN, cindsN=cindsN, iou_thres=iou_thres,
                                  nms_type=nms_type, iou_type=iou_type, num_presv=num_presv)
            return items[prsv_inds]
    elif nms_orderby == NMS_ORDERBY.AREA:
        if border_type == XYXYBorder or border_type == XYWHBorder:
            xyxysN = boxes.export_xyxysN()
            prsv_inds = xyxysN_nms_byarea(xyxysN, cindsN=cindsN, iou_thres=iou_thres,
                                          nms_type=nms_type, iou_type=iou_type, num_presv=num_presv)
            return items[prsv_inds]
        else:
            raise Exception('no impl')
    else:
        raise Exception('err')


# </editor-fold>

# <editor-fold desc='torch nms原型'>
def remap_cindsT(cindsT: torch.Tensor, cluster_index: Union[np.ndarray, str]) -> Union[torch.Tensor, None]:
    if cluster_index is None or (isinstance(cluster_index,str) and cluster_index == CLUSTER_INDEX.NONE):
        return None
    elif isinstance(cluster_index,str) and cluster_index == CLUSTER_INDEX.CLASS:
        return cindsT.long()
    else:
        cluster_index = arrsN2arrsT(np.array(cluster_index), device=cindsT.device)
        return cluster_index[cindsT].long()


def _nmsT(bordersT: torch.Tensor, confsT: torch.Tensor, roprT: Callable, cindsT: Optional[torch.Tensor] = None,
          iou_thres: float = 0.45, iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD,
          num_presv: int = 10000) -> torch.Tensor:
    if confsT.size(0) == 0:
        return torch.zeros((0,), dtype=torch.long, device=bordersT.device)
    if nms_type == NMS_TYPE.NONE:
        return torch.arange(bordersT.size(0), dtype=torch.long, device=bordersT.device)
    if cindsT is None and iou_type == IOU_TYPE.IOU and roprT == xyxyT_ropr:
        if num_presv < confsT.size(0):
            order = torch.argsort(confsT, descending=True)[:num_presv]
            bordersT, confsT = bordersT[order], confsT[order]
            inds = torchvision.ops.nms(bordersT, confsT, iou_threshold=iou_thres)
            return order[inds]
        else:
            return torchvision.ops.nms(bordersT, confsT, iou_threshold=iou_thres)
    elif cindsT is None:
        order = torch.argsort(confsT, descending=True)[:num_presv]
        bordersT = bordersT[order]
        flags = confsT[order].float()
        prsv_inds = []
        for i in range(bordersT.size(0)):
            if torch.isnan(confsT[i]):
                continue
            prsv_inds.append(order[i])
            res_inds = i + 1 + torch.nonzero(~torch.isnan(flags[i + 1:]), as_tuple=True)[0]
            if len(res_inds) == 0:
                break
            boxesT1 = bordersT[i:i + 1].repeat(len(res_inds), 1)
            ious = roprT(boxesT1, bordersT[res_inds], opr_type=iou_type)
            flags[res_inds[ious > iou_thres]] = torch.nan
        prsv_inds = torch.Tensor(prsv_inds).long()

    else:
        prsv_inds = []
        num_cls = int(torch.max(cindsT).item())
        for i in range(num_cls + 1):
            inds = cindsT == i
            if torch.any(inds):
                prsv_inds_cls = _nmsT(bordersT[inds], confsT[inds], roprT, cindsT=None, iou_thres=iou_thres,
                                      iou_type=iou_type, num_presv=num_presv)
                inds = torch.nonzero(inds, as_tuple=False).squeeze(dim=1)
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = torch.cat(prsv_inds, dim=0)
    return prsv_inds


def xyxysT_nms(xyxysT: torch.Tensor, confsT: torch.Tensor, cindsT: Optional[torch.Tensor] = None,
               iou_thres: float = 0.45,
               iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> torch.Tensor:
    return _nmsT(xyxysT, confsT, xyxyT_ropr, cindsT, iou_thres, iou_type, nms_type=nms_type, num_presv=num_presv)


def xyzxyzsT_nms(xyzxyzsT: torch.Tensor, confsT: torch.Tensor, cindsT: Optional[torch.Tensor] = None,
                 iou_thres: float = 0.45,
                 iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> torch.Tensor:
    return _nmsT(xyzxyzsT, confsT, xyzxyzT_ropr, cindsT, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xywhsT_nms(xywhsT: torch.Tensor, confsT: torch.Tensor, cindsT: Optional[torch.Tensor] = None,
               iou_thres: float = 0.45,
               iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> torch.Tensor:
    xyxysT = xywhT2xyxyT(xywhsT)
    return _nmsT(xyxysT, confsT, xyxyT_ropr, cindsT, iou_thres, iou_type, nms_type=nms_type, num_presv=num_presv)


def xyzwhlsT_nms(xyzwhlsT: torch.Tensor, confsT: torch.Tensor, cindsT: Optional[torch.Tensor] = None,
                 iou_thres: float = 0.45,
                 iou_type=IOU_TYPE.IOU, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> torch.Tensor:
    return _nmsT(xyzwhlsT, confsT, xyzwhlT_ropr, cindsT, iou_thres, iou_type, nms_type=nms_type,
                 num_presv=num_presv)


def xywhasT_nms(xywhasT: torch.Tensor, confsT: torch.Tensor, cindsT: Optional[torch.Tensor] = None,
                iou_thres: float = 0.45,
                nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, num_presv: int = 10000) -> torch.Tensor:
    return _nmsT(xywhasT, confsT, xywhaT_ropr, cindsT, iou_thres, iou_type, nms_type=nms_type, num_presv=num_presv)


def xysT_dlsT_nms(xysT: torch.Tensor, dlsT: torch.Tensor, confsT: torch.Tensor, cindsT: Optional[torch.Tensor] = None,
                  iou_thres: float = 0.0, nms_type=NMS_TYPE.HARD, num_presv: int = 10000) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    if xysT.size(0) == 0:
        prsv_inds = torch.zeros((0,), dtype=torch.long, device=dlsT.device)
        dls_clpd = torch.zeros_like(dlsT, dtype=torch.float, device=dlsT.device)
        return prsv_inds, dls_clpd
    if nms_type == NMS_TYPE.NONE:
        prsv_inds = torch.arange(xysT.size(0), dtype=torch.long, device=dlsT.device)
        return prsv_inds, dlsT

    if cindsT is None:
        return _xysT_dlsT_nms(xysT, dlsT, confsT, num_presv=num_presv, iou_thres=iou_thres)
    else:
        prsv_inds, dls_clpd = [], []
        num_cls = int(torch.max(cindsT).item())
        for i in range(num_cls + 1):
            inds = torch.nonzero((cindsT == i).bool(), as_tuple=False).squeeze(dim=1)
            if inds.size(0) > 0:
                prsv_inds_cls, dls_clpd_cls = _xysT_dlsT_nms(xysT[inds], dlsT[inds], confsT[inds], num_presv=num_presv,
                                                             iou_thres=iou_thres)
                prsv_inds.append(inds[prsv_inds_cls])
                dls_clpd.append(dls_clpd_cls)
        prsv_inds = torch.cat(prsv_inds, dim=0)
        dls_clpd = torch.cat(dls_clpd, dim=0)
    return prsv_inds, dls_clpd


def _xysT_dlsT_nms(xysT: torch.Tensor, dlsT: torch.Tensor, confsT: torch.Tensor,
                   iou_thres: float = 0.0, num_presv: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
    confsT = copy.deepcopy(confsT)  # 隔离
    order = torch.argsort(confsT, descending=True)[:num_presv]
    xysT, dlsT, confsT = xysT[order], dlsT[order], confsT[order]
    prsv_inds = []
    dls_clpd = [torch.zeros(size=(0, dlsT.size(-1)), device=dlsT.device)]
    for i in range(xysT.size(0)):
        if confsT[i] == 0:
            continue
        cen_cur, dl_cur = xysT[i], dlsT[i]
        prsv_inds.append(order[i])
        dls_clpd.append(dl_cur[None])
        res_inds = i + 1 + torch.nonzero(confsT[i + 1:] > 0, as_tuple=True)[0]
        censT_res, dlsT_res = xysT[res_inds], dlsT[res_inds]
        dts_cen = censT_res - cen_cur
        dists_cen = torch.norm(dts_cen, dim=-1)
        dl_pos = dl_cur[xysT2iasT(dts_cen, num_div=dl_cur.size(-1))]
        dl_neg = torch.gather(dlsT_res, index=xysT2iasT(-dts_cen, num_div=dl_cur.size(-1))[..., None], dim=-1)[..., 0]
        fltr_nms = torch.maximum(dl_pos, dl_neg) > dists_cen * (1 + iou_thres)
        confsT[res_inds[fltr_nms]] = 0
        # 碰撞抑制
        fltr_hit = ~fltr_nms * (dl_pos + dl_neg > dists_cen)
        hit_inds = res_inds[fltr_hit]
        censT_hit, dlsT_hit = xysT[hit_inds], dlsT[hit_inds]
        xypsT_hit = xyT_dpT2xypT(censT_hit, dlsT_hit)
        dts_bdr = xypsT_hit - cen_cur
        dists_bdr = torch.norm(dts_bdr, dim=-1)
        dists_limt = dl_cur[xysT2iasT(dts_bdr, num_div=dl_cur.size(-1))]
        scaler = torch.maximum(dists_bdr, dists_limt) / dists_bdr
        xypsT_rpj = cen_cur + dts_bdr * scaler[..., None]
        dlsT_rpj = xyT_xypT2dpT(censT_hit, xypsT_rpj, num_div=dl_cur.size(-1))
        dlsT[hit_inds] = dlsT_rpj

    prsv_inds = torch.Tensor(prsv_inds).long()
    dls_clpd = torch.cat(dls_clpd, dim=0)
    return prsv_inds, dls_clpd

# </editor-fold>
