from matplotlib.collections import PolyCollection

from .define import *
from .define import _determine_color, _item_text, _ensure_rgb
from ..typings import ps_int_multiply
from ..label import *


# <editor-fold desc='坐标轴设置'>
def get_axis(axis: Optional[PLT_AXIS] = None, tick: bool = True, figsize: tuple = (6, 6), **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot()
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, )
        fig.show()
    if not tick:
        axis.set_xticks([])
        axis.set_yticks([])
    if not axis.yaxis_inverted():
        axis.invert_yaxis()
    axis.xaxis.set_ticks_position('top')
    axis.set_aspect('equal', 'box')
    return axis


def init_axis(axis: PLT_AXIS, img_size: tuple, extend: float = 0, ):
    dtw = ps_int_multiply(extend, reference=img_size[0])
    dth = ps_int_multiply(extend, reference=img_size[1])
    axis.set_xlim(-dtw, img_size[0] + dtw)
    axis.set_ylim(-dth, img_size[1] + dth)
    if not axis.yaxis_inverted():
        axis.invert_yaxis()
    return axis


def post_axis():
    return None


# </editor-fold>


def _axis_rgn(axis: PLT_AXIS) -> np.ndarray:
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    return np.array([min(xlim), min(ylim), max(xlim), max(ylim)])


def _determind_corner(xysN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    xysN = xysN_clip(xysN, xyxyN_rgn)
    if len(xysN) == 0:
        return np.array([0, 0])
    idx = np.argmin(xysN[:, 1] + xysN[:, 0] * 0.2)
    return xysN[idx]


# <editor-fold desc='标签展示'>

@REGISTER_AXPLT.registry(XYSSurface)
def _axplt_vsurf2d(vsurf2d: XYSSurface, axis: PLT_AXIS, color: PLT_COLOR = 'w',
                   edge_color: PLT_COLOR = 'k', linewidth: float = 1.0,
                   text: Optional[str] = None, text_color: Optional[PLT_COLOR] = 'w',
                   alpha: float = 1.0, **kwargs):
    init_axis(axis, img_size=vsurf2d.size)
    xysN, ifsN = vsurf2d._xysN, vsurf2d._surfsN
    collection = PolyCollection(xysN[ifsN], closed=True, alpha=alpha, linewidth=linewidth, facecolor=color,
                                edgecolor=edge_color)
    axis.add_collection(collection)
    if text is not None:
        xy = _determind_corner(xysN, xyxyN_rgn=_axis_rgn(axis))
        axis.text(xy[0], xy[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYSColoredSurface)
def _axplt_cvsurf2d(cvsurf2d: XYSColoredSurface, axis: PLT_AXIS, color: PLT_COLOR = 'w',
                    edge_color: PLT_COLOR = 'k', linewidth: float = 0.0,
                    text: Optional[str] = None, text_color: Optional[PLT_COLOR] = 'w',
                    alpha: float = 1.0, **kwargs):
    init_axis(axis, img_size=cvsurf2d.size)
    xysN, surfsN = cvsurf2d._xysN, cvsurf2d._surfsN
    facecolors = np.mean(cvsurf2d.vcolorsN[surfsN], axis=1) / 255
    facecolors = np.clip(facecolors, a_min=0.1, a_max=0.9)

    collection = PolyCollection(xysN[surfsN], closed=True, linewidth=linewidth, facecolors=facecolors)
    axis.add_collection(collection)
    if text is not None:
        xy = _determind_corner(xysN, xyxyN_rgn=_axis_rgn(axis))
        axis.text(xy[0], xy[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYSPoint)
def _axplt_pcd2d(pcd2d: XYSPoint, axis: PLT_AXIS, color: PLT_COLOR = 'r', markersize: float = 2,
                 text: Optional[str] = None, text_color: Optional[PLT_COLOR] = 'w',
                 alpha: float = 1.0, **kwargs):
    init_axis(axis, img_size=pcd2d.size)
    xysN = pcd2d._xysN
    axis.scatter(xysN[:, 0], xysN[:, 1], color=color, linewidths=markersize)
    if text is not None:
        xy = _determind_corner(xysN, xyxyN_rgn=_axis_rgn(axis))
        axis.text(xy[0], xy[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYSGraph)
def _axplt_vgraph2d(vgraph: XYSGraph, axis: PLT_AXIS, color: PLT_COLOR = 'r', linewidth: float = 1.0,
                    markersize: float = 2.0, text: Optional[str] = None, text_color: Optional[PLT_COLOR] = 'w',
                    alpha: float = 1.0, **kwargs):
    init_axis(axis, img_size=vgraph.size)
    xysN, ipsN = vgraph._xysN, vgraph._edgesN
    axis.plot(xysN[:, 0], xysN[:, 1], '.', markersize=markersize, color=color)
    lines = xysN[ipsN]
    for line in lines:
        axis.plot(line[:, 0], line[:, 1], '-', linewidth=linewidth, color=color)
    if text is not None:
        xy = _determind_corner(xysN, xyxyN_rgn=_axis_rgn(axis))
        axis.text(xy[0], xy[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYPBorder)
def _axplt_xypbdr(xypbdr: XYPBorder, axis: PLT_AXIS, color: PLT_COLOR = 'r', linewidth: float = 1.0,
                  text: Optional[str] = None, text_color: Optional[PLT_COLOR] = 'w',
                  alpha: float = 1.0, linestyle: str = '-', **kwargs):
    init_axis(axis, img_size=xypbdr.size)
    xyp = np.concatenate([xypbdr._xypN, xypbdr._xypN[:1]], axis=0)
    axis.plot(xyp[:, 0], xyp[:, 1], '-', linewidth=linewidth, color=color, linestyle=linestyle)
    if text is not None:
        xy = _determind_corner(xyp, xyxyN_rgn=_axis_rgn(axis))
        axis.text(xy[0], xy[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYXYBorder)
def _axplt_xyxybdr(xyxybdr: XYXYBorder, axis: PLT_AXIS, **kwargs):
    return _axplt_xypbdr(XYPBorder.convert(xyxybdr), axis=axis, **kwargs)


@REGISTER_AXPLT.registry(XYWHBorder)
def _axplt_xywhbdr(xywhbdr: XYWHBorder, axis: PLT_AXIS, **kwargs):
    return _axplt_xypbdr(XYPBorder.convert(xywhbdr), axis=axis, **kwargs)


@REGISTER_AXPLT.registry(XYWHABorder)
def _axplt_xywhabdr(xywhabdr: XYWHABorder, axis: PLT_AXIS, **kwargs):
    return _axplt_xypbdr(XYPBorder.convert(xywhabdr), axis=axis, **kwargs)


@REGISTER_AXPLT.registry(AbsBoolRegion, AbsValRegion, NailValRegion)
def _axplt_absboolrgn(absboolrgn: AbsBoolRegion, axis: PLT_AXIS, color: PLT_COLOR = 'r',
                      alpha: float = 0.7, text: Optional[str] = None,
                      text_color: Optional[PLT_COLOR] = 'w', **kwargs):
    init_axis(axis, img_size=absboolrgn.size)
    maskN = absboolrgn.maskNb.astype(np.float32)
    color = _ensure_rgb(color, unit=True)
    mask_coled = np.broadcast_to(color, (maskN.shape[0], maskN.shape[1], 3))
    maskN_merge = np.concatenate([mask_coled, maskN[..., None]], axis=2)
    axis.imshow(maskN_merge, alpha=alpha, extent=(0, maskN_merge.shape[1], maskN_merge.shape[0], 0))
    if text is not None:
        ys, xs = np.where(maskN)
        axis.text(xs[0], ys[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )

    return axis


@REGISTER_AXPLT.registry(RefValRegion)
def _axplt_refvalrgn(refvalrgn: RefValRegion, axis: PLT_AXIS, color: PLT_COLOR = 'r', alpha: float = 0.7,
                     text: Optional[str] = None,
                     text_color: Optional[PLT_COLOR] = 'w', **kwargs):
    init_axis(axis, img_size=refvalrgn.size)

    xyxy_rgn = refvalrgn.xyxyN
    maskN_ref = refvalrgn.maskNb_ref[..., None].astype(np.float32)
    color = _ensure_rgb(color, unit=True)
    if np.prod(maskN_ref.shape[:2]) > 0:
        mask_color = np.broadcast_to(color, (maskN_ref.shape[0], maskN_ref.shape[1], 3))
        maskN_merge = np.concatenate([mask_color, maskN_ref], axis=2)
        axis.imshow(maskN_merge, alpha=alpha, extent=(xyxy_rgn[0], xyxy_rgn[2], xyxy_rgn[3], xyxy_rgn[1]))
    if text is not None:
        axis.text(xyxy_rgn[0], xyxy_rgn[1], text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(StereoBoxItem)
def _axplt_sboxitem(sboxitem: StereoBoxItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                    text_color: Optional[PLT_COLOR] = 'w', linewidth: float = 1.0,
                    markersize: float = 2.0, alpha: float = 1.0, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=sboxitem.category.cindN, unit=True)
    text = _item_text(sboxitem.category, name=sboxitem.get('name', None)) if with_text else None

    _axplt_vgraph2d(sboxitem.border, axis=axis, color=color, markersize=markersize,
                    linewidth=linewidth, text=text, alpha=alpha, text_color=text_color, **kwargs)
    return axis


@REGISTER_AXPLT.registry(StereoMixItem)
def _axplt_sboxmitem(sboxmitem: StereoMixItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                     text_color: Optional[PLT_COLOR] = 'w', linewidth: float = 1.0,
                     markersize: float = 2.0, alpha: float = 1.0, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=sboxmitem.category.cindN, unit=True)
    text = _item_text(sboxmitem.category, name=sboxmitem.get('name', None)) if with_text else None

    _axplt_vgraph2d(sboxmitem.border, axis=axis, color=color, markersize=markersize,
                    linewidth=linewidth, text=text, alpha=alpha, text_color=text_color, **kwargs)
    _axplt_pcd2d(sboxmitem.border_mv, axis=axis, color=color, markersize=markersize, )
    return axis


@REGISTER_AXPLT.registry(PointItem)
def _axplt_pntitem(pntitem: PointItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                   text_color: Optional[PLT_COLOR] = 'w', linewidth: float = 1.0,
                   markersize: float = 2.0, alpha: float = 1.0, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=pntitem.category.cindN, unit=True)
    text = _item_text(pntitem.category, name=pntitem.get('name', None)) if with_text else None

    _axplt_pcd2d(pntitem.pnts, axis=axis, color=color, markersize=markersize,
                 linewidth=linewidth, text=text, alpha=alpha, text_color=text_color, **kwargs)
    return axis


@REGISTER_AXPLT.registry(BoxItem)
def _axplt_boxitem(boxitem: BoxItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                   text_color: Optional[PLT_COLOR] = 'w', linewidth: float = 1.0,
                   alpha: float = 0.5, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=boxitem.category.cindN, unit=True)
    text = _item_text(boxitem.category, name=boxitem.get('name', None)) if with_text else None

    _axplt_item(boxitem.border, axis=axis, text=text,
                color=color, text_color=text_color, linewidth=linewidth, alpha=alpha, **kwargs)
    return axis


@REGISTER_AXPLT.registry(DualBoxItem)
def axplt_dualboxitem(dualboxitem: DualBoxItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                      text_color: Optional[PLT_COLOR] = 'w', linewidth: float = 1.0, linestyle: str = '-',
                      alpha: float = 0.5, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=dualboxitem.category.cindN, unit=True)
    text = _item_text(dualboxitem.category, name=dualboxitem.get('name', None)) if with_text else None
    _axplt_item(dualboxitem.border, axis=axis, text=text,
                color=color, text_color=text_color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha, **kwargs)
    _axplt_item(dualboxitem.border2, axis=axis,
                color=color, linestyle=linestyle, **kwargs)
    return axis


@REGISTER_AXPLT.registry(SegItem)
def _axplt_segitem(segitem: SegItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                   text_color: Optional[PLT_COLOR] = 'w', alpha: float = 0.5, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=segitem.category.cindN, unit=True)
    text = _item_text(segitem.category, name=segitem.get('name', None)) if with_text else None

    _axplt_absboolrgn(AbsBoolRegion.convert(segitem.rgn), axis=axis, text=text, text_color=text_color,
                      color=color, alpha=alpha, **kwargs)

    return axis


@REGISTER_AXPLT.registry(InstItem)
def _axplt_institem(institem: InstItem, axis: PLT_AXIS, color: Optional[PLT_COLOR] = None,
                    text_color: Optional[PLT_COLOR] = 'w', linewidth: float = 1.0, linestyle: str = '-',
                    alpha: float = 0.5, with_text: bool = True, **kwargs):
    color = _determine_color(color, index=institem.category.cindN, unit=True)
    text = _item_text(institem.category, name=institem.get('name', None)) if with_text else None

    _axplt_xypbdr(xypbdr=XYPBorder.convert(institem.border), axis=axis, text=text,
                  color=color, text_color=text_color, linewidth=linewidth,
                  linestyle=linestyle, alpha=alpha, **kwargs)
    _axplt_refvalrgn(RefValRegion.convert(institem.rgn), axis=axis, color=color, alpha=alpha, **kwargs)
    return axis


@REGISTER_AXPLT.registry(np.ndarray)
def _axplt_imgN(imgN: np.ndarray, axis: PLT_AXIS, **kwargs):
    assert len(imgN.shape) in (2, 3), 'size err'
    extent = (0, imgN.shape[1], imgN.shape[0], 0)
    axis.imshow(imgN / 255, extent=extent)
    return axis


@REGISTER_AXPLT.registry(Image.Image)
def _axplt_imgP(imgP: Image.Image, axis: PLT_AXIS, **kwargs):
    return _axplt_imgN(imgP2imgN(imgP), axis=axis)


@REGISTER_AXPLT.registry(torch.Tensor)
def _axplt_imgT(imgT: torch.Tensor, axis: PLT_AXIS, **kwargs):
    return _axplt_imgN(imgT2imgN(imgT), axis=axis)


@REGISTER_AXPLT.registry(ImageLabel)
def _axplt_imglabel(imglabel: ImageLabel, axis: PLT_AXIS, color: PLT_COLOR = 'k',
                    ctx_color: PLT_COLOR = 'k', with_ctx: bool = True, title_append: str = '',
                    show_meta: bool = True, index: Optional[int] = None, **kwargs):
    axis = init_axis(axis, img_size=imglabel.img_size)
    msg = '[%d]' % index if index is not None else ''
    if show_meta and imglabel.meta is not None:
        msg += ' ' + str(imglabel.meta)
    msg += title_append
    axis.set_title(label=msg, pad=4, color=color, fontdict=FONT_DICT_XLARGE)
    if with_ctx:
        _axplt_xypbdr(imglabel.ctx_border, axis, color=ctx_color, linewidth=2)
    return axis


@REGISTER_AXPLT.registry(CategoryLabel)
def _axplt_catelabel(catelabel: CategoryLabel, axis: PLT_AXIS,
                     show_meta: bool = True, index: Optional[int] = None, with_ctx: bool = True, **kwargs):
    title_append = ' ' + _item_text(catelabel.category, name=catelabel.get('name', None))
    _axplt_imglabel(catelabel, axis=axis, color='k', with_ctx=with_ctx, ctx_color='k',
                    show_meta=show_meta, index=index, title_append=title_append, **kwargs)
    return axis


@REGISTER_AXPLT.registry(PointsLabel, BoxesLabel, SegsLabel, InstsLabel, ImageItemsLabel)
def _axplt_imgitemslabel(imgitemslabel: ImageItemsLabel, axis: PLT_AXIS,
                         show_meta: bool = True, index: Optional[int] = None, with_ctx: bool = True, **kwargs):
    _axplt_imglabel(imgitemslabel, axis=axis, color='k', with_ctx=with_ctx, ctx_color='k',
                    show_meta=show_meta, index=index, **kwargs)
    for item in imgitemslabel:
        _axplt_item(item=item, axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT.registry(Sequence, list, tuple)
def _axplt_seq(seq: Sequence, axis, **kwargs):
    for item in seq:
        if item is not None:
            _axplt_item(item=item, axis=axis, **kwargs)
    return axis


def _axplt_item(item, axis: PLT_AXIS = None, **kwargs):
    pltor = REGISTER_AXPLT[item.__class__]
    pltor(item, axis=axis, **kwargs)
    return axis


def _axplt_items(*items, axis: PLT_AXIS = None, tick: bool = True, **kwargs):
    axis = get_axis(axis=axis, tick=tick)
    for item in items:
        if item is not None:
            _axplt_item(item=item, axis=axis, **kwargs)
    return axis

# </editor-fold>
