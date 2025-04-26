from PIL import ImageFont

from .axplt import _determind_corner
from .define import *
from .define import _determine_color, _item_text
from ..label import *

REGISTER_PILRND = Register()


def get_canvas(canvas: Optional[Image.Image] = None, size: tuple = (0, 0), color_bkgd=(0, 0, 0)) -> Image.Image:
    if canvas is None:
        canvas = Image.new(size=size, mode='RGB', color=color_bkgd)
    return canvas


# <editor-fold desc='基础元素绘制'>
@REGISTER_PILRND.registry(Image.Image)
def _pilrnd_imgP(imgP: Image.Image, canvas: Image.Image, alpha: float = 1.0, **kwargs):
    if alpha == 1.0:
        return imgP
    else:
        canvas = get_canvas(canvas, imgP.size, **kwargs)
        imgN_mixed = imgP2imgN(canvas) * (1 - alpha) + imgP2imgN(imgP) * alpha
        return imgN2imgP(imgN_mixed)


@REGISTER_PILRND.registry(np.ndarray)
def _pilrnd_imgN(imgN: np.ndarray, canvas: Image.Image, alpha: float = 1.0, **kwargs):
    return _pilrnd_imgP(imgN2imgP(imgN), canvas=canvas, alpha=alpha, **kwargs)


@REGISTER_PILRND.registry(torch.Tensor)
def _pilrnd_imgT(imgT: torch.Tensor, canvas: Image.Image, alpha: float = 1.0, **kwargs):
    return _pilrnd_imgP(imgT2imgP(imgT), canvas=canvas, alpha=alpha, **kwargs)


def _canvas_rgn(canvas: Image.Image) -> np.ndarray:
    return np.array([0, 0, canvas.size[0], canvas.size[1]])


def _pilrnd_text(xy, text: str, canvas: Image.Image, color: PIL_COLOR = (0, 0, 0),
                 text_color: PIL_COLOR = (255, 255, 255), font_size: int = 20):
    color = random_color(0, unit=False) if color is None else color
    draw = PIL.ImageDraw.Draw(canvas)
    font = ImageFont.truetype(PILRND_FONT_PTH, size=font_size)
    textsize = draw.textbbox(xy=(0, 0), text=text, font=font)[2:]
    draw.rectangle((xy[0], xy[1] - textsize[1], xy[0] + textsize[0], xy[1]), fill=color)
    draw.text((xy[0], xy[1] - textsize[1]), text, fill=text_color, font=font)
    return canvas


@REGISTER_PILRND.registry(XYSPoint)
def _pilrnd_pcd2d(pcd2d: XYSPoint, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                  text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                  **kwargs):
    canvas = get_canvas(canvas, size=pcd2d.size, **kwargs)
    draw = PIL.ImageDraw.Draw(canvas)
    xysN = pcd2d._xysN
    draw.point(tuple(xysN.reshape(-1)), fill=color)
    if text is not None:
        xy = _determind_corner(xysN, xyxyN_rgn=_canvas_rgn(canvas))
        _pilrnd_text(xy, text, canvas=canvas, color=color, text_color=text_color, font_size=font_size)
    return canvas


@REGISTER_PILRND.registry(XYSGraph)
def _pilrnd_vgraph2d(vgraph2d: XYSGraph, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                     text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                     line_width: float = 2, **kwargs):
    canvas = get_canvas(canvas, size=vgraph2d.size, **kwargs)
    draw = PIL.ImageDraw.Draw(canvas)
    xysN = vgraph2d._xysN
    draw.point(tuple(xysN.reshape(-1)), fill=color)
    lines = xysN[vgraph2d._edgesN]
    for line in lines:
        draw.line(list(line.reshape(-1)), fill=color, width=line_width)
    if text is not None:
        xy = _determind_corner(xysN, xyxyN_rgn=_canvas_rgn(canvas))
        _pilrnd_text(xy, text, canvas=canvas, color=color, text_color=text_color, font_size=font_size)
    return canvas


@REGISTER_PILRND.registry(XYSSurface)
def _pilrnd_surf2d(surf2d: XYSSurface, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                   text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                   line_width: float = 2, **kwargs):
    vgraph2d = XYSGraph.convert(surf2d)
    return _pilrnd_vgraph2d(vgraph2d, canvas, color=color, text=text, text_color=text_color, font_size=font_size,
                            line_width=line_width, **kwargs)


@REGISTER_PILRND.registry(XYPBorder)
def _pilrnd_xypbdr(xypbdr: XYPBorder, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                   text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                   line_width: float = 2, **kwargs):
    canvas = get_canvas(canvas, size=xypbdr.size, **kwargs)
    draw = PIL.ImageDraw.Draw(canvas)
    xyp = xypbdr._xypN
    xyp = np.concatenate([xyp, xyp[:1]], axis=0)
    draw.line(list(xyp.reshape(-1)), fill=color, width=line_width)
    if text is not None:
        xy = _determind_corner(xyp, xyxyN_rgn=_canvas_rgn(canvas))
        _pilrnd_text(xy, text, canvas=canvas, color=color, text_color=text_color, font_size=font_size)
    return canvas


@REGISTER_PILRND.registry(XYXYBorder)
def _pilrnd_xyxybdr(xyxybdr: XYXYBorder, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                    text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                    line_width: float = 2, **kwargs):
    return _pilrnd_xypbdr(XYPBorder.convert(xyxybdr), canvas, color=color, text=text, text_color=text_color,
                          font_size=font_size, line_width=line_width, **kwargs)


@REGISTER_PILRND.registry(XYWHBorder)
def _pilrnd_xywhbdr(xywhbdr: XYWHBorder, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                    text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                    line_width: float = 2, **kwargs):
    return _pilrnd_xypbdr(XYPBorder.convert(xywhbdr), canvas, color=color, text=text, text_color=text_color,
                          font_size=font_size, line_width=line_width, **kwargs)


@REGISTER_PILRND.registry(XYWHABorder)
def _pilrnd_xywhabdr(xywhabdr: XYWHABorder, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                     text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                     line_width: float = 2, **kwargs):
    return _pilrnd_xypbdr(XYPBorder.convert(xywhabdr), canvas, color=color, text=text, text_color=text_color,
                          font_size=font_size, line_width=line_width, **kwargs)


@REGISTER_PILRND.registry(AbsBoolRegion)
def _pilrnd_absboolrgn(absboolrgn: AbsBoolRegion, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                       alpha: float = 0.7, line_width: int = 0, rgn_bdr_color: PIL_COLOR = (255, 255, 255),
                       text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0), font_size: int = 20,
                       **kwargs):
    canvas = get_canvas(canvas, size=absboolrgn.size, **kwargs)

    maskN = absboolrgn.maskNb.astype(np.float32) * 255
    mask = imgN2imgP(maskN * alpha)
    mask_color = Image.new(mode='RGB', size=mask.size, color=color)
    canvas = Image.composite(mask_color, canvas, mask)

    if line_width > 0:
        kernel = np.ones(shape=(line_width * 2 + 1, line_width * 2 + 1))
        maskN_expd = cv2.dilate(maskN, kernel)

        mask_bdr = imgN2imgP((maskN_expd - maskN) * alpha)
        mask_bdr_color = Image.new(mode='RGB', size=mask.size, color=rgn_bdr_color)
        canvas = Image.composite(mask_bdr_color, canvas, mask_bdr)

    if text is not None:
        xs, ys = np.where(maskN)
        _pilrnd_text((xs[0], ys[0]), text, canvas=canvas, color=color, text_color=text_color, font_size=font_size)
    return canvas


@REGISTER_PILRND.registry(RefValRegion)
def _pilrnd_refvalrgn(refvalrgn: RefValRegion, canvas: Optional[Image.Image] = None, color: PIL_COLOR = (255, 0, 0),
                      alpha: float = 0.9, text: Optional[str] = None, text_color: Optional[PIL_COLOR] = (0, 0, 0),
                      font_size: int = 20, **kwargs):
    canvas = get_canvas(canvas, size=refvalrgn.size, **kwargs).convert('RGBA')
    mask_ref = refvalrgn.maskNb_ref * 255 * alpha
    mask_color = np.broadcast_to(np.array(color), (mask_ref.shape[0], mask_ref.shape[1], 3))
    mask_raw = np.concatenate([mask_color, mask_ref[..., None]], axis=-1).astype(np.uint8)
    maskP_merge = Image.fromarray(mask_raw, mode='RGBA')
    dest = np.clip(refvalrgn.xyxyN[:2], a_min=0, a_max=None).astype(int)
    source = (dest - refvalrgn.xyxyN[:2]).astype(int)
    # print(maskP_merge.size, (int(refvalrgn.xyxyN[0]), int(refvalrgn.xyxyN[1])))
    # canvas.alpha_composite(maskP_merge, (int(refvalrgn.xyxyN[0]), int(refvalrgn.xyxyN[1])))
    canvas.alpha_composite(maskP_merge, dest=tuple(dest), source=tuple(source))
    if text is not None:
        xy = _determind_corner(xyxyN2xypN(refvalrgn.xyxyN), xyxyN_rgn=_canvas_rgn(canvas))
        _pilrnd_text(xy, text, canvas=canvas, color=color, text_color=text_color, font_size=font_size)

    return canvas


@REGISTER_PILRND.registry(BoxItem)
def _pilrnd_boxitem(boxitem: BoxItem, canvas: Optional[Image.Image] = None, color: Optional[PIL_COLOR] = None,
                    text_color: Optional[PIL_COLOR] = (255, 255, 255), font_size: int = 20, with_text: bool = True,
                    line_width: float = 2, **kwargs):
    color = _determine_color(color, index=boxitem.category.cindN, unit=False)
    text = _item_text(boxitem.category, name=boxitem.get('name', None)) if with_text else None

    canvas = _pilrnd_xypbdr(XYPBorder.convert(boxitem.border), canvas=canvas, color=color,
                            text=text, text_color=text_color, font_size=font_size, line_width=int(line_width),
                            **kwargs)
    return canvas


@REGISTER_PILRND.registry(SegItem)
def _pilrnd_segitem(segitem: SegItem, canvas: Optional[Image.Image] = None, color: Optional[PIL_COLOR] = None,
                    alpha: float = 0.3, text_color: Optional[PIL_COLOR] = (255, 255, 255), font_size: int = 20,
                    with_text: bool = True, **kwargs):
    color = _determine_color(color, index=segitem.category.cindN, unit=False)
    text = _item_text(segitem.category, name=segitem.get('name', None)) if with_text else None

    canvas = _pilrnd_absboolrgn(AbsBoolRegion.convert(segitem.rgn), canvas=canvas, color=color,
                                text=text, text_color=text_color, font_size=font_size, alpha=alpha, **kwargs)
    return canvas


@REGISTER_PILRND.registry(InstItem)
def _pilrnd_institem(institem: InstItem, canvas: Optional[Image.Image] = None, color: Optional[PIL_COLOR] = None,
                     alpha: float = 0.3, text_color: Optional[PIL_COLOR] = (255, 255, 255), font_size: int = 20,
                     with_text: bool = True, line_width: float = 2, **kwargs):
    color = _determine_color(color, index=institem.category.cindN, unit=False)
    text = _item_text(institem.category, name=institem.get('name', None)) if with_text else None

    canvas = _pilrnd_refvalrgn(RefValRegion.convert(institem.rgn), canvas=canvas, color=color,
                               text=text, text_color=text_color, font_size=font_size, alpha=alpha, **kwargs)

    canvas = _pilrnd_xypbdr(XYPBorder.convert(institem.border), canvas=canvas, color=color,
                            text=text, text_color=text_color, font_size=font_size, line_width=line_width, **kwargs)

    return canvas


@REGISTER_PILRND.registry(ImageLabel)
def _pilrnd_imglabel(imglabel: ImageLabel, canvas: Optional[Image.Image] = None, with_ctx: bool = True, **kwargs):
    if with_ctx:
        canvas = _pilrnd_xypbdr(imglabel.ctx_border, canvas=canvas, )
    return canvas


@REGISTER_PILRND.registry(ImageItemsLabel, PointsLabel, BoxesLabel, SegsLabel, InstsLabel)
def _pilrnd_imgitemslabel(imgitemslabel: ImageItemsLabel, canvas: Optional[Image.Image] = None, with_ctx: bool = True,
                          **kwargs):
    canvas = _pilrnd_imglabel(imgitemslabel, canvas=canvas, with_ctx=with_ctx)
    for item in imgitemslabel:
        canvas = _pilrnd_item(item, canvas=canvas, **kwargs)
    return canvas


@REGISTER_PILRND.registry(Sequence, list, tuple)
def _pilrnd_seq(seq: Sequence, canvas: Optional[Image.Image] = None, **kwargs):
    for item in seq:
        if item is not None:
            canvas = _pilrnd_item(item, canvas=canvas, **kwargs)
    return canvas


def _pilrnd_item(item, canvas: Optional[Image.Image] = None, **kwargs):
    rndor = REGISTER_PILRND[item.__class__]
    img = rndor(item, canvas=canvas, **kwargs)
    return img


def _pilrnd_items(*items, canvas: Optional[Image.Image] = None, **kwargs):
    for item in items:
        if item is not None:
            canvas = _pilrnd_item(item, canvas=canvas, **kwargs)
    return canvas
