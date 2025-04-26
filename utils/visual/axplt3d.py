from .axplt import *
from .axplt import _axplt_item, _axplt_imglabel
from .define import _item_text, _determine_color


# <editor-fold desc='坐标轴设置'>
def get_axis3d(axis: Optional[PLT_AXIS3D] = None, tick: bool = True, figsize: tuple = (6, 6), **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(projection='3d')
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, )
        fig.show()
    if not tick:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
    return axis


def _axplt3d_item(item, axis: PLT_AXIS3D, **kwargs):
    pltor = REGISTER_AXPLT3D[item.__class__]
    pltor(item, axis=axis, **kwargs)
    return axis


def post_axis(axis: PLT_AXIS3D):
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    zlim = axis.get_zlim()
    axis.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    return axis


def init_axis():
    return None


# </editor-fold>

# <editor-fold desc='画图片'>
@REGISTER_AXPLT3D.registry(np.ndarray)
def _axplt3d_imgN(img: np.ndarray, axis: PLT_AXIS3D, **kwargs):
    size = img2size(img)
    assert img.shape[-1] == 1
    img = np.reshape(img, newshape=(size[1], size[0]))
    xs = np.arange(size[0]) + 0.5
    ys = np.arange(size[1]) + 0.5
    xs, ys = np.meshgrid(xs, ys)
    img = (img - np.mean(img)) / np.std(img) * (np.mean(size) / 40)
    axis.plot_surface(xs, ys, img, cmap='viridis')
    return axis


@REGISTER_AXPLT3D.registry(Image.Image)
def _axplt3d_imgP(imgP: Image.Image, axis: PLT_AXIS, **kwargs):
    return _axplt3d_imgN(imgP2imgN(imgP), axis=axis)


@REGISTER_AXPLT3D.registry(torch.Tensor)
def _axplt3d_imgT(imgT: torch.Tensor, axis: PLT_AXIS, **kwargs):
    return _axplt3d_imgN(imgT2imgN(imgT), axis=axis)


# </editor-fold>

# <editor-fold desc='画物体'>
@REGISTER_AXPLT.registry(MCamera)
def _axplt_camera(camera: MCamera, axis: PLT_AXIS, **kwargs):
    init_axis(axis, img_size=camera.size, **kwargs)
    return axis


@REGISTER_AXPLT3D.registry(MCamera)
def _axplt3d_camera(camera: MCamera, axis: PLT_AXIS3D, zdist: float = 10, color: PLT_COLOR = 'k',
                    linewidth: float = 1.0, **kwargs):
    xyxy_rgn = np.array([0, 0, camera.size[0], camera.size[1]])
    xys_rgn = xyxyN2xypN(xyxy_rgn)
    xyzs_rgn = np.concatenate([xys_rgn, np.ones(shape=(xys_rgn.shape[0], 1))], axis=1) * zdist
    xyzs_rgn = xyzs_rgn @ np.linalg.inv(camera._intrinsicN).T
    for i in range(4):
        x, y, z = xyzs_rgn[i]
        x_n, y_n, z_n = xyzs_rgn[(i + 1) % 4]
        axis.plot3D([0, x], [0, y], [0, z], color=color, linewidth=linewidth)
        axis.plot3D([x, x_n], [y, y_n], [z, z_n], color=color, linewidth=linewidth)
    return axis


@REGISTER_AXPLT.registry(XYZSColoredSurface)
def _axplt_csurf(csurf: XYZSColoredSurface, axis: PLT_AXIS, camera: MCamera,
                 **kwargs):
    _axplt_item(csurf.project(camera), axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT3D.registry(XYZSColoredSurface)
def _axplt3d_csurf(csurf: XYZSColoredSurface, axis: PLT_AXIS3D, color: PLT_COLOR = 'lightgray',
                   **kwargs):
    xyzsN = csurf.xyzsN
    vcolors = csurf.vcolorsN
    if vcolors is not None:
        axis.scatter(xyzsN[:, 0], xyzsN[:, 1], xyzsN[:, 2], c=vcolors / 255)
    else:
        axis.plot_trisurf(xyzsN[:, 0], xyzsN[:, 1], xyzsN[:, 2], triangles=csurf.surfsN, color=color)
    return axis


@REGISTER_AXPLT.registry(XYZSSurface)
def _axplt_surf(surf: XYZSSurface, axis: PLT_AXIS, camera: MCamera,
                **kwargs):
    _axplt_item(surf.project(camera), axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT3D.registry(XYZSSurface)
def _axplt3d_surf(surf: XYZSSurface, axis: PLT_AXIS3D, color: PLT_COLOR = 'lightgray', **kwargs):
    xyzsN = surf.xyzsN
    axis.plot_trisurf(xyzsN[:, 0], xyzsN[:, 1], xyzsN[:, 2], triangles=surf.surfsN, color=color)
    return axis


@REGISTER_AXPLT.registry(XYZSPoint)
def _axplt_pcd3d(pcd3d: XYZSPoint, axis: PLT_AXIS, camera: MCamera,
                 **kwargs):
    _axplt_item(pcd3d.project(camera), axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT3D.registry(XYZSPoint)
def _axplt3d_pcd3d(pcd3d: XYZSPoint, axis: PLT_AXIS3D, color: PLT_COLOR = 'r',
                   markersize: float = 2.0, text: Optional[str] = None, alpha: float = 0.7,
                   text_color: Optional[PLT_COLOR] = 'w', **kwargs):
    xyzsN = pcd3d.xyzsN
    axis.plot3D(xyzsN[:, 0], xyzsN[:, 1], xyzsN[:, 2], '.', markersize=markersize, color=color)
    if text is not None:
        x, y, z = xyzsN[0]
        axis.text(x, y, z, text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYZSGraph)
def _axplt_graph(graph: XYZSGraph, axis: PLT_AXIS, camera: MCamera,
                 **kwargs):
    _axplt_item(graph.project(camera), axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT3D.registry(XYZSGraph)
def _axplt3d_graph(graph: XYZSGraph, axis: PLT_AXIS3D, color: PLT_COLOR = 'r',
                   linewidth: float = 1.0, markersize: float = 2.0, text: Optional[str] = None, alpha: float = 0.7,
                   text_color: Optional[PLT_COLOR] = 'w', **kwargs):
    xyzsN, ipsN = graph.xyzsN, graph.edgesN
    axis.plot3D(xyzsN[:, 0], xyzsN[:, 1], xyzsN[:, 2], '.', markersize=markersize, color=color)
    lines = xyzsN[ipsN]
    for line in lines:
        axis.plot3D(line[:, 0], line[:, 1], line[:, 2], '-', linewidth=linewidth, color=color)
    if text is not None:
        x, y, z = xyzsN[0]
        axis.text(x, y, z, text, color=text_color, fontdict=FONT_DICT_SMALL,
                  bbox=dict(facecolor=color, alpha=alpha, edgecolor=None, lw=0, boxstyle='square,pad=0.05'), )
    return axis


@REGISTER_AXPLT.registry(XYZWHLQBorder)
def _axplt_sbox(sbox: XYZWHLQBorder, axis: PLT_AXIS, camera: MCamera,
                **kwargs):
    _axplt_item(sbox.project(camera), axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT3D.registry(XYZWHLQBorder)
def _axplt3d_sbox(sbox: XYZWHLQBorder, axis: PLT_AXIS3D, **kwargs):
    return _axplt3d_graph(XYZSGraph.convert(sbox), axis=axis, **kwargs)


@REGISTER_AXPLT3D.registry(StereoBoxItem)
def _axplt3d_sboxitem(sboxitem: StereoBoxItem, axis: PLT_AXIS3D, color: Optional[PLT_COLOR] = None,
                      linewidth: float = 1.0, markersize: float = 2.0, with_text: bool = True, alpha: float = 0.7,
                      text_color: Optional[PLT_COLOR] = 'w', **kwargs):
    text = _item_text(sboxitem.category, name=sboxitem.get('name', None)) if with_text else None
    color = _determine_color(color, index=sboxitem.category.cindN, unit=True)
    _axplt3d_item(sboxitem.border, axis=axis, linewidth=linewidth, markersize=markersize,
                  color=color, text=text, text_color=text_color, alpha=alpha)
    return axis


@REGISTER_AXPLT3D.registry(ImageLabel)
def _axplt3d_imglabel(imglabel: ImageLabel, axis: PLT_AXIS3D, color: PLT_COLOR = 'k',
                      title_append: str = '', show_meta: bool = False, index: Optional[int] = None, **kwargs):
    msg = '[%d]' % index if index is not None else ''
    if show_meta and imglabel.meta is not None:
        msg += ' ' + str(imglabel.meta)
    msg += title_append
    axis.set_title(label=msg, pad=4, color=color, fontdict=FONT_DICT_XLARGE)
    return axis


@REGISTER_AXPLT3D.registry(StereoItemsLabel)
def _axplt3d_sitemslabel(sitemslabel: StereoItemsLabel, axis: PLT_AXIS3D, show_meta: bool = False,
                         index: Optional[int] = None, **kwargs):
    _axplt3d_imglabel(sitemslabel, axis=axis, show_meta=show_meta, index=index)
    _axplt3d_camera(sitemslabel.camera, axis=axis, **kwargs)
    for item in sitemslabel:
        _axplt3d_item(item=item, axis=axis, **kwargs)
    return axis


@REGISTER_AXPLT.registry(StereoItemsLabel)
def _axplt_sitemslabel(sitemslabel: StereoItemsLabel, axis: PLT_AXIS, show_meta: bool = False,
                       index: Optional[int] = None, ctx_color: PLT_COLOR = 'k', with_ctx: bool = True, **kwargs):
    _axplt_imglabel(sitemslabel, axis=axis, show_meta=show_meta, index=index, ctx_color=ctx_color, with_ctx=with_ctx)
    camera = sitemslabel.camera
    for item in sitemslabel:
        if isinstance(item, Projectable):
            _axplt_item(item=item.project(camera), axis=axis, **kwargs)
        else:
            _axplt_item(item=item, axis=axis, **kwargs)
    return axis


# </editor-fold>

@REGISTER_AXPLT3D.registry(Sequence)
def _axplt3d_seq(seq: Sequence, axis: PLT_AXIS3D, **kwargs):
    for item in seq:
        if item is not None:
            _axplt3d_item(item=item, axis=axis, **kwargs)
    return axis


def _axplt3d_items(*items, axis: PLT_AXIS3D = None, tick: bool = True, **kwargs):
    axis = get_axis3d(axis=axis, tick=tick)
    for item in items:
        if item is not None:
            _axplt3d_item(item=item, axis=axis, **kwargs)
    post_axis(axis)
    return axis
