try:
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer import SoftPhongShader, TensorProperties, BlendParams, hard_rgb_blend, Materials
    from pytorch3d.renderer.mesh.rasterizer import Fragments
    from pytorch3d.renderer.mesh.shading import _apply_lighting
except Exception as e:
    Fragments = None
    TensorProperties = None
    pass

from .imglabel import *
from .imgitem3d import *


class StereoItemsLabel(ImageLabel, HasFiltList, HasPermutationList, Clipable, HasFiltMeasureList, HasCalibration,
                       HasXYXYSN, CategoryExportable):
    def __getitem__(self, item):
        if isinstance(item, Iterable):
            items = self.empty()
            for ind in item:
                items.append(self[ind])
            return items
        elif isinstance(item, slice):
            items = self.empty()
            items += list.__getitem__(self, item)
            return items
        else:
            return list.__getitem__(self, item)

    @property
    def ctx_size(self):
        return self.ctx_border.size

    @ctx_size.setter
    def ctx_size(self, ctx_size):
        self.ctx_border = XYPBorder(xyxyN2xypN(np.array([0, 0, ctx_size[0], ctx_size[1]])), size=ctx_size)
        self.camera.size = ctx_size

    def __init__(self, *items, camera: MCamera, meta=None, **kwargs):
        list.__init__(self, items)
        self.camera = camera
        ImageLabel.__init__(self, img_size=camera.size, meta=meta, **kwargs)

    def refrom_xysN(self, xysN: np.ndarray, size: tuple, **kwargs):
        homographyN = xysN2perspective(self.ctx_border._xypN, xysN)
        self.ctx_border._xypN = xysN
        self.ctx_border._size = size
        self.camera.perspective_(homographyN=homographyN, size=size, )
        for item in self:
            if isinstance(item, Movable):
                item.perspective_(homographyN=homographyN, size=size, **kwargs)
        return self

    def calibrate_(self, intrinsicN: np.ndarray, size: tuple):
        rot3 = self.camera.calibrate2rotationN_(intrinsicN, size=size, )
        if np.all(rot3 == np.eye(3)):
            return self
        for item in self:
            if isinstance(item, Movable3D):
                item.transform_(rotation=rot3)
        return self

    @property
    def xyxysN(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4))
        else:
            return np.stack([itm.xyxyN_projected(self.camera) for itm in self], axis=0)

    def empty(self):
        items = self.__new__(self.__class__)
        items.__init__(camera=self.camera, meta=self.meta, **self.kwargs)
        items.ctx_border = self.ctx_border
        return copy.deepcopy(items)

    def linear_(self, size: tuple, biasN: np.ndarray = BIAS_IDENTITY, scaleN: np.ndarray = SCALE_IDENTIIY, **kwargs):
        self.camera.linear_(biasN=biasN, scaleN=scaleN, size=size, )
        for item in self:
            if isinstance(item, Movable):
                item.linear_(biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        ImageLabel.linear_(self, biasN=biasN, scaleN=scaleN, size=size, **kwargs)
        return self

    def perspective_(self, size: tuple, homographyN: np.ndarray = HOMOGRAPHY_IDENTITY, **kwargs):
        self.camera.perspective_(homographyN=homographyN, size=size, )
        for item in self:
            if isinstance(item, Movable):
                item.perspective_(homographyN=homographyN, size=size, **kwargs)
        ImageLabel.perspective_(self, homographyN=homographyN, size=size, **kwargs)
        return self

    def clip_(self, xyxyN_rgn: np.ndarray, **kwargs):
        for item in self:
            item.clip3d_(xyxyN_rgn=xyxyN_rgn, camera=self.camera, **kwargs)
            if isinstance(item, Clipable):
                item.clip_(xyxyN_rgn=xyxyN_rgn, **kwargs)
        return self

    def filt_measure_(self, thres: float = -1):
        for i in range(len(self) - 1, -1, -1):
            item = self[i]
            if thres > 0 and item.measure_projected(self.camera) < thres:
                self.pop(i)
        return self

    def filt_measure(self, thres: float = -1):
        buffer = self.empty()
        for i, item in enumerate(self):
            if thres > 0 and item.measure_projected(self.camera) < thres:
                continue
            buffer.append(item)
        return buffer

    @staticmethod
    def from_xyzNs_surfNs_colorNs_cinds_confs(
            xyzNs: Sequence[np.ndarray], surfNs: Sequence[np.ndarray],
            colorNs: Sequence[np.ndarray], cinds: Union[np.ndarray, Sequence[int]],
            confs: Union[np.ndarray, Sequence[float]],
            num_cls: int, camera: MCamera,
            cind2name: Optional[Callable] = None):
        if confs is None:
            confs = np.ones_like(cinds, dtype=np.float32)
        label = StereoItemsLabel(camera=camera)
        for j, (xyz, surf, color, cind, conf) in enumerate(zip(xyzNs, surfNs, colorNs, cinds, confs)):
            vobj = XYZSColoredSurface(xyzsN=xyz, surfsN=surf, vcolorsN=color)
            item = StereoBoxItem(vobj, category=IndexCategory(cindN=cind, num_cls=num_cls, confN=conf))
            if cind2name is not None:
                item['name'] = cind2name(cind)
            label.append(item)
        return label

    @staticmethod
    def from_xyzNs_cinds_confs(
            xyzNs: Sequence[np.ndarray], cinds: Union[np.ndarray, Sequence[int]],
            confs: Union[np.ndarray, Sequence[float]],
            num_cls: int, camera: MCamera,
            cind2name: Optional[Callable] = None, **kwargs):
        label = StereoItemsLabel(camera=camera, **kwargs)
        for j, (xyz, cind, conf) in enumerate(zip(xyzNs, cinds, confs)):
            vobj = XYZSPoint(xyzsN=xyz)
            item = StereoBoxItem(vobj, category=IndexCategory(cindN=cind, num_cls=num_cls, confN=conf))
            if cind2name is not None:
                item['name'] = cind2name(cind)
            label.append(item)
        return label

    @staticmethod
    def from_xyzNs_edgeNs_cinds_confs(
            xyzNs: Sequence[np.ndarray],
            edgeNs: Sequence[np.ndarray],
            cinds: Union[np.ndarray, Sequence[int]],
            confs: Union[np.ndarray, Sequence[float]],
            num_cls: int, camera: MCamera,
            cind2name: Optional[Callable] = None, **kwargs):
        label = StereoItemsLabel(camera=camera, **kwargs)
        for j, (xyz, edge, cind, conf) in enumerate(zip(xyzNs, edgeNs, cinds, confs)):
            vobj = XYZSGraph(xyzsN=xyz, edgesN=edge)
            item = StereoBoxItem(vobj, category=IndexCategory(cindN=cind, num_cls=num_cls, confN=conf))
            if cind2name is not None:
                item['name'] = cind2name(cind)
            label.append(item)
        return label


# <editor-fold desc='注册json变换'>
@REGISTER_JSON_ENC.registry(StereoItemsLabel)
def _sitems_label2json_dct(sitems_label: StereoItemsLabel) -> dict:
    items = [obj2json_dct(item) for item in sitems_label]
    return {'meta': sitems_label.img_size,
            'camera': obj2json_dct(sitems_label.camera),
            'kwargs': obj2json_dct(sitems_label.kwargs),
            'items': items}


def _json_dct2sitems_label(json_dct: dict, cls=None) -> np.ndarray:
    img_size = json_dct['img_size']
    meta = json_dct['meta']
    kwargs = json_dct2obj(json_dct['kwargs'])
    camera = json_dct2obj(json_dct['camera'])
    items = [json_dct2obj(item) for item in json_dct['items']]
    return cls(items, meta=meta, camera=camera, **kwargs)


REGISTER_JSON_DEC[StereoItemsLabel.__name__] = partial(_json_dct2sitems_label, cls=StereoItemsLabel)


# </editor-fold>


# <editor-fold desc='渲染'>


def meshesT_lightsT2colorsT(meshesT: Meshes, camerasT, img_size: Tuple[int, int], lightsT) \
        -> (torch.Tensor, torch.Tensor, Fragments):
    device = meshesT.device
    # 渲染
    lightsT = lightsT.to(device)
    raster_settings = RasterizationSettings(
        image_size=(img_size[1], img_size[0]),
        bin_size=0,
        # blur_radius=1e-5
    )
    rasterizer = MeshRasterizer(cameras=camerasT, raster_settings=raster_settings).to(device)
    fragments = rasterizer(meshesT)
    verts = meshesT.verts_packed()  # (V, 3)
    faces = meshesT.faces_packed()  # (F, 3)
    vertex_normals = meshesT.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    texels = meshesT.sample_textures(fragments)
    materials = Materials(device=device)
    pixel_coords_in_camera = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords_in_camera, pixel_normals, lightsT, camerasT, materials
    )
    colors = (ambient + diffuse) * texels + specular
    blend_params = BlendParams(background_color=(0, 0, 0))
    imgs_masks_rnd = hard_rgb_blend(colors, fragments, blend_params)
    imgs_masks_rnd = imgs_masks_rnd.permute(0, 3, 1, 2)  # BHWC->BCHW
    imgs_masks_rnd = torch.clip(imgs_masks_rnd, min=0.0, max=1.0)
    imgs_rnd, masks_rnd = imgs_masks_rnd[:, :3], imgs_masks_rnd[:, 3:4]
    return imgs_rnd, masks_rnd, fragments


def meshesT2colorsT(meshesT: Meshes, camerasT, img_size: Tuple[int, int]) \
        -> (torch.Tensor, torch.Tensor, Fragments):
    device = meshesT.device
    raster_settings = RasterizationSettings(
        image_size=(img_size[1], img_size[0]),
        bin_size=0,
    )
    rasterizer = MeshRasterizer(cameras=camerasT, raster_settings=raster_settings).to(device)
    fragments = rasterizer(meshesT)
    colors = meshesT.sample_textures(fragments)
    blend_params = BlendParams(background_color=(0, 0, 0))
    imgs_masks_rnd = hard_rgb_blend(colors, fragments, blend_params)
    imgs_masks_rnd = imgs_masks_rnd.permute(0, 3, 1, 2)  # BHWC->BCHW
    imgs_masks_rnd = torch.clip(imgs_masks_rnd, min=0.0, max=1.0)
    imgs_rnd, masks_rnd = imgs_masks_rnd[:, :3], imgs_masks_rnd[:, 3:4]
    return imgs_rnd, masks_rnd, fragments


def meshesT_fragments2normsT(meshesT: Meshes, fragments) -> torch.Tensor:
    faces = meshesT.faces_packed()  # (F, 3)
    vertex_normals = meshesT.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    B, H, W, _, C = pixel_normals.size()
    pnorms = pixel_normals.view(B, H, W, C).permute(0, 3, 1, 2)
    return pnorms


def fragments2zbufsT(fragments) -> torch.Tensor:
    zbuf = fragments.zbuf
    B, H, W, _ = zbuf.size()
    zbuf = zbuf.view(B, 1, H, W)
    return zbuf


def t3drnd_meshesT(meshesT: Meshes, camerasT, img_size: Tuple[int, int], lightsT: Optional = None) \
        -> (torch.Tensor, torch.Tensor):
    # 渲染
    shader = HardPhongShader(device=meshesT.device, cameras=camerasT, lights=lightsT).to(meshesT.device)
    # shader = SoftPhongShader(device=meshesT.device, cameras=camerasT, lights=lightsT)
    raster_settings = RasterizationSettings(
        image_size=(img_size[1], img_size[0]),
        bin_size=0,
        # blur_radius=1e-5
    )
    rasterizer = MeshRasterizer(cameras=camerasT, raster_settings=raster_settings).to(meshesT.device)
    fragments = rasterizer(meshesT)
    imgs_masks_rnd = shader(fragments, meshesT)

    # z = fragments.zbuf.view(12, 1, 320, 320)
    # # show_labels(z)
    # # plt.pause(1e5)
    imgs_masks_rnd = imgs_masks_rnd.permute(0, 3, 1, 2)  # BHWC->BCHW
    imgs_masks_rnd = torch.clip(imgs_masks_rnd, min=0.0, max=1.0)
    imgs_rnd, masks_rnd = imgs_masks_rnd[:, :3], imgs_masks_rnd[:, 3:4]
    return imgs_rnd, masks_rnd


def t3drnd_sitems_label(label: StereoItemsLabel, device=DEVICE, lightsT: Optional = None) -> (
        torch.Tensor, torch.Tensor):
    camerasT = cameras2camerasT(label.camera, device=device)
    csurfs = [item.border for item in label]
    meshes = csurfs2meshesT_concat(csurfs, device=device)
    return t3drnd_meshesT(meshesT=meshes, camerasT=camerasT, img_size=label.img_size, lightsT=lightsT)


# </editor-fold>

# <editor-fold desc='标签转化'>
@BoxesLabel.REGISTER_COVERT.registry(StereoItemsLabel)
def _sitems_label2boxes_label(sitems: StereoItemsLabel, border_type=XYXYBorder):
    boxes_new = BoxesLabel(img_size=sitems.img_size, meta=sitems.meta, **sitems.kwargs)
    for sitem in sitems:
        box = sitem.project(sitems.camera)
        box = BoxItem.convert(box, border_type=border_type)
        boxes_new.append(box)
    return boxes_new


@InstsLabel.REGISTER_COVERT.registry(StereoItemsLabel)
def _sitems_label2insts_label(sitems: StereoItemsLabel):
    label_new = InstsLabel(img_size=sitems.img_size, meta=sitems.meta, **sitems.kwargs)
    for sitem in sitems:
        item = sitem.project(sitems.camera)
        item = InstItem.convert(item)
        label_new.append(item)
    return label_new

# </editor-fold>
