import base64

try:
    from datas.coco import mask_utils
except Exception as e:
    pass
from datas.base import MNameMapper,  MDataset, MDataSource, TASK_TYPE
from datas.base import load_nu_pcdbin
from utils import *


# <editor-fold desc='NUBase'>

def camera_proj(xyzsN: np.ndarray, cam_intrinsic: np.ndarray):
    xyzsN_proj = xyzsN @ cam_intrinsic
    xysN_proj, zsN = xyzsN_proj[:, :2], xyzsN_proj[:, 2:]
    xysN_proj = xysN_proj / zsN
    return xysN_proj, zsN[:, 0]


class NUDataset(MDataset):

    def __init__(self, root, set_name, img_folder, json_folder):
        self._root = root
        self._set_name = set_name
        self._json_folder = json_folder
        self._img_folder = img_folder

    @property
    def root(self):
        return self._root

    @property
    def set_name(self):
        return self._set_name

    @property
    def img_folder(self):
        return self._img_folder

    @property
    def json_folder(self):
        return self._json_folder

    @property
    def json_dir(self):
        return os.path.join(self._root, self._json_folder)

    @property
    def img_dir(self):
        return os.path.join(self._root, self._img_folder)

    def _flit_by(self, dct: dict, fn: Optional[Callable] = None) -> dict:
        fltd = OrderedDict()
        for key, value in dct.items():
            flag = value if fn is None else fn(value)
            if flag:
                fltd[key] = value
        return fltd

    def _split_by(self, dct: dict, cluster_names: list, fn: Optional[Callable] = None) -> List[dict]:
        clusters = OrderedDict([(name, {}) for name in cluster_names])
        for key, value in dct.items():
            clusters[value if fn is None else fn(value)][key] = value
        return list(clusters.values())

    def _load_json(self, json_name: str, fn: Optional[Callable] = None) -> dict:
        json_msg = load_json(os.path.join(self.json_dir, ensure_extend(json_name, 'json')))
        json_mapper = OrderedDict()
        for item in json_msg:
            value = item if fn is None else fn(item)
            json_mapper[item.pop('token')] = value
        return json_mapper

    def _cluster_by(self, tokens: Iterable, items: Iterable, key: str) -> list:
        clusters = OrderedDict([(tk, []) for tk in tokens])
        for item in items:
            clusters[item[key]].append(item)
        return list(clusters.values())

    def _map_by(self, items: Iterable, mapper: dict, key_old: str, key_new: str) -> None:
        for item in items:
            value_old = item.pop(key_old)
            if isinstance(value_old, list):
                item[key_new] = [mapper[v] for v in value_old]
            else:
                item[key_new] = mapper[value_old]
        return None


# </editor-fold>

# <editor-fold desc='NUScenes'>
class NUScenesDataset(MNameMapper, NUDataset):

    def __init__(self, root, set_name, img_folder='samples', json_folder='v1.0-mini',
                 sample_name='sample', smpdata_name='sample_data',
                 smpanno_name='sample_annotation', instance_name='instance', scene_name='scene',
                 sensor_name='sensor', calisnr_name='calibrated_sensor', egopos_name='ego_pose',
                 category_name='category', attribute_name='attribute', log_name='log',
                 cls_names=None, **kwargs):
        NUDataset.__init__(self, root, set_name, img_folder, json_folder)

        # 加载简单映射
        self._cname_mapper = self._load_json(category_name, fn=lambda item: item['name'].split('.')[-1])
        self._log_mapper = self._load_json(log_name, fn=None)
        self._sample_mapper = self._load_json(sample_name, fn=None)
        self._egopos_mapper = self._load_json(egopos_name, fn=None)
        self._sensor_mapper = self._load_json(sensor_name, fn=None)
        self._calisnr_mapper = self._load_json(calisnr_name, fn=None)
        self._attr_mapper = self._load_json(attribute_name, fn=lambda item: item['name'])
        self._instance_mapper = self._load_json(instance_name, fn=None)
        smpdata_mapper = self._load_json(smpdata_name, fn=None)
        self._smpanno_mapper = self._load_json(smpanno_name, fn=None)
        self._scene_mapper = self._load_json(scene_name, fn=None)

        if cls_names is None:
            cls_names = list(sorted(self._cname_mapper.values()))
        MNameMapper.__init__(self, cls_names=cls_names)

        self._map_by(self._calisnr_mapper.values(), self._sensor_mapper,
                     key_old='sensor_token', key_new='sensor')
        self._map_by(smpdata_mapper.values(), self._calisnr_mapper,
                     key_old='calibrated_sensor_token', key_new='calibrated_sensor')
        self._map_by(smpdata_mapper.values(), self._egopos_mapper,
                     key_old='ego_pose_token', key_new='ego_pose')
        self._map_by(self._smpanno_mapper.values(), self._instance_mapper,
                     key_old='instance_token', key_new='instance')
        self._map_by(self._smpanno_mapper.values(), self._attr_mapper,
                     key_old='attribute_tokens', key_new='attributes')
        self._map_by(self._instance_mapper.values(), self._cname_mapper,
                     key_old='category_token', key_new='category')

        radardata_mapper, lidardata_mapper, imgdata_mapper = self._split_by(
            smpdata_mapper, cluster_names=['radar', 'lidar', 'camera'],
            fn=lambda item: item['calibrated_sensor']['sensor']['modality'])

        self._radardata_mapper = radardata_mapper
        self._lidardata_mapper = lidardata_mapper
        self._imgdata_mapper = imgdata_mapper

        radardatass = self._cluster_by(self._sample_mapper.keys(), radardata_mapper.values(), key='sample_token')
        lidardatass = self._cluster_by(self._sample_mapper.keys(), lidardata_mapper.values(), key='sample_token')
        imgdatass = self._cluster_by(self._sample_mapper.keys(), imgdata_mapper.values(), key='sample_token')
        smpannoss = self._cluster_by(self._sample_mapper.keys(), self._smpanno_mapper.values(), key='sample_token')

        metas, infos, obj_annoss = self._build_datas(imgdatass, lidardatass, radardatass, smpannoss)
        self._metas = metas
        self._infos = infos
        self._obj_annoss = obj_annoss



    @abstractmethod
    def _build_datas(self, imgdatass, lidardatass, radardatass, smpannoss):
        pass

    @property
    def labels(self):
        return [self._index2label(index) for index in range(len(self))]

    @property
    def metas(self):
        return self._metas

    def __len__(self):
        return len(self._metas)

    def _index2img(self, index: int):
        img_pth = os.path.join(self._root, self._infos[index]['filename'])
        img = load_img_cv2(img_pth)
        return img

    def _meta2data(self, meta: str):
        return self._index2data(self._metas.index(meta))

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _meta2label(self, meta: str):
        return self._index2label(self._metas.index(meta))


class NUScenesStereoBoxDataset(NUScenesDataset):

    def _build_datas(self, imgdatass, lidardatass, radardatass, smpannoss):
        metas = []
        infos = []
        obj_annoss = []
        for imgdatas, smpannos in zip(imgdatass, smpannoss):
            for imgdata in imgdatas:

                camera = imgdata['calibrated_sensor']
                cam_intrinsic = np.array(camera['camera_intrinsic'])
                cam_translation = np.array(camera['translation'])
                cam_rotaion_inv = quaterN_inv(np.array(camera['rotation']))

                ego_pose = imgdata['ego_pose']
                ego_translation = np.array(ego_pose['translation'])
                ego_rotaion_inv = quaterN_inv(np.array(ego_pose['rotation']))

                meta = os.path.splitext(os.path.basename(imgdata['filename']))[0]

                obj_annos = []
                for sampanno in smpannos:
                    translation = np.array(sampanno['translation'])
                    obj_size = np.array(sampanno['size'])
                    whl = np.array([obj_size[1], obj_size[0], obj_size[2]])
                    rotation = np.array(sampanno['rotation'])
                    # 变换坐标系
                    translation = (translation - ego_translation) @ quaterN2rot3N(ego_rotaion_inv).T
                    rotation = quaterN_mul(ego_rotaion_inv, rotation)
                    translation = (translation - cam_translation) @ quaterN2rot3N(cam_rotaion_inv).T
                    rotation = quaterN_mul(cam_rotaion_inv, rotation)
                    # 边框投影
                    imgdata['intrinsic'] = cam_intrinsic
                    if np.all(translation[-1] > 0):
                        sampanno = copy.copy(sampanno)
                        sampanno['xyzN'] = translation
                        sampanno['whlN'] = whl
                        sampanno['quater'] = rotation
                        obj_annos.append(sampanno)
                obj_annoss.append(obj_annos)
                metas.append(meta)
                infos.append(imgdata)
            break

        return metas, infos, obj_annoss

    def _index2data(self, index: int):
        info = self._infos[index]
        obj_annos = self._obj_annoss[index]
        meta = self._metas[index]
        img_pth = os.path.join(self._root, info['filename'])
        img = load_img_cv2(img_pth)
        label = self.obj_annos2label(obj_annos, info=info, meta=meta)
        return img, label

    def obj_annos2label(self, obj_annos, info, meta):
        img_size = (info['width'], info['height'])
        camera = MCamera(intrinsicN=np.array(info['intrinsic']), size=img_size)
        label = StereoItemsLabel( meta=meta, camera=camera)

        for obj_anno in obj_annos:
            name = obj_anno['instance']['category']
            cind = self.name2cind(name)
            category = IndexCategory(cind, confN=1, num_cls=self.num_cls)

            xyzwhlq = np.concatenate([obj_anno['xyzN'], obj_anno['whlN'], np.array(obj_anno['quater'])], axis=0)
            sbox = XYZWHLQBorder(xyzwhlq)

            sitem = StereoBoxItem(sbox, category=category, name=name)
            label.append(sitem)
        label.clip_(np.array([0, 0, img_size[0], img_size[1]]))
        label.filt_measure_(1)
        return label


class NUScenesLidarDataset(NUScenesDataset):

    def _build_datas(self, imgdatass, lidardatass, radardatass, smpannoss):
        metas = []
        infos = []
        obj_annoss = []
        for imgdatas, lidardatas in zip(imgdatass, lidardatass):
            xyzs_all = []
            intsys_all = []
            for lidardata in lidardatas:
                pcl_pth = os.path.join(self._root, lidardata['filename'])
                lidar = lidardata['calibrated_sensor']
                lidar_translation = np.array(lidar['translation'])
                lidar_rotaion = np.array(lidar['rotation'])
                #

                ego_pose = lidardata['ego_pose']
                ego_translation = np.array(ego_pose['translation'])
                ego_rotaion = np.array(ego_pose['rotation'])

                xyz_intsys = load_nu_pcdbin(pcl_pth)
                pnts = xyz_intsys[:, :3]
                pnts = pnts @ quaterN2rot3N(lidar_rotaion).T + lidar_translation
                pnts = pnts @ quaterN2rot3N(ego_rotaion).T + ego_translation
                xyzs_all.append(pnts)
                intsys_all.append(xyz_intsys[:, 3])
            xyzs_all = np.concatenate(xyzs_all, axis=0)
            intsys_all = np.concatenate(intsys_all, axis=0)

            for imgdata in imgdatas:
                camera = imgdata['calibrated_sensor']
                cam_intrinsic = np.array(camera['camera_intrinsic'])
                cam_translation = np.array(camera['translation'])
                cam_rotaion_inv = quaterN_inv(np.array(camera['rotation']))

                ego_pose = imgdata['ego_pose']
                ego_translation = np.array(ego_pose['translation'])
                ego_rotaion_inv = quaterN_inv(np.array(ego_pose['rotation']))

                meta = os.path.splitext(os.path.basename(imgdata['filename']))[0]

                xyzs_cur = (xyzs_all - ego_translation) @ quaterN2rot3N(ego_rotaion_inv).T
                xyzs_cur = (xyzs_cur - cam_translation) @ quaterN2rot3N(cam_rotaion_inv).T
                xys_cur, zs_cur = camera_proj(xyzs_cur, cam_intrinsic.T)

                fltr_valid = zs_cur > 0
                xys_cur = xys_cur[fltr_valid]
                zs_cur = zs_cur[fltr_valid]
                intsys_cur = intsys_all[fltr_valid]

                obj_annoss.append([dict(xys=xys_cur, zs=zs_cur, intsys=intsys_cur)])
                metas.append(meta)
                infos.append(imgdata)
            break

        return metas, infos, obj_annoss

    def obj_annos2label(self, obj_annos, img_size, meta):

        label = ImageItemsLabel(img_size=img_size, meta=meta)
        for obj_anno in obj_annos:
            pcd2d = XYSPoint(xysN=obj_anno['xys'], size=img_size)
            label.append(pcd2d)
        return label

    def _index2data(self, index: int):
        info = self._infos[index]
        obj_annos = self._obj_annoss[index]
        img_size = (info['width'], info['height'])
        meta = self._metas[index]
        img_pth = os.path.join(self._root, info['filename'])
        img = load_img_cv2(img_pth)
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=meta)
        return img, label


class NUScenes(MDataSource):
    CLS_NAMES_FULL = (
        'animal', 'flat.driveable_surface', 'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
        'movable_object.barrier', 'movable_object.debris', 'movable_object.pushable_pullable',
        'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
        'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.ego',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle',
        'vehicle.trailer', 'vehicle.truck'
    )
    CLS_NAMES = tuple([name.split('.')[-1] for name in CLS_NAMES_FULL])

    JSON_FOLDER = 'v1.0-' + PLACEHOLDER.SET_NAME
    IMG_FOLDER = 'samples'

    SAMPLE_NAME = 'sample'
    SMPDATA_NAME = 'sample_data'
    SMPANNO_NAME = 'sample_annotation'
    INSTANCE_NAME = 'instance'
    SCENE_NAME = 'scene'

    SENSOR_NAME = 'sensor'
    CALISNR_NAME = 'calibrated_sensor'
    EGOPOS_NAME = 'ego_pose'
    CATEGORY_NAME = 'category'
    ATTRIBUTE_NAME = 'attribute'
    LOG_NAME = 'log'

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//NUScenes//',
        PLATFORM_DESTOPLAB: 'D://Datasets//NUScenes//',
        PLATFORM_SEV3090: '/home/data-storage/NUScenes',
        PLATFORM_SEV4090: '/home/data-storage/NUScenes',
        PLATFORM_SEVTAITAN: '//home//exspace//dataset//NUScenes',
        PLATFORM_BOARD: ''
    }

    REGISTER_BUILDER = {
        TASK_TYPE.STEREODET: NUScenesStereoBoxDataset,
    }

    SET_NAMES = ('train', 'test', 'val', 'mini')

    def __init__(self, root=None, set_names=SET_NAMES, task_type=TASK_TYPE.STEREODET,
                 json_folder=JSON_FOLDER, img_folder=IMG_FOLDER,
                 sample_name=SAMPLE_NAME, smpdata_name=SMPDATA_NAME, smpanno_name=SMPANNO_NAME,
                 instance_name=INSTANCE_NAME, scene_name=SCENE_NAME, sensor_name=SENSOR_NAME, calisnr_name=CALISNR_NAME,
                 egopos_name=EGOPOS_NAME, category_name=CATEGORY_NAME, attribute_name=ATTRIBUTE_NAME, log_name=LOG_NAME,
                 cls_names=CLS_NAMES, **kwargs):
        MDataSource.__init__(self, root=root, set_names=set_names, task_type=task_type)

        self.img_folder = img_folder
        self.json_folder = json_folder
        self.smpanno_name = smpanno_name
        self.sample_name = sample_name
        self.smpdata_name = smpdata_name
        self.sensor_name = sensor_name
        self.calisnr_name = calisnr_name
        self.egopos_name = egopos_name
        self.category_name = category_name
        self.attribute_name = attribute_name
        self.log_name = log_name
        self.instance_name = instance_name
        self.scene_name = scene_name
        self.kwargs = kwargs
        self.cls_names = cls_names

    def _dataset(self, set_name, task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = NUScenes.REGISTER_BUILDER[task_type]

        kwargs_update = dict(root=self.root, json_folder=self.json_folder, img_folder=self.img_folder,
                             set_name=set_name, cls_names=self.cls_names,
                             smpanno_name=self.smpanno_name, sample_name=self.sample_name,
                             smpdata_name=self.smpdata_name, sensor_name=self.sensor_name,
                             calisnr_name=self.calisnr_name, egopos_name=self.egopos_name,
                             category_name=self.category_name, attribute_name=self.attribute_name,
                             log_name=self.log_name, instance_name=self.instance_name, scene_name=self.scene_name, )
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        kwargs_update['img_folder'] = format_set_folder(set_name, formatter=kwargs_update['img_folder'])
        kwargs_update['json_folder'] = format_set_folder(set_name, formatter=kwargs_update['json_folder'])
        dataset = builder(**kwargs_update)
        return dataset


# </editor-fold>

# <editor-fold desc='NUImages'>
class NUImagesDataset(MNameMapper, NUDataset):

    def __init__(self, root, set_name, json_folder='v1.0-mini', img_folder='samples',
                 objann_name='object_ann', sample_name='sample', smpdata_name='sample_data',
                 sensor_name='sensor', calisnr_name='calibrated_sensor', egopos_name='ego_pose',
                 category_name='category', attribute_name='attribute', log_name='log',
                 surface_name='surface_ann', cls_names=None, **kwargs):
        NUDataset.__init__(self, root, set_name, img_folder, json_folder)

        # 加载简单映射
        self._log_mapper = self._load_json(log_name, fn=None)
        self._cname_mapper = self._load_json(category_name, fn=lambda item: item['name'].split('.')[-1])
        self._sample_mapper = self._load_json(sample_name, fn=None)
        self._egopos_mapper = self._load_json(egopos_name, )
        self._sensor_mapper = self._load_json(sensor_name, fn=lambda item: item['channel'])
        self._calisnr_mapper = self._load_json(calisnr_name, fn=None)
        self._attr_mapper = self._load_json(attribute_name, fn=lambda item: item['name'])
        self._objanno_mapper = self._load_json(objann_name, fn=None)
        self._imgdata_mapper = self._load_json(smpdata_name, fn=None)
        self._surface_mapper = self._load_json(surface_name, fn=None)

        if cls_names is None:
            cls_names = list(sorted(self._cname_mapper.values()))
        MNameMapper.__init__(self, cls_names=cls_names)

        self._map_by(self._calisnr_mapper.values(), self._sensor_mapper,
                     key_old='sensor_token', key_new='sensor')
        self._map_by(self._imgdata_mapper.values(), self._calisnr_mapper,
                     key_old='calibrated_sensor_token', key_new='calibrated_sensor')
        self._map_by(self._imgdata_mapper.values(), self._egopos_mapper,
                     key_old='ego_pose_token', key_new='ego_pose')
        # self._map_by(self._imgdata_mapper.values(), self._egopos_mapper,
        #              key_old='log_token', key_new='log')
        self._map_by(self._objanno_mapper.values(), self._attr_mapper,
                     key_old='attribute_tokens', key_new='attributes')
        self._map_by(self._objanno_mapper.values(), self._cname_mapper,
                     key_old='category_token', key_new='category')
        self._map_by(self._surface_mapper.values(), self._cname_mapper,
                     key_old='category_token', key_new='category')

        self._imgdata_mapper = self._flit_by(self._imgdata_mapper, fn=lambda data: 'samples/' in data['filename'])

        objdatass = self._cluster_by(self._imgdata_mapper.keys(), self._objanno_mapper.values(),
                                     key='sample_data_token')
        surfdatass = self._cluster_by(self._imgdata_mapper.keys(), self._surface_mapper.values(),
                                      key='sample_data_token')
        imgdatas = list(self._imgdata_mapper.values())

        metas, infos, obj_annoss = self._build_datas(imgdatas, objdatass, surfdatass)
        self._metas = metas
        self._infos = infos
        self._obj_annoss = obj_annoss



    @abstractmethod
    def _build_datas(self, imgdatas, objdatass, surfdatass):
        pass

    @property
    def labels(self):
        return [self._index2label(index) for index in range(len(self))]

    @property
    def metas(self):
        return self._metas

    def __len__(self):
        return len(self._metas)

    def _index2img(self, index: int):
        img_pth = os.path.join(self._root, self._infos[index]['filename'])
        img = load_img_cv2(img_pth)
        return img

    def _meta2data(self, meta: str):
        return self._index2data(self._metas.index(meta))

    def _meta2img(self, meta: str):
        return self._index2img(self._metas.index(meta))

    def _meta2label(self, meta: str):
        return self._index2label(self._metas.index(meta))


class NUImagesDetectionDataset(NUImagesDataset):

    def _build_datas(self, imgdatas, objdatass, surfdatass):
        metas = []
        infos = []
        obj_annoss = []
        for imgdata, objdatas in zip(imgdatas, objdatass):
            meta = os.path.splitext(os.path.basename(imgdata['filename']))[0]
            imgdata['meta'] = meta
            metas.append(meta)
            infos.append(imgdata)
            obj_annoss.append(objdatas)
        return metas, infos, obj_annoss

    def obj_annos2label(self, obj_annos, img_size, meta):
        label = BoxesLabel(img_size=img_size, meta=meta)
        for obj_anno in obj_annos:
            border = XYXYBorder(obj_anno['bbox'], size=img_size)
            name = obj_anno['category']
            cind = self.name2cind(name)
            cate = IndexCategory(cindN=cind, num_cls=self.num_cls)
            box = BoxItem(border=border, category=cate, name=name)
            label.append(box)
        return label

    def _index2label(self, index: int):
        obj_annos = self._obj_annoss[index]
        info = self._infos[index]
        img_size = (info['width'], info['height'])
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=self._metas[index], )
        return label

    def _index2data(self, index: int):
        obj_annos = self._obj_annoss[index]
        info = self._infos[index]
        img_size = (info['width'], info['height'])
        img_pth = os.path.join(self._root, info['filename'])
        img = load_img_cv2(img_pth)
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=self._metas[index])
        return img, label


class NUImagesInstanceDataset(NUImagesDetectionDataset):

    def obj_annos2label(self, obj_annos, img_size, meta):
        label = InstsLabel(img_size=img_size, meta=meta)
        for obj_anno in obj_annos:
            border = XYXYBorder(obj_anno['bbox'], size=img_size)
            name = obj_anno['category']
            cind = self.name2cind(name)
            category = IndexCategory(cindN=cind, confN=1, num_cls=self.num_cls)
            rle_dct = obj_anno['mask']
            rle_dct['counts'] = base64.b64decode(rle_dct['counts'])
            maskN = np.array(mask_utils.decode(rle_dct), dtype=np.float32)
            rgn = RefValRegion.from_maskNb_xyxyN(maskN, border._xyxyN)
            inst = InstItem(border=border, rgn=rgn, category=category, name=name)
            label.append(inst)
        return label

    def _index2label(self, index: int):
        obj_annos = self._obj_annoss[index]
        info = self._infos[index]
        img_size = (info['width'], info['height'])
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=self._metas[index])
        return label

    def _index2data(self, index: int):
        obj_annos = self._obj_annoss[index]
        info = self._infos[index]
        img_size = (info['width'], info['height'])
        img_pth = os.path.join(self._root, info['filename'])
        img = load_img_cv2(img_pth)
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=self._metas[index])
        return img, label


class NUImagesPanopticDataset(NUImagesDataset):

    def _build_datas(self, imgdatas, objdatass, surfdatass):
        metas = []
        infos = []
        obj_annoss = []
        for imgdata, objdatas, surfdatas in zip(imgdatas, objdatass, surfdatass):
            meta = os.path.splitext(os.path.basename(imgdata['filename']))[0]
            imgdata['meta'] = meta
            metas.append(meta)
            infos.append(imgdata)
            obj_annoss.append(objdatas + surfdatas)

        return metas, infos, obj_annoss

    def obj_annos2label(self, obj_annos, img_size, meta):
        label = SegsLabel(img_size=img_size, meta=meta)
        for obj_anno in obj_annos:
            name = obj_anno['category']
            cind = self.name2cind(name)
            category = IndexCategory(cindN=cind, confN=1, num_cls=self.num_cls)

            rle_dct = obj_anno['mask']
            rle_dct['counts'] = base64.b64decode(rle_dct['counts'])
            maskN = np.array(mask_utils.decode(rle_dct), dtype=bool)
            rgn = AbsBoolRegion(maskN)
            seg = SegItem(rgn=rgn, category=category, name=name)
            label.append(seg)
        return label

    def _index2label(self, index: int):
        obj_annos = self._obj_annoss[index]
        info = self._infos[index]
        img_size = (info['width'], info['height'])
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=self._metas[index])
        return label

    def _index2data(self, index: int):
        obj_annos = self._obj_annoss[index]
        info = self._infos[index]
        img_size = (info['width'], info['height'])
        img_pth = os.path.join(self._root, info['filename'])
        img = load_img_cv2(img_pth)
        label = self.obj_annos2label(obj_annos, img_size=img_size, meta=self._metas[index])
        return img, label


class NUImages(MDataSource):
    CLS_NAMES_FULL = (
        'animal', 'flat.driveable_surface', 'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
        'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
        'movable_object.barrier', 'movable_object.debris', 'movable_object.pushable_pullable',
        'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
        'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.ego',
        'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle',
        'vehicle.trailer', 'vehicle.truck'
    )
    CLS_NAMES = tuple([name.split('.')[-1] for name in CLS_NAMES_FULL])

    JSON_FOLDER = 'v1.0-' + PLACEHOLDER.SET_NAME
    IMG_FOLDER = 'samples'
    OBJANN_NAME = 'object_ann'
    SAMPLE_NAME = 'sample'
    SMPDATA_NAME = 'sample_data'
    SENSOR_NAME = 'sensor'
    CALISNR_NAME = 'calibrated_sensor'
    EGOPOS_NAME = 'ego_pose'
    CATEGORY_NAME = 'category'
    ATTRIBUTE_NAME = 'attribute'
    LOG_NAME = 'log'
    SURFACE_NAME = 'surface_ann'

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//NUImages//',
        PLATFORM_DESTOPLAB: 'D://Datasets//NUImages//',
        PLATFORM_SEV3090: '/home/datas-storage/NUImages',
        PLATFORM_SEV4090: '/home/datas-storage/NUImages',
        PLATFORM_SEVTAITAN: '//home//exspace//dataset//NUImages',
        PLATFORM_BOARD: ''
    }

    REGISTER_BUILDER = {
        TASK_TYPE.DETECTION: NUImagesDetectionDataset,
        TASK_TYPE.PANOPTICSEG: NUImagesPanopticDataset,
        TASK_TYPE.INSTANCESEG: NUImagesInstanceDataset,
    }

    SET_NAMES = ('train', 'test', 'val', 'mini')

    def __init__(self, root=None, set_names=SET_NAMES, task_type=TASK_TYPE.DETECTION,
                 json_folder=JSON_FOLDER, img_folder=IMG_FOLDER,
                 objann_name=OBJANN_NAME, sample_name=SAMPLE_NAME, smpdata_name=SMPDATA_NAME,
                 sensor_name=SENSOR_NAME, calisnr_name=CALISNR_NAME, egopos_name=EGOPOS_NAME,
                 category_name=CATEGORY_NAME, attribute_name=ATTRIBUTE_NAME, log_name=LOG_NAME,
                 surface_name=SURFACE_NAME, cls_names=CLS_NAMES, **kwargs):
        MDataSource.__init__(self, root=root, set_names=set_names, task_type=task_type)

        self.img_folder = img_folder
        self.json_folder = json_folder
        self.objann_name = objann_name
        self.sample_name = sample_name
        self.smpdata_name = smpdata_name
        self.sensor_name = sensor_name
        self.calisnr_name = calisnr_name
        self.egopos_name = egopos_name
        self.category_name = category_name
        self.attribute_name = attribute_name
        self.log_name = log_name
        self.surface_name = surface_name

        self.kwargs = kwargs
        self.cls_names = cls_names

    def _dataset(self, set_name, task_type=None, **kwargs):
        task_type = task_type if task_type is not None else self.task_type
        builder = NUImages.REGISTER_BUILDER[task_type]

        kwargs_update = dict(root=self.root, json_folder=self.json_folder, img_folder=self.img_folder,
                             set_name=set_name, cls_names=self.cls_names,
                             objann_name=self.objann_name, sample_name=self.sample_name,
                             smpdata_name=self.smpdata_name, sensor_name=self.sensor_name,
                             calisnr_name=self.calisnr_name, egopos_name=self.egopos_name,
                             category_name=self.category_name, attribute_name=self.attribute_name,
                             log_name=self.log_name, surface_name=self.surface_name)
        kwargs_update.update(self.kwargs)
        kwargs_update.update(kwargs)
        kwargs_update['img_folder'] = format_set_folder(set_name, formatter=kwargs_update['img_folder'])
        kwargs_update['json_folder'] = format_set_folder(set_name, formatter=kwargs_update['json_folder'])
        dataset = builder(**kwargs_update)
        return dataset

# </editor-fold>
