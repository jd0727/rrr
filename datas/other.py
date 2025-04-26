import scipy
from datas.coco import *
from datas.base.folder import FolderClassificationDataset
from datas.voc import *
from .base import *


class PCBDefect(VOCCommon):
    CLS_NAMES = ('Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper')
    COLORS = VOCCommon.COLORS
    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VOCCommon.MASK_FOLDER
    INST_FOLDER = VOCCommon.INST_FOLDER
    SET_FOLDER = VOCCommon.SET_FOLDER

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//PCB//',
        PLATFORM_DESTOPLAB: 'D://Datasets//PCB//',
        PLATFORM_SEV3090: '//home//datas-storage//JD//PCB',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/PCB',
        PLATFORM_BOARD: '/home/jd/datas/DataSets/PCB'
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class ToyCarView(VOCCommon):
    CLS_NAMES = ('bump', 'granary', 'CrossWalk', 'cone', 'bridge', 'pig', 'tractor', 'corn')
    COLORS = VOCCommon.COLORS
    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VOCCommon.MASK_FOLDER
    INST_FOLDER = VOCCommon.INST_FOLDER
    SET_FOLDER = VOCCommon.SET_FOLDER

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//ToyCarView//',
        PLATFORM_DESTOPLAB: 'D://Datasets//ToyCarView//',
        PLATFORM_SEV3090: '/home/data-storage/ToyCarView',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/ToyCarView',
        PLATFORM_BOARD: '/home/jd/datas/DataSets/ToyCarView'
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'val',), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class SteelSurface(COCOCommon):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//SteelSurface//',
        PLATFORM_DESTOPLAB: 'D://Datasets//SteelSurface//',
        PLATFORM_SEV3090: '//home//datas-storage//SteelSurface',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER = COCO.IMG_FOLDER
    JSON_NAME = COCO.JSON_NAME
    JSON_FOLDER = COCO.JSON_FOLDER

    CLS_NAMES = ('crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches')

    CIND2NAME_REMAPPER = None

    def __init__(self, root=None, json_name=JSON_NAME, img_folder=IMG_FOLDER, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=None,
                 set_names=None, **kwargs):
        COCOCommon.__init__(self, root=root, json_name=json_name, img_folder=img_folder,
                            json_folder=json_folder, cind2name_remapper=cind2name_remapper, task_type=task_type,
                            cls_names=cls_names, set_names=set_names, **kwargs)


class OXFlower(FolderDataSource):
    @staticmethod
    def make_dataset(setid_pth, imgllb_pth, img_dir, root):
        setid = scipy.io.loadmat(setid_pth)
        imgllb = scipy.io.loadmat(imgllb_pth)
        labels = imgllb['labels'][0]
        img_pths = [os.path.join(img_dir, img_name) for img_name in sorted(os.listdir(img_dir))]
        for set_name, key in zip(['train', 'test', 'val'], ['trnid', 'tstid', 'valid']):
            set_dir = ensure_folder_pth(os.path.join(root, set_name))
            ids = np.array(setid[key][0]) - 1
            for id in ids:
                print(id)
                label = labels[id]
                lb_dir = ensure_folder_pth(os.path.join(set_dir, 'c' + str(label)))
                img_pth = img_pths[id]
                shutil.copy(img_pth, os.path.join(lb_dir, os.path.basename(img_pth)))
        return True

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//OXFlower//',
        PLATFORM_DESTOPLAB: 'D://Datasets//OXFlower//',
        PLATFORM_SEV3090: '//home//data-storage//OXFlower//',
        PLATFORM_SEV4090: '//home//data-storage//OXFlower//',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//OXFlower//',
        PLATFORM_BOARD: ''
    }
    CLS_NAMES = tuple(['c%d' % i for i in range(1, 103)])
    SET_NAMES = ('train', 'test')

    def __init__(self, root=None, resample=None, cls_names=CLS_NAMES, set_names=SET_NAMES, task_type=TASK_TYPE.AUTO,
                 **kwargs):
        FolderDataSource.__init__(
            self, root=root, cls_names=cls_names, resample=resample, set_names=set_names, task_type=task_type,
            **kwargs)
