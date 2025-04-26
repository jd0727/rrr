from datas.voc import *
from .base import *


class InsulatorC(FolderDataSource):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorC//',
        PLATFORM_SEV3090: '/ses-datas/JD//InsulatorC//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '//home//user1//JD//Datasets//InsulatorC//',
        PLATFORM_BOARD: ''
    }

    CLS_NAMES = ('abnormal', 'normal')

    ROOT_RAWC = '//home//datas-storage//JD//RawC//unknown//'
    ROOT_BKGD = '//ses-img//JD//Bkgd//'

    def __init__(self, root=None, resample=None, cls_names=CLS_NAMES, **kwargs):
        super(InsulatorC, self).__init__(
            root=root, resample=resample, cls_names=cls_names, set_names=('train', 'test'), )


class TransNetworkC(FolderDataSource):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '/home/datas-storage/JD/TransNetC//',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }
    CLS_NAMES = ('insulator_normal', 'insulator_blast', 'insulator_comp', 'metal_normal', 'metal_rust',
                 'clamp_normal', 'clamp_rust', 'background')

    def __init__(self, root=None, resample=None, cls_names=CLS_NAMES, **kwargs):
        super(TransNetworkC, self).__init__(
            root=root, resample=resample, cls_names=cls_names, set_names=('train', 'test'), )


class InsulatorD(VOCCommon):
    CLS_NAMES = ('insulator_normal', 'insulator_blast', 'insulator_comp', 'cap', 'cap_missing',)
    CLS_NAMES_BORDER = ('insulator',)
    CLS_NAMES_MERGE = ('insulator_glass', 'insulator_comp',)
    COLORS = VOC.COLORS

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VOCCommon.MASK_FOLDER
    INST_FOLDER = VOCCommon.INST_FOLDER
    SET_FOLDER = VOCCommon.SET_FOLDER

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorD//',
        PLATFORM_SEV3090: '//home//data-storage//JD//InsulatorD',
        PLATFORM_SEV4090: '//home//data-storage//JD//InsulatorD',
        PLATFORM_SEVTAITAN: '/home/user/JD/Datasets/InsulatorD',
        PLATFORM_BOARD: '/home/jd/datas/DataSets/InsulatorD'
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val'), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class TransNetwork(VOCCommon):
    CLS_NAMES = ('insulator_normal', 'insulator_blast', 'insulator_comp', 'metal_normal', 'metal_rust',
                 'clamp_normal', 'clamp_rust')

    CLS_NAMES_ZN = ('正常玻璃绝缘子', '自爆玻璃绝缘子', '复合绝缘子', '正常金具', '锈蚀金具',
                    '正常线夹', '锈蚀线夹')
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: '',
        PLATFORM_SEV3090: '//home//data-storage//JD//TransNet',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VOCCommon.MASK_FOLDER
    INST_FOLDER = VOCCommon.INST_FOLDER
    SET_FOLDER = VOCCommon.SET_FOLDER
    COLORS = VOCCommon.COLORS

    GROUP_NAMES = (('insulator_normal', 'insulator_blast', 'insulator_comp',), ('metal_normal', 'metal_rust',),
                   ('clamp_normal', 'clamp_rust',),)
    CLUSTER_INDEX = MNameMapper.create_cluster_index(CLS_NAMES, GROUP_NAMES, offset=0)

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test',), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class LineDefect(TransNetwork):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//LineDefect/',
        PLATFORM_DESTOPLAB: 'D://Datasets//LineDefect/',
        PLATFORM_SEV3090: '/home/datas-storage/JD/LineDefect/',
        PLATFORM_SEV4090: '/home/datas-storage/JD/LineDefect/',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }


class TransNetworkPyramid(TransNetwork):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D://Datasets//TransNetPry/',
        PLATFORM_DESTOPLAB: 'D://Datasets//TransNetPry/',
        PLATFORM_SEV3090: '/home/datas-storage/JD/TransNetPry/',
        PLATFORM_SEV4090: '/home/datas-storage/JD/TransNetPry/',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }


class InsulatorDI(VOCCommon):
    CLS_NAMES = ('insulator_normal', 'insulator_blast')

    IMG_FOLDER = 'JPEGImages'
    ANNO_FOLDER = 'Annotations'
    MASK_FOLDER = VOCCommon.MASK_FOLDER
    INST_FOLDER = VOCCommon.INST_FOLDER
    SET_FOLDER = VOCCommon.SET_FOLDER
    COLORS = VOCCommon.COLORS

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D:\Datasets\InsulatorDI',
        PLATFORM_DESTOPLAB: 'D:\Datasets\InsulatorDI',
        PLATFORM_SEV3090: '//home//datas-storage//JD//InsulatorDI',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


class InsulatorObj(VOCCommon):
    COLORS = VOC.COLORS
    CLS_NAMES = ('insulator_glass',)
    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    ANNO_FOLDER = 'PatchAnnotations'
    INST_FOLDER = 'PatchInstance'
    MASK_FOLDER = 'PatchMask'

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//InsulatorObj/',
        PLATFORM_SEV3090: '//ses-datas//JD//InsulatorObj',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS, task_type=TASK_TYPE.DETECTION,
                 mask_folder=MASK_FOLDER, inst_folder=INST_FOLDER, set_folder=SET_FOLDER, img_folder=IMG_FOLDER,
                 anno_folder=ANNO_FOLDER, set_names=('train', 'test', 'val', 'trainval', 'example'), **kwargs):
        VOCCommon.__init__(self, root=root, cls_names=cls_names, colors=colors, task_type=task_type,
                           mask_folder=mask_folder, inst_folder=inst_folder, set_folder=set_folder,
                           img_folder=img_folder, anno_folder=anno_folder, set_names=set_names, **kwargs)


def recode_file(file_dir, offset=0.0, fmt='%6d'):
    file_names = sorted(os.listdir(file_dir))
    msgs = []
    for i, file_name in enumerate(file_names):
        code = fmt % (i + offset)
        file_anme_new = code + '.' + file_name.split('.')[-1]
        file_pth = os.path.join(file_dir, file_name)
        file_pth_new = os.path.join(file_dir, file_anme_new)
        os.rename(file_pth, file_pth_new)
        msgs.append(code + ' <- ' + file_name)
    return msgs

# if __name__ == '__main__':
#     ds = InsulatorD2()
#     print(ds.dataset(set_name='train'))
#     print(ds.dataset(set_name='test'))
# loader = ds.loader(set_name='test', batch_size=4, num_workers=0, aug_seq=None)
# imgs, labels = next(iter(loader))

# if __name__ == '__main__':
#     ds = InsulatorD(anno_folder='AnnotationsCap')
#     print(ds.dataset('val'))

# 拷贝val数据集所有图像
# if __name__ == '__main__':
#     ds = InsulatorD()
#     dataset = ds.dataset('val')
#     save_dir = os.path.join(ds.root, 'buff')
#     for img_pth in dataset.img_pths:
#         shutil.copy(img_pth, os.path.join(save_dir, os.path.basename(img_pth)))

# 拷贝blast数据集所有图像和标注
# if __name__ == '__main__':
#     ds = InsulatorD()
#     save_dir = ensure_folder_pth(os.path.join(ds.root, 'buff'))
#
#     for set_name in ['train', 'test', 'val']:
#         dataset = ds.dataset(set_name)
#         for anno_pth, img_pth in zip(dataset.anno_pths, dataset.img_pths):
#             label = VocDetectionDataset.prase_anno(anno_pth)
#             if any([item['name'] == 'insulator_blast' for item in label]):
#                 print(img_pth)
#                 shutil.copy(img_pth, os.path.join(save_dir, os.path.basename(img_pth)))
#                 shutil.copy(anno_pth, os.path.join(save_dir, os.path.basename(anno_pth)))
