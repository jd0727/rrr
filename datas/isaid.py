import os

from datas.coco import COCO, COCOCommon
from datas.voc import VOC, VOCCommon

from .base import *


class ISAID(COCOCommon):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAID//',
        PLATFORM_SEV3090: '//home//datas-storage//ISAID',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER = COCO.IMG_FOLDER
    JSON_NAME = COCO.JSON_NAME
    JSON_FOLDER = COCO.JSON_FOLDER

    CLS_NAMES = ('storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'plane', 'ship',
                 'Swimming_pool', 'Harbor', 'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field',
                 'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')

    CIND2NAME_REMAPPER = None

    def __init__(self, root=None, json_name=JSON_NAME, img_folder=IMG_FOLDER, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES,
                 set_names=None, **kwargs):
        COCOCommon.__init__(self, root=root, json_name=json_name, img_folder=img_folder,
                            json_folder=json_folder, task_type=task_type,
                            cls_names=cls_names, set_names=set_names, **kwargs)


class ISAIDPatch(COCO):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAID//',
        PLATFORM_SEV3090: '//home//datas-storage//ISAID',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    IMG_FOLDER_FMT = 'patches_%s'
    JSON_NAME_FMT = 'instances_%s'
    JSON_FOLDER = 'annotation_ptch'

    CLS_NAMES = ISAID.CLS_NAMES

    ID2NAME_DICT = {
        9: 'Small_Vehicle', 8: 'Large_Vehicle', 14: 'plane', 2: 'storage_tank', 1: 'ship',
        11: 'Swimming_pool', 15: 'Harbor', 4: 'tennis_court', 6: 'Ground_Track_Field', 13: 'Soccer_ball_field',
        3: 'baseball_diamond', 7: 'Bridge', 5: 'basketball_court', 12: 'Roundabout', 10: 'Helicopter'}

    def __init__(self, root=None, json_name_fmt=JSON_NAME_FMT, img_folder_fmt=IMG_FOLDER_FMT, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, set_names=None, **kwargs):
        COCO.__init__(self, root=root, json_name_fmt=json_name_fmt, img_folder_fmt=img_folder_fmt,
                      json_folder=json_folder, task_type=task_type,
                      cls_names=cls_names, set_names=set_names, **kwargs)


class ISAIDObj(VOCCommon):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAIDObj//',
        PLATFORM_SEV3090: '//ses-datas//JD//ISAIDObj',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }
    ROOT_SEV_NEW1 = '//ses-datas//JD//ISAIDObj1'
    ROOT_SEV_NEW2 = '//ses-datas//JD//ISAIDObj2'
    ROOT_SEV_NEW3 = '//ses-datas//JD//ISAIDObj3'

    CLS_NAMES = ISAID.CLS_NAMES
    CLS_NAMES1 = ('Small_Vehicle',)
    CLS_NAMES2 = ('Large_Vehicle', 'ship')
    CLS_NAMES3 = ('storage_tank', 'plane', 'Swimming_pool', 'Harbor', 'tennis_court',
                  'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond',
                  'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')

    IMG_FOLDER = 'Patches'
    SET_FOLDER = 'ImageSets/Patch'
    COLORS = VOC.COLORS

    def __init__(self, root=None, cls_names=CLS_NAMES, colors=COLORS,
                 task_type=TASK_TYPE.INSTANCESEG, img_folder=IMG_FOLDER, anno_folder=VOC.ANNO_FOLDER,
                 mask_folder=VOC.MASK_FOLDER, inst_folder=VOC.INST_FOLDER, set_folder=SET_FOLDER, **kwargs):
        VOCCommon.__init__(self, root, cls_names, colors, task_type, mask_folder, inst_folder, set_folder,
                           img_folder, anno_folder, **kwargs)


class ISAIDPart(COCOCommon):
    CLS_NAMES = ISAID.CLS_NAMES
    CIND2NAME_REMAPPER = ISAID.CIND2NAME_REMAPPER

    IMG_FOLDER = ISAID.IMG_FOLDER
    JSON_NAME = ISAID.JSON_NAME
    JSON_FOLDER = ISAID.JSON_FOLDER

    REGISTER_ROOT = {
        PLATFORM_LAPTOP: '',
        PLATFORM_DESTOPLAB: 'D://Datasets//ISAIDPart//',
        PLATFORM_SEV3090: '//home//datas-storage//ISAIDPart',
        PLATFORM_SEV4090: '',
        PLATFORM_SEVTAITAN: '',
        PLATFORM_BOARD: ''
    }

    def __init__(self, root=None, json_name=JSON_NAME, img_folder_fmt=IMG_FOLDER, json_folder=JSON_FOLDER,
                 task_type=TASK_TYPE.DETECTION, cls_names=CLS_NAMES, cind2name_remapper=CIND2NAME_REMAPPER,
                 set_names=None, **kwargs):
        COCOCommon.__init__(self, root=root, json_name=json_name, img_folder=img_folder_fmt,
                            json_folder=json_folder, cind2name_remapper=cind2name_remapper, task_type=task_type,
                            cls_names=cls_names,
                            set_names=set_names, **kwargs)


if __name__ == '__main__':
    ds_voc = ISAIDObj()
