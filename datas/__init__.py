from .base.folder import FolderDataSource
from .cifar import CIFAR100, CIFAR10, CINIC10
from .coco import COCO, COCODataset, COCODetectionDataSet, \
    COCOInstanceDataSet, COCOSegmentationDataSet
from .cub200 import Cub200
from .dota import Dota
from .hrsc import HRSC, HRSCObj
from .imgnet import ImageNet, TinyImageNet
from .isaid import ISAID, ISAIDPatch, ISAIDObj, ISAIDPart
from .nuscenes import NUScenes, NUScenesLidarDataset, NUScenesStereoBoxDataset, NUImages, NUImagesPanopticDataset, \
    NUImagesDetectionDataset, NUImagesInstanceDataset
from .other import OXFlower, SteelSurface, ToyCarView, PCBDefect
from .svhn import SVHN
from .tools import *
from .transnet import InsulatorC, InsulatorD, InsulatorObj, InsulatorDI, TransNetwork, TransNetworkPyramid
from .voc import VOC, VOCCommon, VOCDetectionDataset, VOCInstanceDataset, VOCSegmentationDataset, \
    VOCDataset
