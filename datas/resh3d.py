from datas.base import TASK_TYPE, DATA_MODE
from datas.base.inplabel import InpPKLDataSource
from utils import PLATFORM_SEV3090, np, PLATFORM_SEV4090, PLATFORM_LAPTOP, chain, show_distribute, plt
import os


class Resh3DDataSource(InpPKLDataSource):
    REGISTER_ROOT = {
        PLATFORM_LAPTOP: 'D:\Datasets/Resh3D',
        PLATFORM_SEV3090: '/ses-data/JD/Resh3D',
        PLATFORM_SEV4090: '/ses-data/JD/Resh3D',
    }
    LABEL_EXTEND = 'pkl'
    IMG_EXTEND = 'jpg'
    IMG_FOLDER = 'images'
    LABEL_FOLDER = 'labels'
    WOBJ_FOLDER = 'wobjs'
    INFO_NAME = 'info'

    CLUSTERS = {
        'incera_arm': ['incera_arm-1', 'incera_arm-2', 'incera_arm-3', 'incera_arm-4', ],
        'incera_col': ['incera_col-1', 'incera_col-2', 'incera_col-3', 'incera_col-4', 'incera_col-5',
                       'incera_col-6', ],
        'incera_bowl': ['incera_bowl-1', 'incera_bowl-2', 'incera_bowl-3', 'incera_bowl-4', ],
        'clp_strain': ['clp_strain-1', 'clp_strain-2', 'clp_strain-3', 'clp_strain-4', ],
        'hammer': ['hammer-1', 'hammer-2'],
        'incomp_arm': ['incomp_arm-1', 'incomp_arm-2', 'incomp_arm-3'],
        'incomp_bar': ['incomp_bar-1', 'incomp_bar-2', 'incomp_bar-3'],
        'incomp_col': ['incomp_col-1', 'incomp_col-2', 'incomp_col-3', 'incomp_pin-1', 'incomp_pin-2'],
    }
    CLUSTERS_PRIORI = {
        'incera_arm': 0.537306064880113,
        'incera_col': 0.483145275035261,
        'incera_bowl': 1.03032440056417,
        'clp_strain': 0.553314527503526,
        'hammer': 0.148236953455571,
        'incomp_arm': 0.146262341325811,
        'incomp_bar': 0.230535966149506,
        'incomp_col': 0.858956276445698,
    }
    MAPPER_ZN = {
        'incera_disk': '陶瓷碟式绝缘子',
        'incera_arm': '陶瓷横担绝缘子',
        'incera_bar': '陶瓷拉棒绝缘子',
        'incera_col': '陶瓷立柱绝缘子',
        'incera_bowl': '陶瓷盘式绝缘子',
        'incera_cas': '陶瓷套管',
        'incera_pin': '陶瓷针式绝缘子',
        'incera_pilr': '陶瓷支柱绝缘子',
        'clp_strain': '螺栓耐张线夹',
        'clp_wedge': '楔形耐张线夹',
        'hammer': '防震锤',
        'incomp_arm': '复合横担绝缘子',
        'incomp_bar': '复合拉棒绝缘子',
        'incomp_col': '复合立柱避雷器',
        'incomp_pin': '复合针式绝缘子',
        'incomp_pilr': '复合支柱绝缘子',
    }
    CLASS_NAMES_TST = tuple(sorted(CLUSTERS.keys()))
    CLASS_NAMES = tuple(sorted(chain(*CLUSTERS.values())))

    def __init__(self, root=None, img_folder: str = IMG_FOLDER, label_folder: str = LABEL_FOLDER, img_extend=IMG_EXTEND,
                 label_extend=LABEL_EXTEND, cls_names=CLASS_NAMES, set_names=None,
                 task_type=TASK_TYPE.AUTO, data_mode=DATA_MODE.FULL, **kwargs):
        InpPKLDataSource.__init__(self, root=root, set_names=set_names, img_folder=img_folder, task_type=task_type,
                                  label_folder=label_folder, img_extend=img_extend, label_extend=label_extend,
                                  cls_names=cls_names, data_mode=data_mode)
        self.CLASS_NAMES_ZN = tuple(self.MAPPER_ZN[n.split('-')[0]] + '-' + n.split('-')[1] for n in cls_names)
        self.CLASS_NAMES_TST_ZN = tuple(self.MAPPER_ZN[n] + '-1' for n in self.CLASS_NAMES_TST)
        self.cind2name_tst = lambda cind: self.CLASS_NAMES_TST[cind]
        # 计算映射
        _tmp = [[(nj, i) for nj in self.CLUSTERS[n]] for i, n in enumerate(self.CLASS_NAMES_TST)]
        _mapper = dict(chain(*_tmp))
        self.CLUSTER_INDEX = np.array(tuple(_mapper[n] for n in cls_names))

        # 计算数量先验
        self.NUMS_PRIORI_TST = np.array([self.CLUSTERS_PRIORI[n] for n in self.CLASS_NAMES_TST])
        _divs = [len(self.CLUSTERS[n]) for n in self.CLASS_NAMES_TST]
        self.NUMS_PRIORI = (self.NUMS_PRIORI_TST / _divs)[self.CLUSTER_INDEX]

        self.wobj_dir = os.path.join(self.root, self.WOBJ_FOLDER)
        self.info_pth = os.path.join(self.root, self.WOBJ_FOLDER, self.INFO_NAME)


# if __name__ == '__main__':
#     ds = Resh3DDataSource(
#         root='/ses-data/JD/Resh3DX',
#         # img_folder='images_rnd',
#         img_folder='JPEGImages',
#         data_mode=DATA_MODE.FULL, )
#     dataset = ds.dataset('all')
#     # split_dict = {'all': 1.0,}
#     split_dict = {'example': 0.0001, }
#     # split_dict = {'train': 0.18, 'test': 0.07, 'other': 0.75}
#     dataset.partition_set_(split_dict)

if __name__ == '__main__':
    for i, n in enumerate(Resh3DDataSource.CLASS_NAMES):
        print(i, n)
# if __name__ == '__main__':
#     ds = Resh3DDataSource(root='/ses-data/JD/Resh3D',
#                           label_folder='labels_ori',
#                           img_folder='images_ori',
#                           data_mode=DATA_MODE.FULL, )
#     dataset = ds.dataset('train')
#     labels = dataset.labels
#     items = chain(*labels)
#
#     measures = np.array([item.measure for item in items])
#     show_distribute(measures)
#     plt.pause(1e5)
# print(len(measures))
# print(np.mean(measures))
