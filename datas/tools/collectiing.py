import os.path
import shutil

import numpy as np

from utils import *


# <editor-fold desc='图像文件夹去除重复'>

def _load_img_vec(img_pth: str, pool_size: TV_Int2 = (8, 8)) -> np.ndarray:
    pool_size = ps_int2_repeat(pool_size)
    img = load_img_pil(img_pth)
    img_pld = img.resize(pool_size)
    img_pld = np.array(img_pld).astype(np.int32)
    return img_pld.reshape(-1)


def _load_img_vecs_pths(img_dir: str, extend: str = 'jpg', pool_size: TV_Int2 = (8, 8),
                        cache_pth: Optional[str] = 'img_vec_cache.pkl') -> (np.ndarray, np.ndarray, np.ndarray):
    file_pths = []
    file_sizes = []
    vecs = []
    if cache_pth is not None and os.path.exists(cache_pth):
        print('Using existing cache at', cache_pth)
        arrs_ext, file_pths_ext, file_sizes_ext = load_pkl(cache_pth)
        ext_mapper = dict([(fpth, (arr, fsize)) for arr, fpth, fsize in zip(arrs_ext, file_pths_ext, file_sizes_ext)])
    else:
        print('No cache at', cache_pth)
        ext_mapper = {}

    for i, file_name in MEnumerate(sorted(os.listdir(img_dir)), step=10):
        if not file_name.endswith(extend):
            continue
        file_pth = os.path.join(img_dir, file_name)
        if file_pth in ext_mapper.keys():
            vec, file_size = ext_mapper[file_pth]
        else:
            vec = _load_img_vec(file_pth, pool_size=pool_size)
            file_size = os.path.getsize(file_pth)
        file_sizes.append(file_size)
        vecs.append(vec)
        file_pths.append(file_pth)
    vecs = np.stack(vecs, axis=0)
    file_sizes = np.array(file_sizes)
    file_pths = np.array(file_pths)

    if cache_pth is not None:
        print('Create cache at ', cache_pth)
        ensure_file_dir(cache_pth)
        save_pkl(cache_pth, (vecs, file_pths, file_sizes))
    return vecs, file_pths, file_sizes


# class RMREPEAT_PROC:
#     SRC_COPY='SRC'
#     SRC_MOVE='SRC'
#     SRC_COPY_REF_COPY='SRC'
#     SRC_COPY='SRC'
#     SRC_COPY='SRC'
def _proced_pairs(file_pths_ref: List[str], file_pths_src: List[str], dst_dir: str):
    ensure_folder_pth(dst_dir)
    for _, (file_pth_src, file_pth_ref) in MEnumerate(list(zip(file_pths_src, file_pths_ref))):
        if os.path.exists(file_pth_src):
            shutil.move(file_pth_src, os.path.join(dst_dir, os.path.basename(file_pth_src)))
        # if os.path.exists(file_pth_ref):
        #     shutil.copy(file_pth_ref, os.path.join(dst_dir, os.path.basename(file_pth_ref)))
    return True


def img_dir_rmrepeat_inner(src_dir: str, dst_dir: str, pool_size: TV_Int2 = (8, 8), sim_thres: float = 15,
                           cache_pth: Optional[str] = 'img_vec_cache.pkl'):
    print('Load imgs at ', src_dir)
    vecs, file_pths, file_sizes = _load_img_vecs_pths(src_dir, pool_size=pool_size, cache_pth=cache_pth)
    print('Start comparing')
    order = np.argsort(-file_sizes, kind='stable')
    vecs = vecs[order]
    file_pths = file_pths[order]
    fpslds_small = []
    fpslds_large = []
    vecs = vecs.astype(np.int16)
    for idx_small, vec in MEnumerate(vecs, step=10):
        sim = np.mean(np.abs(vec - vecs), axis=-1)
        idxs_large = np.nonzero(sim < sim_thres)[0]
        for idx_large in idxs_large[idxs_large > idx_small]:
            fpslds_small.append(file_pths[idx_small])
            fpslds_large.append(file_pths[idx_large])

    print('Start moving')
    _proced_pairs(file_pths_ref=fpslds_small, file_pths_src=fpslds_large, dst_dir=dst_dir)
    return True


def img_dir_rmrepeat_inter(ref_dir: str, src_dir: str, dst_dir: str, pool_size: TV_Int2 = (8, 8), sim_thres: float = 15,
                           cache_pth_ref: Optional[str] = 'img_vec_cache1.pkl',
                           cache_pth_src: Optional[str] = 'img_vec_cache2.pkl'):
    ensure_folder_pth(dst_dir)
    print('Load imgs at ', ref_dir)
    vecs_ref, file_pths_ref, _ = _load_img_vecs_pths(ref_dir, pool_size=pool_size, cache_pth=cache_pth_ref)
    print('Load imgs at ', src_dir)
    vecs_src, file_pths_src, _ = _load_img_vecs_pths(src_dir, pool_size=pool_size, cache_pth=cache_pth_src)
    print('Start comparing')
    fpslds_src = []
    fpslds_ref = []
    vecs_src = vecs_src.astype(np.int16)
    vecs_ref = vecs_ref.astype(np.int16)
    for idx_src, arr_src in MEnumerate(vecs_src, step=10):
        sim = np.mean(np.abs(arr_src - vecs_ref), axis=-1)
        idxs_ref = np.nonzero(sim < sim_thres)[0]
        for idx_ref in idxs_ref:
            fpslds_src.append(file_pths_src[idx_src])
            fpslds_ref.append(file_pths_ref[idx_ref])
    print('Start moving')
    _proced_pairs(file_pths_ref=fpslds_ref, file_pths_src=fpslds_src, dst_dir=dst_dir)
    return True


# </editor-fold>

# 对齐ref_dir文件和src_dir文件
# 将src_dir多余的文件剪切到dst_dir

def dir_align(ref_dir: str, src_dir: str, dst_dir: str, only_show: bool = True):
    if only_show:
        print('Only showing')
    else:
        print('Start align')
    ref_metas = [os.path.splitext(sn)[0] for sn in os.listdir(ref_dir)]
    cnt = 0
    for file_name in os.listdir(src_dir):
        meta = os.path.splitext(file_name)[0]
        if meta not in ref_metas:
            file_pth = os.path.join(src_dir, file_name)
            file_pth_new = os.path.join(dst_dir, file_name)
            if only_show:
                print(file_pth, '->', file_pth_new)
            else:
                ensure_folder_pth(dst_dir)
                shutil.move(file_pth, file_pth_new)
            cnt += 1
    print('Align num', cnt)
    return True


# <editor-fold desc='图像文件收集并编号'>
_IMG_EXTEDNS = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG')


def img_dir_gather_recode(src_dir: str, dst_dir: str, recursive: bool,
                          extends: Optional[Tuple[str, ...]] = _IMG_EXTEDNS,
                          code_fmt: str = '%06d', code_start: int = 0, recd_pth: Optional[str] = 'name.txt',
                          only_show: bool = True):
    if only_show:
        print('Only showing')
    else:
        print('Start gathering')
        ensure_folder_pth(dst_dir)
    file_pths = listdir_recursive(src_dir, recursive=recursive, extends=extends)
    code = code_start
    lines = []

    for i, file_pth in MEnumerate(file_pths):
        file_name_pure = file_pth.replace(src_dir, '').replace(os.sep, '_').strip('_')
        file_name_pure, extend_cur = os.path.splitext(file_name_pure)
        meta = code_fmt % code
        file_pth_new = os.path.join(dst_dir, ensure_extend(meta, extend_cur.lower()))
        code = code + 1
        lines.append(meta + '\t' + file_name_pure)
        if only_show:
            print(file_pth, ' -> ', file_pth_new)
        else:
            shutil.copy(file_pth, file_pth_new)
    print('Gathered from ', code_fmt % code_start, ' to ', code_fmt % code)
    if recd_pth is not None:
        print('Name save at ', recd_pth)
        ensure_file_dir(recd_pth)
        save_txt(recd_pth, lines, append_mode=True)
    return code


# </editor-fold>


# <editor-fold desc='图像文件夹修复破损'>
def reload_imgs_cv2(img_dir: str, extends: Optional[Tuple[str, ...]] = _IMG_EXTEDNS,
                    reload_all: bool = True):
    img_names = listdir_extend(img_dir, extends)
    for i, img_name in MEnumerate(sorted(img_names)):
        img_pth = os.path.join(img_dir, img_name)
        img_name_pure = os.path.splitext(img_name)[0]
        img_name_pure_new = img_name_pure.replace('.', '_').replace('、', '_').replace('，', '_')
        img_pth_new = os.path.join(img_dir, img_name_pure_new + '.jpg')

        if reload_all or (not img_pth == img_pth_new):
            img = reload_img_cv2(img_pth, img_pth_new)
            print(img_pth, ' -> ', img_pth_new)
            if img is None and os.path.exists(img_pth):
                print('File err, remove', img_pth)
                os.remove(img_pth)
        if not img_pth == img_pth_new and os.path.exists(img_pth):
            os.remove(img_pth)

    return True

# </editor-fold>
