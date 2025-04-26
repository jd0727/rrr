from datas.base import *
from datas.voc import pretty_xml_node, ET, VOCDetectionDataset
from utils import *


class _FNAME_META_OPT_XYXY:
    @staticmethod
    def enc(meta: str, opt: str, index: int = 0, cls_name: str = 'test', xyxy: Iterable = (3, 3, 8, 8),
            cls_name_new: str = 'new', xyxy_new: Iterable = (2, 2, 7, 7)) -> str:
        def _xyxy2str(xyxy: Iterable) -> str:
            return '(' + '_'.join(['%04d' % int(v) for v in xyxy]) + ')'

        if opt == 'F':
            msg = '_'.join(
                [meta, opt, '%02d' % index, _xyxy2str(xyxy), '[%s]' % cls_name]) + '=' + '[%s]' % cls_name_new
        elif opt == 'A':
            msg = '_'.join([meta, opt, '%02d' % index, _xyxy2str(xyxy), '[%s]' % cls_name]) + '=' + _xyxy2str(xyxy_new)
        elif opt == 'D':
            msg = '_'.join([meta, opt, '%02d' % index, _xyxy2str(xyxy), '[%s]' % cls_name])
        elif opt == 'N':
            msg = '_'.join([meta, opt, '%02d' % index, _xyxy2str(xyxy_new), '[%s]' % cls_name_new])
        else:
            raise Exception('opt fmt err')
        return msg

    @staticmethod
    def dec(meta_encd: str) -> (str, str, dict):
        def _prase_cls(msg):
            cls_name = re.findall(r'\[.*\]', msg)
            assert len(cls_name) == 1
            cls_name = cls_name[0].replace('[', '').replace(']', '')
            return cls_name

        def _prase_xyxy(msg):
            xyxy = re.findall(r'\(.*\)', msg)
            assert len(xyxy) == 1
            xyxy = [int(v) for v in xyxy[0].replace('(', '').replace(')', '').split('_')]
            return xyxy

        def _prase_itm(msg):
            xyxy = re.findall(r'\(.*\)', msg)
            assert len(xyxy) == 1
            index, cls_name = msg.split(xyxy[0])
            index = int(index.replace('_', ''))
            xyxy = _prase_xyxy(xyxy[0])
            cls_name = _prase_cls(cls_name)
            return index, xyxy, cls_name

        opts = re.findall(r'_[ADNF]_', meta_encd)
        assert len(opts) == 1, 'opt err ' + meta_encd
        opt = opts[0].replace('_', '')
        meta, other = meta_encd.split(opts[0])
        if opt == 'A':
            assert len(re.findall(r'=', other)) == 1, 'a err'
            itm_old, itm_new = other.split('=')
            index, xyxy, cls_name = _prase_itm(itm_old)
            para = dict(index=index, xyxy=xyxy, cls_name=cls_name, xyxy_new=_prase_xyxy(itm_new))
        elif opt == 'F':
            assert len(re.findall(r'=', other)) == 1, 'a err'
            itm_old, itm_new = other.split('=')
            index, xyxy, cls_name = _prase_itm(itm_old)
            para = dict(index=index, xyxy=xyxy, cls_name=cls_name, cls_name_new=_prase_cls(itm_new))
        elif opt == 'D':
            index, xyxy, cls_name = _prase_itm(other)
            para = dict(index=index, xyxy=xyxy, cls_name=cls_name)
        else:
            xyxy = _prase_xyxy(other)
            cls_name = _prase_cls(other)
            para = dict(xyxy_new=xyxy, cls_name_new=cls_name)
        return meta, opt, para


# C 正确匹配
# A 边界调整
# F 分类调整
# D 未检测
# N 新目标
def match_boxes(boxes_ds, boxes_md, iou_thres: float = 0.5, iou_ignore: float = 0.2,
                use_cind: bool = False,
                mtch_types: str = 'AFDN'):
    with_correct = 'C' in mtch_types
    with_false = 'F' in mtch_types
    with_adjust = 'A' in mtch_types
    with_delete = 'D' in mtch_types
    with_new = 'N' in mtch_types
    # 排序
    if len(boxes_md) > 0:
        boxes_md.orderby_conf_(ascend=False)

    xyxys_md = boxes_md.export_xyxysN()
    matched_md = [False] * len(boxes_md)
    xyxys_lb = boxes_ds.export_xyxysN()
    matched_lb = [False] * len(boxes_ds)

    # 匹配
    iou_mat = xyxyN_ropr(xyxys_md[:, None, :], xyxys_lb[None, :, :], opr_type=OPR_TYPE.IOU)
    bxs_md_new = []
    for ind_md, bx_md in enumerate(boxes_md):
        iou_max = np.max(iou_mat[ind_md, :]) if len(boxes_ds) > 0 else 0
        if iou_max < iou_ignore:
            if with_new:
                bxs_md_new.append(bx_md)
            matched_md[ind_md] = True

    pairs_correct = []
    pairs_false = []
    for ind_md, bx_md in enumerate(boxes_md):
        if matched_md[ind_md]:
            continue
        iou_max = np.max(iou_mat[ind_md, :]) if len(boxes_ds) > 0 else 0
        ind_lb = np.argmax(iou_mat[ind_md, :]) if len(boxes_ds) > 0 else -1
        if iou_max > iou_thres:
            bx_ds = boxes_ds[ind_lb]
            if use_cind and bx_ds.category.cindN == bx_md.category.cindN:
                if with_correct:
                    pairs_correct.append((bx_md, bx_ds, ind_lb))
            elif (not use_cind) and bx_ds.get('name', '') == bx_md.get('name', ''):
                if with_correct:
                    pairs_correct.append((bx_md, bx_ds, ind_lb))
            else:
                if with_false:
                    pairs_false.append((bx_md, bx_ds, ind_lb))
            matched_md[ind_md] = True
            matched_lb[ind_lb] = True
            iou_mat[:, ind_lb] = 0
            iou_mat[ind_md, :] = 0
    pairs_adjust = []
    bxs_ds_delete = []
    for ind_lb, bx_lb in enumerate(boxes_ds):
        if matched_lb[ind_lb]:
            continue
        iou_max = np.max(iou_mat[:, ind_lb]) if len(boxes_md) > 0 else 0
        ind_md = np.argmax(iou_mat[:, ind_lb]) if len(boxes_md) > 0 else -1
        if iou_max > iou_ignore:
            if with_adjust:
                pairs_adjust.append((boxes_md[ind_md], bx_lb, ind_lb))
            matched_md[ind_md] = True
            matched_lb[ind_lb] = True
            iou_mat[:, ind_lb] = 0
            iou_mat[ind_md, :] = 0
        else:
            if with_delete:
                bxs_ds_delete.append((bx_lb, ind_lb))
    return pairs_correct, pairs_false, pairs_adjust, bxs_md_new, bxs_ds_delete


def xyxyN_expand_with_size(xyxyN: np.ndarray, size: tuple, ratio: float = 1.6) -> np.ndarray:
    xywh = xyxyN2xywhN(xyxyN)
    xywh[2:4] = np.clip(xywh[2:4] * ratio, a_min=4, a_max=None)
    xyxyN_rec = xyxyN_clip(xywhN2xyxyN(xywh), xyxyN_rgn=np.array(size))
    return xyxyN_rec


def _imgP_draw_box(imgP: Image.Image, xyxyN: np.ndarray, color: tuple = (255, 0, 0, 0),
                   line_width: Union[int, float] = 8) -> Image.Image:
    xyp = xyxyN2xypN(xyxyN)
    draw = PIL.ImageDraw.ImageDraw(imgP)
    scale = np.sqrt(np.prod(imgP.size))
    line_width = ps_int_multiply(line_width, scale)
    if line_width>0:
        draw.line(list(xyp.reshape(-1)) + list(xyp[0]), fill=color, width=line_width)
    return imgP


def _imgP_crop_draw_2(imgP: Image.Image, xyxyN1: np.ndarray, xyxyN2: np.ndarray, color1: tuple = (255, 0, 0, 0),
                      line_width1: Union[int, float] = 8, color2: tuple = (255, 0, 0, 0),
                      line_width2: Union[int, float] = 8, ratio: float = 1.6, ) -> Image.Image:
    xyxy_union = np.concatenate([np.minimum(xyxyN1[:2], xyxyN2[:2]), np.maximum(xyxyN1[2:], xyxyN2[2:])], axis=0)
    xyxy_exd = xyxyN_expand_with_size(xyxy_union, size=imgP.size, ratio=ratio)
    patch = imgP.crop(tuple(xyxy_exd.astype(np.int32)))
    for xyxy, col, lw in zip([xyxyN1, xyxyN2], [color1, color2], [line_width1, line_width2]):
        xyxy_ref = copy.deepcopy(xyxy) - xyxy_exd[[0, 1, 0, 1]]
        _imgP_draw_box(patch, xyxyN=xyxy_ref, color=col, line_width=lw)
    return patch


def _imgP_crop_draw(imgP: Image.Image, xyxyN: np.ndarray, ratio: float = 1.6, color: tuple = (255, 0, 0, 0),
                    line_width: Union[int, float] = 8):
    xyxy_exd = xyxyN_expand_with_size(xyxyN, size=imgP.size, ratio=ratio)
    patch = imgP.crop(tuple(xyxy_exd.astype(np.int32)))
    xyxy_ref = copy.deepcopy(xyxyN) - xyxy_exd[[0, 1, 0, 1]]
    _imgP_draw_box(patch, xyxyN=xyxy_ref, color=color, line_width=line_width)
    return patch


def process_match(img: Union[Image.Image, np.ndarray], meta: str, dst_dir, pairs_correct, pairs_false, pairs_adjust,
                  bxs_md_new,
                  bxs_ds_delete, ratio: float = 2.5, line_width: Union[int, float] = 8, color_md=(0, 0, 255, 0),
                  color_ds=(255, 0, 0, 0), img_extend: str = 'jpg', thres: float = 20):
    imgP = img2imgP(img)
    for bx_md, bx_ds, ind_ds in pairs_correct:
        bdr_lb = XYXYBorder.convert(bx_ds.border)
        if bdr_lb.measure < thres:
            continue
        patch = _imgP_crop_draw(imgP, bdr_lb._xyxyN, ratio=ratio, color=color_ds, line_width=line_width)
        file_name = _FNAME_META_OPT_XYXY.enc(
            meta, opt='F', index=ind_ds, xyxy=bdr_lb._xyxyN, cls_name=bx_ds['name'], cls_name_new=bx_md['name'], )
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'C_' + bx_md['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))

    for bx_md, bx_ds, ind_ds in pairs_false:
        bdr_lb = XYXYBorder.convert(bx_ds.border)
        if bdr_lb.measure < thres:
            continue
        patch = _imgP_crop_draw(imgP, bdr_lb._xyxyN, ratio=ratio, color=color_ds, line_width=line_width)
        file_name = _FNAME_META_OPT_XYXY.enc(
            meta, opt='F', index=ind_ds, xyxy=bdr_lb._xyxyN, cls_name=bx_ds['name'], cls_name_new=bx_md['name'], )
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'F_' + bx_md['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))

    for bx_md, bx_ds, ind_ds in pairs_adjust:
        bdr_lb = XYXYBorder.convert(bx_ds.border)
        bdr_md = XYXYBorder.convert(bx_md.border)
        if bdr_lb.measure < thres or bdr_md.measure < thres:
            continue
        patch = _imgP_crop_draw_2(
            imgP, xyxyN1=bdr_lb._xyxyN, xyxyN2=bdr_md._xyxyN, color2=color_md, line_width2=line_width,
            ratio=ratio, color1=color_ds, line_width1=line_width)
        file_name = _FNAME_META_OPT_XYXY.enc(
            meta, opt='A', index=ind_ds, xyxy=bdr_lb._xyxyN, cls_name=bx_ds['name'], xyxy_new=bdr_md._xyxyN, )
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'A_' + bx_ds['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))

    for bx_md in bxs_md_new:
        bdr_md = XYXYBorder.convert(bx_md.border)
        if bdr_md.measure < thres:
            continue
        patch = _imgP_crop_draw(imgP, bdr_md._xyxyN, ratio=ratio, color=color_md, line_width=line_width)
        file_name = _FNAME_META_OPT_XYXY.enc(meta, opt='N', xyxy_new=bdr_md._xyxyN, cls_name_new=bx_md['name'])
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'N_' + bx_md['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))

    for bx_ds, ind_ds in bxs_ds_delete:
        bdr_lb = XYXYBorder.convert(bx_ds.border)
        if bdr_lb.measure < thres:
            continue
        patch = _imgP_crop_draw(imgP, bdr_lb._xyxyN, ratio=ratio, color=color_ds, line_width=line_width)
        file_name = _FNAME_META_OPT_XYXY.enc(meta, opt='D', index=ind_ds, xyxy=bdr_lb._xyxyN, cls_name=bx_ds['name'])
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'D_' + bx_ds['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))
    return True


def imprv_by_match(dst_dir, img, boxes_ds, boxes_md, iou_thres: float = 0.5, iou_ignore: float = 0.2,
                   use_cind: bool = False,
                   ratio: float = 2.5, line_width: Union[int, float] = 8, color_md=(0, 0, 255, 0),
                   color_ds=(255, 0, 0, 0), img_extend: str = 'jpg', thres: float = 20, mtch_types: str = 'AFDN'):
    pairs_correct, pairs_false, pairs_adjust, bxs_md_new, bxs_ds_delete = \
        match_boxes(boxes_ds, boxes_md, iou_thres=iou_thres, iou_ignore=iou_ignore, use_cind=use_cind,
                    mtch_types=mtch_types)
    if pairs_correct or pairs_false or pairs_adjust or bxs_md_new or bxs_ds_delete:
        if isinstance(img, str):
            img = load_img_pil(img)
        process_match(img, meta=boxes_ds.meta, dst_dir=dst_dir, pairs_correct=pairs_correct, pairs_false=pairs_false,
                      pairs_adjust=pairs_adjust, bxs_md_new=bxs_md_new, bxs_ds_delete=bxs_ds_delete, ratio=ratio,
                      line_width=line_width, color_md=color_md, color_ds=color_ds, img_extend=img_extend, thres=thres)
        return True
    else:
        return False


class SELECT_TYPE:
    INDEX = 'index'
    BORDER = 'index'
    BORDER_CLS = 'border_cls'


def _voc_match_index(objs, para, select_type=SELECT_TYPE.INDEX):
    if select_type == SELECT_TYPE.INDEX:
        return para['index']
    else:
        xyxy = para['xyxy']
        name = para['cls_name']
        for i, obj in enumerate(objs):
            name_obj = obj.find('name').text
            bndbox = obj.find('bndbox')
            xyxy_obj = [int(float(bndbox.find(tag).text)) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]
            bdr_mtchd = all([abs(v - v_obj) <= 1 for v, v_obj in zip(xyxy, xyxy_obj)])
            if not bdr_mtchd:
                continue
            if select_type == SELECT_TYPE.BORDER:
                return i
            if not name == name_obj:
                continue
            if select_type == SELECT_TYPE.BORDER_CLS:
                return i
    return -1


def collect_imprv(imprv_dir, folder_override=True):
    recorder = {}
    for cur_dir, folder_names, file_names in os.walk(imprv_dir):
        cur_folder = os.path.basename(cur_dir)
        result = re.findall(r'[DFANP]_', cur_folder)
        if len(result) == 1:
            opt_fdr = result[0].replace('_', '')
            cls_name_fdr = cur_folder.replace(result[0], '')
        else:
            opt_fdr, cls_name_fdr = None, None
        for file_name in file_names:
            file_name_pure = os.path.splitext(file_name)[0]
            meta, opt, para = _FNAME_META_OPT_XYXY.dec(file_name_pure)
            if folder_override and opt_fdr is not None:
                assert opt_fdr == opt or opt_fdr == 'P' or not opt_fdr == 'A', 'ferr ' + file_name
                assert opt_fdr == opt or opt_fdr == 'P' or not opt == 'N', 'ferr ' + file_name
                opt = opt_fdr
                para['cls_name_new'] = cls_name_fdr
            if meta not in recorder.keys():
                recorder[meta] = []
            rec_lst = recorder[meta]
            rec_lst.append([opt, para])
    return recorder


def voc_accept_imprv(imprv_dir, anno_dir, anno_dir_new, prob_dir, img_dir,
                     select_type=SELECT_TYPE.INDEX, folder_override=True):
    ensure_folder_pth(anno_dir_new)

    # 统计修改
    recorder = collect_imprv(imprv_dir, folder_override=folder_override)
    print('Collected edit item ', sum([len(v) for v in recorder.values()]))
    num_succ = 0
    num_prob = 0
    for i, anno_name in MEnumerate(sorted(os.listdir(anno_dir)), step=10):
        anno_pth = os.path.join(anno_dir, anno_name)
        anno_pth_dst = os.path.join(anno_dir_new, anno_name)
        meta = anno_name.split('.')[0]
        if meta not in recorder.keys():
            if not anno_pth == anno_pth_dst:
                shutil.copy(anno_pth, anno_pth_dst)
            continue
        rec_lst = recorder[meta]
        # 读取修改
        root = ET.parse(anno_pth).getroot()
        w = int(root.find('size').find('width').text)
        h = int(root.find('size').find('height').text)
        objs = root.findall('object')

        for _, para in filter(lambda itm: itm[0] == 'F', rec_lst):
            ind = _voc_match_index(objs, para, select_type=select_type)
            if ind >= 0:
                obj = objs[ind]
                obj.find('name').text = para['cls_name_new']
                num_succ += 1
            else:
                print('no match', para)

        for _, para in filter(lambda itm: itm[0] == 'A', rec_lst):
            ind = _voc_match_index(objs, para, select_type=select_type)
            if ind >= 0:
                bndbox = objs[ind].find('bndbox')
                xyxy = para.get('xyxy_new', None) or para.get('xyxy', None)
                xyxy[0:4:2] = np.clip(xyxy[0:4:2], a_min=0, a_max=w)
                xyxy[1:4:2] = np.clip(xyxy[1:4:2], a_min=0, a_max=h)
                for k, tag in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                    bndbox.find(tag).text = str(xyxy[k])
                num_succ += 1
            else:
                print('no match', para)

        for _, para in filter(lambda itm: itm[0] == 'N', rec_lst):
            xyxy = para.get('xyxy_new', None) or para.get('xyxy', None)
            cls_name = para['cls_name_new']
            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = cls_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'
            bndbox = ET.SubElement(obj, 'bndbox')
            xyxy[0:4:2] = np.clip(xyxy[0:4:2], a_min=0, a_max=w)
            xyxy[1:4:2] = np.clip(xyxy[1:4:2], a_min=0, a_max=h)
            for k, tag in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                ET.SubElement(bndbox, tag).text = str(xyxy[k])
            num_succ += 1

        rec_D = list(filter(lambda itm: itm[0] == 'D', rec_lst))
        rec_D = sorted(rec_D, key=lambda x: -x[1]['index'])  # 索引从大到小排序
        for _, para in rec_D:
            ind = _voc_match_index(objs, para, select_type=select_type)
            if ind >= 0 and objs[ind] in root:
                root.remove(objs[ind])
                num_succ += 1
            else:
                print('no match', para)

        # 问题过滤
        rec_P = list(filter(lambda itm: itm[0] == 'P', rec_lst))

        pretty_xml_node(root)
        root_dst = ET.ElementTree(root)
        root_dst.write(anno_pth_dst, encoding='utf-8')

        if len(rec_P) > 0:
            ensure_folder_pth(prob_dir)
            shutil.copy(anno_pth_dst, os.path.join(prob_dir, anno_name))
            shutil.copy(os.path.join(img_dir, meta + '.jpg'), os.path.join(prob_dir, meta + '.jpg'))
            num_prob += 1
    print('Complete edit item ', num_succ)
    if num_prob > 0:
        print('Prob num ', num_prob, ' at ', prob_dir)
    return None


def check_label_repeat(label, cluster_index=CLUSTER_INDEX.NONE, iou_thres: float = 0.6, select_larger: bool = True,
                       opr_type=OPR_TYPE.IOU):
    xyxys = label.export_xyxysN()
    cinds = label.export_cindsN()
    areas = xyxyN2areaN(xyxys)
    pairs_delete = []
    if len(cinds) == 0:
        return pairs_delete
    if cluster_index is None or cluster_index == CLUSTER_INDEX.NONE:
        cinds = np.zeros_like(cinds)
        num_cls = 1
    elif cluster_index == CLUSTER_INDEX.CLASS:
        num_cls = np.max(cinds) + 1
    else:
        cinds = np.array(cluster_index)[cinds]
        num_cls = np.max(cinds) + 1
    for cind in range(num_cls):
        mask_cls = cinds == cind
        inds_cls = np.nonzero(mask_cls)[0]
        xyxys_cls = xyxys[mask_cls]
        ious = xyxyN_ropr(xyxys_cls[:, None, :], xyxys_cls[None, :, :], opr_type=opr_type)
        idxs1, idxs2 = np.nonzero(ious - np.eye(len(inds_cls)) > iou_thres)
        # 去重复
        idxs12 = np.stack([idxs1, idxs2], axis=1)
        idxs12 = np.sort(idxs12, axis=1)
        idxs12 = np.unique(idxs12, axis=0)
        idxs1, idxs2 = idxs12[:, 0], idxs12[:, 1]
        # 匹配
        for idx1, idx2 in zip(inds_cls[idxs1], inds_cls[idxs2]):
            if select_larger == (areas[idx1] > areas[idx2]):
                pairs_delete.append((label[idx1], idx1, label[idx2]))
            else:
                pairs_delete.append((label[idx2], idx2, label[idx1]))
    return pairs_delete


def voc_delete_byfunc(anno_pth, img_pth, dst_dir,
                      name2cind=None, item_fltr: Optional[Callable] = None,
                      ratio: float = 2.5, line_width: Union[int, float] = 8,
                      color_del=(255, 0, 0, 0), img_extend='jpg'):
    label = VOCDetectionDataset.load_anno(anno_pth, name2cind=name2cind)
    if item_fltr is None:
        return False
    items_delete = [(i, item) for i, item in enumerate(label) if item_fltr(item)]
    if len(items_delete) == 0:
        return False
    imgP = load_img_pil(img_pth)
    for ind_del, item_del in items_delete:
        xyxy_del = XYXYBorder.convert(item_del.border)._xyxyN
        patch = _imgP_crop_draw(
            imgP, xyxyN=xyxy_del, color=color_del, line_width=line_width, ratio=ratio)

        file_name = _FNAME_META_OPT_XYXY.enc(
            label.meta, opt='D', index=ind_del, xyxy=xyxy_del, cls_name=item_del['name'], )
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'D_' + item_del['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))
    return True


def voc_delete_overlap(anno_pth, img_pth, dst_dir, cluster_index=CLUSTER_INDEX.NONE, opr_type=OPR_TYPE.IOU,
                       name2cind=None, iou_thres=0.6, select_larger=True,
                       ratio: float = 2.5, line_width: Union[int, float] = 8, color_ref=(0, 0, 255, 0),
                       color_del=(255, 0, 0, 0), img_extend='jpg'):
    label = VOCDetectionDataset.load_anno(anno_pth, name2cind=name2cind)
    pairs_delete = check_label_repeat(label, cluster_index=cluster_index, iou_thres=iou_thres,
                                      select_larger=select_larger, opr_type=opr_type)
    if len(pairs_delete) == 0:
        return False
    imgP = load_img_pil(img_pth)

    for item_del, ind_del, item_ref in pairs_delete:
        xyxy_del = XYXYBorder.convert(item_del.border)._xyxyN
        xyxy_ref = XYXYBorder.convert(item_ref.border)._xyxyN

        patch = _imgP_crop_draw_2(
            imgP, xyxyN1=xyxy_del, color1=color_del, xyxyN2=xyxy_ref, color2=color_ref, line_width2=line_width,
            ratio=ratio, line_width1=line_width)

        file_name = _FNAME_META_OPT_XYXY.enc(
            label.meta, opt='D', index=ind_del, xyxy=xyxy_del, cls_name=item_del['name'], )
        sub_dir = ensure_folder_pth(os.path.join(dst_dir, 'D_' + item_del['name']))
        patch.save(os.path.join(sub_dir, file_name + '.' + img_extend))
    return True


def voc_cut_pbox(anno_pth: str, img_pth: str, dst_dir: str, ratio: float = 2.5, line_width: Union[int, float] = 8,
                 color=(0, 0, 255, 0), img_extend: str = 'jpg', fltr: Optional[Callable] = None,
                 func: Optional[Callable] = None):
    label = VOCDetectionDataset.load_anno(anno_pth)
    label.filt_(fltr)
    if len(label) == 0:
        return True
    imgP = load_img_pil(img_pth)
    for j, box in enumerate(label):
        box_name = box['name'].lower()
        save_dir = os.path.join(dst_dir, 'F_' + box_name)
        ensure_folder_pth(save_dir)

        xyxyN = XYXYBorder.convert(box.border)._xyxyN.astype(np.int32)
        patch = _imgP_crop_draw(imgP=imgP, xyxyN=xyxyN, ratio=ratio, color=color, line_width=line_width)
        file_name = _FNAME_META_OPT_XYXY.enc(
            label.meta, opt='F', index=j, xyxy=xyxyN, cls_name=box_name, cls_name_new=box_name)
        ptch_pth = os.path.join(save_dir, file_name + '.' + img_extend)
        if func is not None:
            patch = func(patch)
        img2imgP(patch).save(ptch_pth, quality=100)

    return True
