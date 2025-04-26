

try:
    from datas.coco import COCOeval
except Exception as e:
    pass

from datas.coco import COCODataset
from utils import *


def _summarize_coco_eval(coco_eval, cind2name: Optional[Callable] = None):
    def _summarize_coco(coco_eval, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        s = coco_eval.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        nums = np.sum(s > -1, axis=(0, 1, 3))
        sum_vals = np.sum(np.where(s == -1, np.zeros_like(s), s), axis=(0, 1, 3))
        mean_vals = np.where(nums > 0, sum_vals / nums, np.full_like(sum_vals, fill_value=0))
        total_val = np.mean(s[s > -1]) if np.any(nums > 0) else 0
        pkd_vals = np.concatenate([mean_vals, [total_val]], axis=0)
        return pkd_vals

    data = pd.DataFrame()
    cls_names = [cind2name(i) if cind2name else str(i) for i in coco_eval.params.catIds]
    data['Class'] = cls_names + ['Total']
    data['AP'] = _summarize_coco(coco_eval, iouThr=None, areaRng='all', maxDets=100)
    data['AP50'] = _summarize_coco(coco_eval, iouThr=0.5, areaRng='all', maxDets=coco_eval.params.maxDets[2])
    data['AP75'] = _summarize_coco(coco_eval, iouThr=0.75, areaRng='all', maxDets=coco_eval.params.maxDets[2])
    data['APs'] = _summarize_coco(coco_eval, iouThr=None, areaRng='small', maxDets=coco_eval.params.maxDets[2])
    data['APm'] = _summarize_coco(coco_eval, iouThr=None, areaRng='medium', maxDets=coco_eval.params.maxDets[2])
    data['APl'] = _summarize_coco(coco_eval, iouThr=None, areaRng='large', maxDets=coco_eval.params.maxDets[2])
    return data


def _eval_coco_obj(coco_dct_md: Union[Sequence, Dict],
                   coco_dct_lb: Dict, eval_type: str = 'bbox', ignore_class: bool = False):
    if isinstance(coco_dct_md, dict):
        annotations = coco_dct_md['annotations']
    elif isinstance(coco_dct_md, list):
        annotations = coco_dct_md
    else:
        raise Exception('err json')

    for anno in annotations:
        if 'score' not in anno.keys():
            anno['score'] = 1
    if ignore_class:
        for item in coco_dct_md:
            item['category_id'] = 0
        for item in coco_dct_lb['annotations']:
            item['category_id'] = 0
            item['category_name'] = 'object'
        cind2name = lambda cind: 'object'
        coco_dct_lb['categories'] = [{'id': 0, 'name': 'object'}]
    else:
        cates = coco_dct_lb['categories']
        cate_dict = dict([(cate['id'], cate['name']) for cate in cates])
        cind2name = lambda cind: cate_dict[cind]
    coco_lb = COCODataset.json_dct2coco_obj(coco_dct_lb)
    coco_md = coco_lb.loadRes(coco_dct_md)

    coco_eval = COCOeval(coco_lb, coco_md, eval_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    data = _summarize_coco_eval(coco_eval, cind2name=cind2name)
    return data


def _eval_coco_json(coco_dct_pth_md: str,
                    coco_dct_pth_lb: str, eval_type: str = 'bbox', ignore_class: bool = False):
    json_dict_md = load_json(coco_dct_pth_md)
    json_dict_lb = load_json(coco_dct_pth_lb)
    return _eval_coco_obj(
        coco_dct_md=json_dict_md, coco_dct_lb=json_dict_lb, eval_type=eval_type, ignore_class=ignore_class)
