from .borders import *
from ..define import Register


class OPR_TYPE:
    IOU = 'iou'
    GIOU = 'giou'
    CIOU = 'ciou'
    DIOU = 'diou'
    AREAI = 'iarea'
    AREAU = 'uarea'
    AREAB = 'barea'
    RATEI1 = 'irate1'
    RATEI2 = 'irate2'
    RATEU1 = 'urate1'
    RATEU2 = 'urate2'
    RATEB1 = 'brate1'
    RATEB2 = 'brate2'
    UOB = 'uob'
    KL = 'kl'
    KLIOU = 'kliou'
    # CRATE = 'crate'
    UNION = 'union'
    INTER = 'inter'


class IOU_TYPE:
    IOU = OPR_TYPE.IOU
    GIOU = OPR_TYPE.GIOU
    CIOU = OPR_TYPE.CIOU
    DIOU = OPR_TYPE.DIOU
    KLIOU = OPR_TYPE.KLIOU
    IRATE2 = OPR_TYPE.RATEI2


# <editor-fold desc='numpy cordcord通用'>

def cordcordN_union(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    return np.concatenate([cord_min_min, cord_max_max], axis=-1)


def cordcordN_inter(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    return np.concatenate([cord_min_max, cord_max_min], axis=-1)


def cordcordN_cpcti(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    sizei = np.maximum(cord_max_min - cord_min_max, 0)
    cpcti = np.prod(sizei, axis=-1)
    return cpcti


def cordcordN_cpctb(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpctb = np.prod(sizeb, axis=-1)
    return cpctb


def cordcordN_cpctu(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cpcti = cordcordN_cpcti(cordcordN1, cordcordN2)
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)

    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    cpctu = cpct1 + cpct2 - cpcti
    return cpctu


def cordcordN_iou(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cpcti = cordcordN_cpcti(cordcordN1, cordcordN2)
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)

    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    cpctu = cpct1 + cpct2 - cpcti
    iou = cpcti / cpctu
    return iou


def cordcordN_rateu1(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cpcti = cordcordN_cpcti(cordcordN1, cordcordN2)
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)

    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct1 / cpctu


def cordcordN_rateu2(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cpcti = cordcordN_cpcti(cordcordN1, cordcordN2)
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)

    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct2 / cpctu


def cordcordN_ratei1(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cpcti = cordcordN_cpcti(cordcordN1, cordcordN2)
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)
    return cpcti / cpct1


def cordcordN_ratei2(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cpcti = cordcordN_cpcti(cordcordN1, cordcordN2)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)

    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    return cpcti / cpct2


def cordcordN_giou(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)

    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    sizei = np.maximum(cord_max_min - cord_min_max, 0)
    cpcti = np.prod(sizei, axis=-1)
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpctb = np.prod(sizeb, axis=-1)
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)
    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    cpctu = cpct1 + cpct2 - cpcti
    return cpcti / np.clip(cpctu, a_min=1e-7, a_max=None) - (cpctb - cpctu) / np.clip(cpctb, a_min=1e-7, a_max=None)


def cordcordN_diou(cordcordN1: np.ndarray, cordcordN2: np.ndarray) -> np.ndarray:
    cord1_min, cord1_max = np.split(cordcordN1, 2, axis=-1)
    cord2_min, cord2_max = np.split(cordcordN2, 2, axis=-1)
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    sizei = np.maximum(cord_max_min - cord_min_max, 0)
    cpcti = np.prod(sizei, axis=-1)
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpct1 = np.clip(np.prod(cord1_max - cord1_min, axis=-1), a_min=0, a_max=None)
    cpct2 = np.clip(np.prod(cord2_max - cord2_min, axis=-1), a_min=0, a_max=None)
    cpctu = cpct1 + cpct2 - cpcti
    cord1_c = (cord1_max + cord1_min) / 2
    cord2_c = (cord2_max + cord2_min) / 2
    diagb = np.sum(sizeb ** 2, axis=-1)
    diagc = np.sum((cord1_c - cord2_c) ** 2, axis=-1)
    return cpcti / cpctu - diagc / diagb


# </editor-fold>

# <editor-fold desc='numpy ccrdsize通用'>
def ccrdsizeN_cpcti(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    ccrd1, size1 = np.split(ccrdsizeN1, 2, axis=-1)
    ccrd2, size2 = np.split(ccrdsizeN2, 2, axis=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    sizei = np.maximum(cord_max_min - cord_min_max, 0)
    cpcti = np.prod(sizei, axis=-1)
    return cpcti


def ccrdsizeN_cpctb(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    ccrd1, size1 = np.split(ccrdsizeN1, 2, axis=-1)
    ccrd2, size2 = np.split(ccrdsizeN2, 2, axis=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpctb = np.prod(sizeb, axis=-1)
    return cpctb


def ccrdsizeN_cpctu(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    cpcti = ccrdsizeN_cpcti(ccrdsizeN1, ccrdsizeN2)
    cpct1 = np.prod(np.split(ccrdsizeN1, 2, axis=-1)[1], axis=-1)
    cpct2 = np.prod(np.split(ccrdsizeN2, 2, axis=-1)[1], axis=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpctu


def ccrdsizeN_ratei1(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    cpcti = ccrdsizeN_cpcti(ccrdsizeN1, ccrdsizeN2)
    cpct1 = np.prod(np.split(ccrdsizeN1, 2, axis=-1)[1], axis=-1)
    return cpcti / cpct1


def ccrdsizeN_ratei2(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    cpcti = ccrdsizeN_cpcti(ccrdsizeN1, ccrdsizeN2)
    cpct2 = np.prod(np.split(ccrdsizeN2, 2, axis=-1)[1], axis=-1)
    return cpcti / cpct2


def ccrdsizeN_rateu1(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    cpcti = ccrdsizeN_cpcti(ccrdsizeN1, ccrdsizeN2)
    cpct1 = np.prod(np.split(ccrdsizeN1, 2, axis=-1)[1], axis=-1)
    cpct2 = np.prod(np.split(ccrdsizeN2, 2, axis=-1)[1], axis=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct1 / cpctu


def ccrdsizeN_rateu2(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    cpcti = ccrdsizeN_cpcti(ccrdsizeN1, ccrdsizeN2)
    cpct1 = np.prod(np.split(ccrdsizeN1, 2, axis=-1)[1], axis=-1)
    cpct2 = np.prod(np.split(ccrdsizeN2, 2, axis=-1)[1], axis=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct2 / cpctu


def ccrdsizeN_iou(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    cpcti = ccrdsizeN_cpcti(ccrdsizeN1, ccrdsizeN2)
    cpct1 = np.prod(np.split(ccrdsizeN1, 2, axis=-1)[1], axis=-1)
    cpct2 = np.prod(np.split(ccrdsizeN2, 2, axis=-1)[1], axis=-1)
    cpctu = cpct1 + cpct2 - cpcti
    iou = cpcti / cpctu
    return iou


def ccrdsizeN_giou(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    ccrd1, size1 = np.split(ccrdsizeN1, 2, axis=-1)
    ccrd2, size2 = np.split(ccrdsizeN2, 2, axis=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    sizei = np.maximum(cord_max_min - cord_min_max, 0)
    cpcti = np.prod(sizei, axis=-1)
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpctb = np.prod(sizeb, axis=-1)
    cpct1 = np.prod(size1, axis=-1)
    cpct2 = np.prod(size2, axis=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpcti / np.clip(cpctu, a_min=1e-7, a_max=None) - (cpctb - cpctu) / np.clip(cpctb, a_min=1e-7, a_max=None)


def ccrdsizeN_diou(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    ccrd1, size1 = np.split(ccrdsizeN1, 2, axis=-1)
    ccrd2, size2 = np.split(ccrdsizeN2, 2, axis=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    sizei = np.maximum(cord_max_min - cord_min_max, 0)
    cpcti = np.prod(sizei, axis=-1)
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpct1 = np.prod(size1, axis=-1)
    cpct2 = np.prod(size2, axis=-1)
    cpctu = cpct1 + cpct2 - cpcti
    diagb = np.sum(sizeb ** 2, axis=-1)
    diagc = np.sum((ccrd1 - ccrd2) ** 2, axis=-1)
    return cpcti / cpctu - diagc / diagb


def ccrdsizeN_inter(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    ccrd1, size1 = np.split(ccrdsizeN1, 2, axis=-1)
    ccrd2, size2 = np.split(ccrdsizeN2, 2, axis=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = np.minimum(cord1_max, cord2_max)
    cord_min_max = np.maximum(cord1_min, cord2_min)
    cord_max_min = np.maximum(cord_max_min, cord_min_max)
    return np.concatenate([(cord_max_min - cord_min_max) / 2, cord_max_min - cord_min_max], axis=-1)


def ccrdsizeN_union(ccrdsizeN1: np.ndarray, ccrdsizeN2: np.ndarray) -> np.ndarray:
    ccrd1, size1 = np.split(ccrdsizeN1, 2, axis=-1)
    ccrd2, size2 = np.split(ccrdsizeN2, 2, axis=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_max = np.maximum(cord1_max, cord2_max)
    cord_min_min = np.minimum(cord1_min, cord2_min)
    return np.concatenate([(cord_max_max - cord_min_min) / 2, cord_max_max - cord_min_min], axis=-1)


# </editor-fold>

# <editor-fold desc='numpy xywh运算'>

REGISTER_XYWHN_ROPR = Register()


def xywhN_ropr(xywhN1: np.ndarray, xywhN2: np.ndarray, opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYWHN_ROPR[opr_type]
    return ropr(xywhN1, xywhN2)


REGISTER_XYWHN_ROPR[OPR_TYPE.AREAI] = ccrdsizeN_cpcti
REGISTER_XYWHN_ROPR[OPR_TYPE.AREAU] = ccrdsizeN_cpctu
REGISTER_XYWHN_ROPR[OPR_TYPE.AREAB] = ccrdsizeN_cpctb
REGISTER_XYWHN_ROPR[OPR_TYPE.RATEI1] = ccrdsizeN_ratei1
REGISTER_XYWHN_ROPR[OPR_TYPE.RATEI2] = ccrdsizeN_ratei2
REGISTER_XYWHN_ROPR[OPR_TYPE.RATEU1] = ccrdsizeN_rateu1
REGISTER_XYWHN_ROPR[OPR_TYPE.RATEU2] = ccrdsizeN_rateu2
REGISTER_XYWHN_ROPR[OPR_TYPE.IOU] = ccrdsizeN_iou
REGISTER_XYWHN_ROPR[OPR_TYPE.GIOU] = ccrdsizeN_giou
REGISTER_XYWHN_ROPR[OPR_TYPE.DIOU] = ccrdsizeN_diou
REGISTER_XYWHN_ROPR[OPR_TYPE.INTER] = ccrdsizeN_inter
REGISTER_XYWHN_ROPR[OPR_TYPE.UNION] = ccrdsizeN_union

# </editor-fold>

# <editor-fold desc='numpy xyzwhl运算'>

REGISTER_XYZWHLN_ROPR = Register()


def xyzwhlN_ropr(xyzwhlN1: np.ndarray, xyzwhlN2: np.ndarray, opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYZWHLN_ROPR[opr_type]
    return ropr(xyzwhlN1, xyzwhlN2)


REGISTER_XYZWHLN_ROPR[OPR_TYPE.AREAI] = ccrdsizeN_cpcti
REGISTER_XYZWHLN_ROPR[OPR_TYPE.AREAU] = ccrdsizeN_cpctu
REGISTER_XYZWHLN_ROPR[OPR_TYPE.AREAB] = ccrdsizeN_cpctb
REGISTER_XYZWHLN_ROPR[OPR_TYPE.RATEI1] = ccrdsizeN_ratei1
REGISTER_XYZWHLN_ROPR[OPR_TYPE.RATEI2] = ccrdsizeN_ratei2
REGISTER_XYZWHLN_ROPR[OPR_TYPE.RATEU1] = ccrdsizeN_rateu1
REGISTER_XYZWHLN_ROPR[OPR_TYPE.RATEU2] = ccrdsizeN_rateu2
REGISTER_XYZWHLN_ROPR[OPR_TYPE.IOU] = ccrdsizeN_iou
REGISTER_XYZWHLN_ROPR[OPR_TYPE.GIOU] = ccrdsizeN_giou
REGISTER_XYZWHLN_ROPR[OPR_TYPE.DIOU] = ccrdsizeN_diou
REGISTER_XYZWHLN_ROPR[OPR_TYPE.INTER] = ccrdsizeN_inter
REGISTER_XYZWHLN_ROPR[OPR_TYPE.UNION] = ccrdsizeN_union
# </editor-fold>

# <editor-fold desc='numpy xyxy运算'>
REGISTER_XYXYN_ROPR = Register()


def xyxyN_ropr(xyxysN1: np.ndarray, xyxysN2: np.ndarray, opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYXYN_ROPR[opr_type]
    return ropr(xyxysN1, xyxysN2)


REGISTER_XYXYN_ROPR[OPR_TYPE.AREAI] = cordcordN_cpcti
REGISTER_XYXYN_ROPR[OPR_TYPE.AREAU] = cordcordN_cpctu
REGISTER_XYXYN_ROPR[OPR_TYPE.AREAB] = cordcordN_cpctb
REGISTER_XYXYN_ROPR[OPR_TYPE.RATEI1] = cordcordN_ratei1
REGISTER_XYXYN_ROPR[OPR_TYPE.RATEI2] = cordcordN_ratei2
REGISTER_XYXYN_ROPR[OPR_TYPE.RATEU1] = cordcordN_rateu1
REGISTER_XYXYN_ROPR[OPR_TYPE.RATEU2] = cordcordN_rateu2
REGISTER_XYXYN_ROPR[OPR_TYPE.IOU] = cordcordN_iou
REGISTER_XYXYN_ROPR[OPR_TYPE.GIOU] = cordcordN_giou
REGISTER_XYXYN_ROPR[OPR_TYPE.DIOU] = cordcordN_diou
REGISTER_XYXYN_ROPR[OPR_TYPE.INTER] = cordcordN_inter
REGISTER_XYXYN_ROPR[OPR_TYPE.UNION] = cordcordN_union
# </editor-fold>

# <editor-fold desc='numpy xyzxyz运算'>
REGISTER_XYZXYZN_ROPR = Register()


def xyzxyzN_ropr(xyzxyzsN1: np.ndarray, xyzxyzsN2: np.ndarray, opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYXYN_ROPR[opr_type]
    return ropr(xyzxyzsN1, xyzxyzsN2)


REGISTER_XYZXYZN_ROPR[OPR_TYPE.AREAI] = cordcordN_cpcti
REGISTER_XYZXYZN_ROPR[OPR_TYPE.AREAU] = cordcordN_cpctu
REGISTER_XYZXYZN_ROPR[OPR_TYPE.AREAB] = cordcordN_cpctb
REGISTER_XYZXYZN_ROPR[OPR_TYPE.RATEI1] = cordcordN_ratei1
REGISTER_XYZXYZN_ROPR[OPR_TYPE.RATEI2] = cordcordN_ratei2
REGISTER_XYZXYZN_ROPR[OPR_TYPE.RATEU1] = cordcordN_rateu1
REGISTER_XYZXYZN_ROPR[OPR_TYPE.RATEU2] = cordcordN_rateu2
REGISTER_XYZXYZN_ROPR[OPR_TYPE.IOU] = cordcordN_iou
REGISTER_XYZXYZN_ROPR[OPR_TYPE.GIOU] = cordcordN_giou
REGISTER_XYZXYZN_ROPR[OPR_TYPE.DIOU] = cordcordN_diou
REGISTER_XYZXYZN_ROPR[OPR_TYPE.INTER] = cordcordN_inter
REGISTER_XYZXYZN_ROPR[OPR_TYPE.UNION] = cordcordN_union

# </editor-fold>

# <editor-fold desc='numpy xyps运算'>
REGISTER_XYPNS_ROPR = Register()


def xypNs_ropr(xypNs1: List[np.ndarray], xypNs2: List[np.ndarray], opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYPNS_ROPR[opr_type]
    return ropr(xypNs1, xypNs2)


@REGISTER_XYPNS_ROPR.registry(OPR_TYPE.AREAI)
def xypNs_areai(xypNs1: List[np.ndarray], xypNs2: List[np.ndarray]) -> np.ndarray:
    areais = []
    for xyp1, xyp2 in zip(xypNs1, xypNs2):
        xyp_int = xypN_intersect(xyp1, xyp2)
        iarea = xypN2areaN(xyp_int)
        areais.append(iarea)
    areais = np.array(areais)
    return areais


@REGISTER_XYPNS_ROPR.registry(OPR_TYPE.IOU)
def xypNs_iou(xypNs1: List[np.ndarray], xypNs2: List[np.ndarray]) -> np.ndarray:
    ious = []
    for xyp1, xyp2 in zip(xypNs1, xypNs2):
        xyp_int = xypN_intersect(xyp1, xyp2)
        iarea = xypN2areaN(xyp_int)
        area1 = xypN2areaN(xyp1)
        area2 = xypN2areaN(xyp2)
        uarea = area1 + area2 - iarea
        iou = iarea / (uarea + 1e-16)
        ious.append(iou)
    ious = np.array(ious)
    return ious


REGISTER_XYPN_ROPR = Register()


def xypN_ropr(xypN1: np.ndarray, xypN2: np.ndarray, opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYPN_ROPR[opr_type]
    return ropr(xypN1, xypN2)


@REGISTER_XYPN_ROPR.registry(OPR_TYPE.AREAI)
def xypN_areai(xypN1: np.ndarray, xypN2: np.ndarray) -> np.ndarray:
    assert xypN1.shape == xypN2.shape
    if len(xypN1.shape) > 2:
        return np.stack([xypN_areai(xyp1N_sub, xyp2N_sub)
                         for xyp1N_sub, xyp2N_sub in zip(xypN1, xypN2)], axis=0)
    elif len(xypN1.shape) == 2:
        p1 = Polygon(xypN1)
        p2 = Polygon(xypN2)
        pi = p1.intersection(p2)
        return pi.area
    else:
        raise Exception('size err')


@REGISTER_XYPN_ROPR.registry(OPR_TYPE.IOU)
def xypN_iou(xypN1: np.ndarray, xypN2: np.ndarray) -> np.ndarray:
    area1 = xypN2areaN(xypN1)
    area2 = xypN2areaN(xypN2)
    areai = xypN_areai(xypN1, xypN2)
    areau = area1 + area2 - areai
    return areai / areau


# </editor-fold>

# <editor-fold desc='numpy xywha运算'>
REGISTER_XYWHAN_ROPR = Register()


def xywhaN_ropr(xywhaN1: np.ndarray, xywhaN2: np.ndarray, opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYWHAN_ROPR[opr_type]
    return ropr(xywhaN1, xywhaN2)


@REGISTER_XYWHAN_ROPR.registry(OPR_TYPE.AREAI)
def xywhaN_areai(xywhaN1: np.ndarray, xywhaN2: np.ndarray) -> np.ndarray:
    xyp1 = xywhaN2xypN(xywhaN1)
    xyp2 = xywhaN2xypN(xywhaN2)
    iarea = xypN_areai(xyp1, xyp2)
    return iarea


@REGISTER_XYWHAN_ROPR.registry(OPR_TYPE.IOU)
def xywhaN_iou_arr(xywhaN1: np.ndarray, xywhaN2: np.ndarray) -> np.ndarray:
    iarea = xywhaN_areai(xywhaN1, xywhaN2)
    area1 = xywhaN1[..., 2] * xywhaN1[..., 3]
    area2 = xywhaN2[..., 2] * xywhaN2[..., 3]
    uarea = area1 + area2 - iarea
    iou = iarea / uarea
    return iou


@REGISTER_XYWHAN_ROPR.registry(OPR_TYPE.RATEI1)
def xywhaN_ratei1_arr(xywhaN1: np.ndarray, xywhaN2: np.ndarray) -> np.ndarray:
    iarea = xywhaN_areai(xywhaN1, xywhaN2)
    area1 = xywhaN1[..., 2] * xywhaN1[..., 3]
    return iarea / np.clip(area1, a_min=1e-7, a_max=None)


@REGISTER_XYWHAN_ROPR.registry(OPR_TYPE.RATEI2)
def xywhaN_ratei2_arr(xywhaN1: np.ndarray, xywhaN2: np.ndarray) -> np.ndarray:
    iarea = xywhaN_areai(xywhaN1, xywhaN2)
    area2 = xywhaN2[..., 2] * xywhaN2[..., 3]
    return iarea / np.clip(area2, a_min=1e-7, a_max=None)


@REGISTER_XYWHAN_ROPR.registry(OPR_TYPE.KL)
def xywhaN_kl_arr(xywhaN1: np.ndarray, xywhaN2: np.ndarray) -> np.ndarray:
    w1, h1, a1, = xywhaN1[..., 2], xywhaN1[..., 3], xywhaN1[..., 4]
    w2, h2, a2, = xywhaN2[..., 2], xywhaN2[..., 3], xywhaN2[..., 4]
    x_dt, y_dt = xywhaN2[..., 0] - xywhaN1[..., 0], xywhaN2[..., 1] - xywhaN1[..., 1]
    wr, hr, a_dt = w2 / w1, h2 / h1, a2 - a1
    wh21, wh12, hw12, hw21 = w2 / h1, w1 / h2, h1 / w2, h2 / w1
    cos1, sin1 = np.cos(a1), np.sin(a1)
    cos2, sin2 = np.cos(a2), np.sin(a2)
    cos_dt, sin_dt = np.cos(a_dt), np.sin(a_dt)
    p1 = ((x_dt * cos1 + y_dt * sin1) / w1) ** 2 + ((y_dt * cos1 - x_dt * sin1) / h1) ** 2 \
         + ((x_dt * cos2 + y_dt * sin2) / w2) ** 2 + ((y_dt * cos2 - x_dt * sin2) / h2) ** 2
    p2 = (wr ** 2 + 1 / wr ** 2 + hr ** 2 + 1 / hr ** 2) * cos_dt ** 2 \
         + (wh21 ** 2 + wh12 ** 2 + hw12 ** 2 + hw21 ** 2) * sin_dt ** 2
    kl_dist = p1 + p2 / 4 - 1
    return kl_dist


@REGISTER_XYWHAN_ROPR.registry(OPR_TYPE.KLIOU)
def xywhaN_kliou(xywhaN1: np.ndarray, xywhaN2: np.ndarray) -> np.ndarray:
    kl_dist = xywhaN_kl_arr(xywhaN1, xywhaN2)
    kliou = 1 / (kl_dist + 1)
    return kliou


# </editor-fold>

# <editor-fold desc='numpy xyxy mask运算'>

REGISTER_XYXYSN_MASKNS_ROPR = Register()


def xyxysN_maskNs_ropr(xyxysN1: np.ndarray, maskNs1: List[np.ndarray],
                       xyxysN2: np.ndarray, maskNs2: List[np.ndarray], opr_type=OPR_TYPE.IOU) -> np.ndarray:
    ropr = REGISTER_XYXYSN_MASKNS_ROPR[opr_type]
    return ropr(xyxysN1, maskNs1, xyxysN2, maskNs2)


@REGISTER_XYXYSN_MASKNS_ROPR.registry(OPR_TYPE.AREAI)
def xyxysN_maskNs_areai(xyxysN1: np.ndarray, maskNs1: List[np.ndarray],
                        xyxysN2: np.ndarray, maskNs2: List[np.ndarray]) -> np.ndarray:
    xymax_min = np.minimum(xyxysN1[..., 2:4], xyxysN2[..., 2:4])
    xymin_max = np.maximum(xyxysN1[..., :2], xyxysN2[..., :2])
    whi = np.maximum(xymax_min - xymin_max, 0)
    areai = np.prod(whi, axis=-1)
    xymins_r1 = xymin_max - xyxysN1[..., :2]
    xymaxs_r1 = xymax_min - xyxysN1[..., :2]
    xymins_r2 = xymin_max - xyxysN2[..., :2]
    xymaxs_r2 = xymax_min - xyxysN2[..., :2]
    assert len(areai.shape) == 1, 'dim err ' + str(areai.shape)
    for i in range(areai.shape[0]):
        if areai[i] == 0:
            continue
        mask1_ref = maskNs1[i][xymins_r1[i, 1]:xymaxs_r1[i, 1], xymins_r1[i, 0]:xymaxs_r1[i, 0]]
        mask2_ref = maskNs2[i][xymins_r2[i, 1]:xymaxs_r2[i, 1], xymins_r2[i, 0]:xymaxs_r2[i, 0]]
        areai[i] = np.sum(mask1_ref * mask2_ref)
    return areai


@REGISTER_XYXYSN_MASKNS_ROPR.registry(OPR_TYPE.IOU)
def xyxysN_maskNs_iou(xyxysN1: np.ndarray, maskNs1: List[np.ndarray],
                      xyxysN2: np.ndarray, maskNs2: List[np.ndarray]) -> np.ndarray:
    iarea = xyxysN_maskNs_areai(xyxysN1, maskNs1, xyxysN2, maskNs2)
    area1 = np.array([np.sum(mask1) for mask1 in maskNs1])
    area2 = np.array([np.sum(mask2) for mask2 in maskNs2])
    uarea = area1 + area2 - iarea
    iou = iarea / uarea
    return iou


# </editor-fold>

# <editor-fold desc='numpy 边界裁剪'>
def xyxyN_clip(xyxyN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    return cordcordN_clip(xyxyN, xyxyN_rgn)


def xyN_clip(xyN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    return cordN_clip(xyN, xyxyN_rgn)


def xywhN_clip(xywhN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    xyxy = xywhN2xyxyN(xywhN)
    xyxy = xyxyN_clip(xyxy, xyxyN_rgn=xyxyN_rgn)
    xywhN = xyxyN2xywhN(xyxy)
    return xywhN


def xypN_clip(xypN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    xypN_clpd = xypN_intersect(xypN, xyxyN2xypN(xyxyN_rgn))
    return xypN_clpd


def xysN_clip(xysN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    return np.maximum(np.minimum(xysN, xyxyN_rgn[2:4]), xyxyN_rgn[0:2])


def xywhaN_clip(xywhaN: np.ndarray, xyxyN_rgn: np.ndarray) -> np.ndarray:
    if np.any(xywhaN[2:4] == 0):
        return xywhaN
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    xypN_clpd = xypN_intersect(xywhaN2xypN(xywhaN), xyxyN2xypN(xyxyN_rgn))
    xywha_clpd = xysN_aN2xywhaN(xypN_clpd, xywhaN[4])
    return xywha_clpd


# </editor-fold>

# <editor-fold desc='torch cordcord通用'>

def cordcordT_union(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    return torch.cat([cord_min_min, cord_max_max], dim=-1)


def cordcordT_inter(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    return torch.cat([cord_min_max, cord_max_min], dim=-1)


def cordcordT_cpcti(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    sizei = torch.clip(cord_max_min - cord_min_max, min=0)
    cpcti = torch.prod(sizei, dim=-1)
    return cpcti


def cordcordT_cpctb(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpctb = torch.prod(sizeb, dim=-1)
    return cpctb


def cordcordT_cpctu(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cpcti = cordcordT_cpcti(cordcordT1, cordcordT2)
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    cpct2 = torch.prod(cord2_max - cord2_min, dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpctu


def cordcordT_iou(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cpcti = cordcordT_cpcti(cordcordT1, cordcordT2)
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    cpct2 = torch.prod(torch.clamp(cord2_max - cord2_min, min=0), dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    iou = cpcti / torch.clamp(cpctu, min=1e-7)
    return iou


def cordcordT_rateu1(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cpcti = cordcordT_cpcti(cordcordT1, cordcordT2)
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    cpct2 = torch.prod(torch.clamp(cord2_max - cord2_min, min=0), dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct1 / torch.clamp(cpctu, min=1e-7)


def cordcordT_rateu2(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cpcti = cordcordT_cpcti(cordcordT1, cordcordT2)
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    cpct2 = torch.prod(torch.clamp(cord2_max - cord2_min, min=0), dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct2 / torch.clamp(cpctu, min=1e-7)


def cordcordT_ratei1(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cpcti = cordcordT_cpcti(cordcordT1, cordcordT2)
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    return cpcti / torch.clamp(cpct1, min=1e-7)


def cordcordT_ratei2(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cpcti = cordcordT_cpcti(cordcordT1, cordcordT2)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cpct2 = torch.prod(torch.clamp(cord2_max - cord2_min, min=0), dim=-1)
    return cpcti / torch.clamp(cpct2, min=1e-7)


def cordcordT_giou(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)

    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    sizei = torch.clip(cord_max_min - cord_min_max, min=0)
    cpcti = torch.prod(sizei, dim=-1)
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    sizeb = torch.clamp(cord_max_max - cord_min_min, min=0)
    cpctb = torch.prod(sizeb, dim=-1)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    cpct2 = torch.prod(torch.clamp(cord2_max - cord2_min, min=0), dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpcti / torch.clamp(cpctu, min=1e-7) - (cpctb - cpctu) / torch.clamp(cpctb, min=1e-7)


def cordcordT_diou(cordcordT1: torch.Tensor, cordcordT2: torch.Tensor) -> torch.Tensor:
    cord1_min, cord1_max = torch.chunk(cordcordT1, 2, dim=-1)
    cord2_min, cord2_max = torch.chunk(cordcordT2, 2, dim=-1)
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    sizei = torch.clip(cord_max_min - cord_min_max, min=0)
    cpcti = torch.prod(sizei, dim=-1)
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    sizeb = torch.clamp(cord_max_max - cord_min_min, min=0)
    cpct1 = torch.prod(torch.clamp(cord1_max - cord1_min, min=0), dim=-1)
    cpct2 = torch.prod(torch.clamp(cord2_max - cord2_min, min=0), dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    cord1_c = (cord1_max + cord1_min) / 2
    cord2_c = (cord2_max + cord2_min) / 2
    diagb = torch.sum(sizeb ** 2, dim=-1)
    diagc = torch.sum((cord1_c - cord2_c) ** 2, dim=-1)
    return cpcti / torch.clamp(cpctu, min=1e-7) - diagc / torch.clamp(diagb, min=1e-7)


# </editor-fold>

# <editor-fold desc='torch ccrdsize通用'>
def ccrdsizeT_cpcti(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    ccrd1, size1 = torch.chunk(ccrdsizeT1, 2, dim=-1)
    ccrd2, size2 = torch.chunk(ccrdsizeT2, 2, dim=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    sizei = torch.clip(cord_max_min - cord_min_max, min=0)
    cpcti = torch.prod(sizei, dim=-1)
    return cpcti


def ccrdsizeT_cpctb(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    ccrd1, size1 = torch.chunk(ccrdsizeT1, 2, dim=-1)
    ccrd2, size2 = torch.chunk(ccrdsizeT2, 2, dim=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    sizeb = torch.clamp(cord_max_max - cord_min_min, min=0)
    cpctb = torch.prod(sizeb, dim=-1)
    return cpctb


def ccrdsizeT_cpctu(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    cpcti = ccrdsizeT_cpcti(ccrdsizeT1, ccrdsizeT2)
    cpct1 = torch.prod(torch.chunk(ccrdsizeT1, 2, dim=-1)[1], dim=-1)
    cpct2 = torch.prod(torch.chunk(ccrdsizeT2, 2, dim=-1)[1], dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpctu


def ccrdsizeT_ratei1(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    cpcti = ccrdsizeT_cpcti(ccrdsizeT1, ccrdsizeT2)
    cpct1 = torch.prod(torch.chunk(ccrdsizeT1, 2, dim=-1)[1], dim=-1)
    return cpcti / cpct1


def ccrdsizeT_ratei2(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    cpcti = ccrdsizeT_cpcti(ccrdsizeT1, ccrdsizeT2)
    cpct2 = torch.prod(torch.chunk(ccrdsizeT2, 2, dim=-1)[1], dim=-1)
    return cpcti / cpct2


def ccrdsizeT_rateu1(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    cpcti = ccrdsizeT_cpcti(ccrdsizeT1, ccrdsizeT2)
    cpct1 = torch.prod(torch.chunk(ccrdsizeT1, 2, dim=-1)[1], dim=-1)
    cpct2 = torch.prod(torch.chunk(ccrdsizeT2, 2, dim=-1)[1], dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct1 / cpctu


def ccrdsizeT_rateu2(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    cpcti = ccrdsizeT_cpcti(ccrdsizeT1, ccrdsizeT2)
    cpct1 = torch.prod(torch.chunk(ccrdsizeT1, 2, dim=-1)[1], dim=-1)
    cpct2 = torch.prod(torch.chunk(ccrdsizeT2, 2, dim=-1)[1], dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpct2 / cpctu


def ccrdsizeT_iou(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    cpcti = ccrdsizeT_cpcti(ccrdsizeT1, ccrdsizeT2)
    cpct1 = torch.prod(torch.chunk(ccrdsizeT1, 2, dim=-1)[1], dim=-1)
    cpct2 = torch.prod(torch.chunk(ccrdsizeT2, 2, dim=-1)[1], dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    iou = cpcti / cpctu
    return iou


def ccrdsizeT_giou(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    ccrd1, size1 = torch.chunk(ccrdsizeT1, 2, dim=-1)
    ccrd2, size2 = torch.chunk(ccrdsizeT2, 2, dim=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    sizei = torch.clip(cord_max_min - cord_min_max, min=0)
    cpcti = torch.prod(sizei, dim=-1)
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    sizeb = torch.clamp(cord_max_max - cord_min_min, min=0)
    cpctb = torch.prod(sizeb, dim=-1)
    cpct1 = torch.prod(size1, dim=-1)
    cpct2 = torch.prod(size2, dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    return cpcti / torch.clamp(cpctu, min=1e-7) - (cpctb - cpctu) / torch.clamp(cpctb, min=1e-7)


def ccrdsizeT_diou(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    ccrd1, size1 = torch.chunk(ccrdsizeT1, 2, dim=-1)
    ccrd2, size2 = torch.chunk(ccrdsizeT2, 2, dim=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    sizei = torch.clip(cord_max_min - cord_min_max, min=0)
    cpcti = torch.prod(sizei, dim=-1)
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    sizeb = cord_max_max - cord_min_min
    cpct1 = torch.prod(size1, dim=-1)
    cpct2 = torch.prod(size2, dim=-1)
    cpctu = cpct1 + cpct2 - cpcti
    diagb = torch.sum(sizeb ** 2, dim=-1)
    diagc = torch.sum((ccrd1 - ccrd2) ** 2, dim=-1)
    return cpcti / cpctu - diagc / diagb


def ccrdsizeT_inter(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    ccrd1, size1 = torch.chunk(ccrdsizeT1, 2, dim=-1)
    ccrd2, size2 = torch.chunk(ccrdsizeT2, 2, dim=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_min = torch.minimum(cord1_max, cord2_max)
    cord_min_max = torch.maximum(cord1_min, cord2_min)
    cord_max_min = torch.maximum(cord_max_min, cord_min_max)
    return torch.cat([(cord_max_min - cord_min_max) / 2, cord_max_min - cord_min_max], dim=-1)


def ccrdsizeT_union(ccrdsizeT1: torch.Tensor, ccrdsizeT2: torch.Tensor) -> torch.Tensor:
    ccrd1, size1 = torch.chunk(ccrdsizeT1, 2, dim=-1)
    ccrd2, size2 = torch.chunk(ccrdsizeT2, 2, dim=-1)
    cord1_min = ccrd1 - size1 / 2
    cord1_max = ccrd1 + size1 / 2
    cord2_min = ccrd2 - size2 / 2
    cord2_max = ccrd2 + size2 / 2
    cord_max_max = torch.maximum(cord1_max, cord2_max)
    cord_min_min = torch.minimum(cord1_min, cord2_min)
    return torch.cat([(cord_max_max - cord_min_min) / 2, cord_max_max - cord_min_min], dim=-1)


# </editor-fold>

# <editor-fold desc='torch xywh运算'>
REGISTER_XYWHT_ROPR = Register()


def xywhT_ropr(xywhT1: torch.Tensor, xywhT2: torch.Tensor, opr_type=OPR_TYPE.IOU) -> torch.Tensor:
    ropr = REGISTER_XYWHT_ROPR[opr_type]
    return ropr(xywhT1, xywhT2)


REGISTER_XYWHT_ROPR[OPR_TYPE.AREAI] = ccrdsizeT_cpcti
REGISTER_XYWHT_ROPR[OPR_TYPE.AREAU] = ccrdsizeT_cpctu
REGISTER_XYWHT_ROPR[OPR_TYPE.AREAB] = ccrdsizeT_cpctb
REGISTER_XYWHT_ROPR[OPR_TYPE.RATEI1] = ccrdsizeT_ratei1
REGISTER_XYWHT_ROPR[OPR_TYPE.RATEI2] = ccrdsizeT_ratei2
REGISTER_XYWHT_ROPR[OPR_TYPE.RATEU1] = ccrdsizeT_rateu1
REGISTER_XYWHT_ROPR[OPR_TYPE.RATEU2] = ccrdsizeT_rateu2
REGISTER_XYWHT_ROPR[OPR_TYPE.IOU] = ccrdsizeT_iou
REGISTER_XYWHT_ROPR[OPR_TYPE.GIOU] = ccrdsizeT_giou
REGISTER_XYWHT_ROPR[OPR_TYPE.DIOU] = ccrdsizeT_diou
REGISTER_XYWHT_ROPR[OPR_TYPE.INTER] = ccrdsizeT_inter
REGISTER_XYWHT_ROPR[OPR_TYPE.UNION] = ccrdsizeT_union

# </editor-fold>

# <editor-fold desc='torch xyzwhl运算'>
REGISTER_XYZWHLT_ROPR = Register()


def xyzwhlT_ropr(xyzwhlT1: torch.Tensor, xyzwhlT2: torch.Tensor, opr_type=OPR_TYPE.IOU) -> torch.Tensor:
    ropr = REGISTER_XYZWHLT_ROPR[opr_type]
    return ropr(xyzwhlT1, xyzwhlT2)


REGISTER_XYZWHLT_ROPR[OPR_TYPE.AREAI] = ccrdsizeT_cpcti
REGISTER_XYZWHLT_ROPR[OPR_TYPE.AREAU] = ccrdsizeT_cpctu
REGISTER_XYZWHLT_ROPR[OPR_TYPE.AREAB] = ccrdsizeT_cpctb
REGISTER_XYZWHLT_ROPR[OPR_TYPE.RATEI1] = ccrdsizeT_ratei1
REGISTER_XYZWHLT_ROPR[OPR_TYPE.RATEI2] = ccrdsizeT_ratei2
REGISTER_XYZWHLT_ROPR[OPR_TYPE.RATEU1] = ccrdsizeT_rateu1
REGISTER_XYZWHLT_ROPR[OPR_TYPE.RATEU2] = ccrdsizeT_rateu2
REGISTER_XYZWHLT_ROPR[OPR_TYPE.IOU] = ccrdsizeT_iou
REGISTER_XYZWHLT_ROPR[OPR_TYPE.GIOU] = ccrdsizeT_giou
REGISTER_XYZWHLT_ROPR[OPR_TYPE.DIOU] = ccrdsizeT_diou
REGISTER_XYZWHLT_ROPR[OPR_TYPE.INTER] = ccrdsizeT_inter
REGISTER_XYZWHLT_ROPR[OPR_TYPE.UNION] = ccrdsizeT_union

# </editor-fold>

# <editor-fold desc='torch xyxy运算'>
REGISTER_XYXYT_ROPR = Register()


def xyxyT_ropr(xyxyT1: torch.Tensor, xyxyT2: torch.Tensor, opr_type=OPR_TYPE.IOU) -> torch.Tensor:
    ropr = REGISTER_XYXYT_ROPR[opr_type]
    return ropr(xyxyT1, xyxyT2)


REGISTER_XYXYT_ROPR[OPR_TYPE.AREAI] = cordcordT_cpcti
REGISTER_XYXYT_ROPR[OPR_TYPE.AREAU] = cordcordT_cpctu
REGISTER_XYXYT_ROPR[OPR_TYPE.AREAB] = cordcordT_cpctb
REGISTER_XYXYT_ROPR[OPR_TYPE.RATEI1] = cordcordT_ratei1
REGISTER_XYXYT_ROPR[OPR_TYPE.RATEI2] = cordcordT_ratei2
REGISTER_XYXYT_ROPR[OPR_TYPE.RATEU1] = cordcordT_rateu1
REGISTER_XYXYT_ROPR[OPR_TYPE.RATEU2] = cordcordT_rateu2
REGISTER_XYXYT_ROPR[OPR_TYPE.IOU] = cordcordT_iou
REGISTER_XYXYT_ROPR[OPR_TYPE.GIOU] = cordcordT_giou
REGISTER_XYXYT_ROPR[OPR_TYPE.DIOU] = cordcordT_diou
REGISTER_XYXYT_ROPR[OPR_TYPE.INTER] = cordcordT_inter
REGISTER_XYXYT_ROPR[OPR_TYPE.UNION] = cordcordT_union

# </editor-fold>

# <editor-fold desc='torch xyzxyz运算'>
REGISTER_XYZXYZT_ROPR = Register()


def xyzxyzT_ropr(xyzxyzT1: torch.Tensor, xyzxyzT2: torch.Tensor, opr_type=OPR_TYPE.IOU) -> torch.Tensor:
    ropr = REGISTER_XYZXYZT_ROPR[opr_type]
    return ropr(xyzxyzT1, xyzxyzT2)


REGISTER_XYZXYZT_ROPR[OPR_TYPE.AREAI] = cordcordT_cpcti
REGISTER_XYZXYZT_ROPR[OPR_TYPE.AREAU] = cordcordT_cpctu
REGISTER_XYZXYZT_ROPR[OPR_TYPE.AREAB] = cordcordT_cpctb
REGISTER_XYZXYZT_ROPR[OPR_TYPE.RATEI1] = cordcordT_ratei1
REGISTER_XYZXYZT_ROPR[OPR_TYPE.RATEI2] = cordcordT_ratei2
REGISTER_XYZXYZT_ROPR[OPR_TYPE.RATEU1] = cordcordT_rateu1
REGISTER_XYZXYZT_ROPR[OPR_TYPE.RATEU2] = cordcordT_rateu2
REGISTER_XYZXYZT_ROPR[OPR_TYPE.IOU] = cordcordT_iou
REGISTER_XYZXYZT_ROPR[OPR_TYPE.GIOU] = cordcordT_giou
REGISTER_XYZXYZT_ROPR[OPR_TYPE.DIOU] = cordcordT_diou
REGISTER_XYZXYZT_ROPR[OPR_TYPE.INTER] = cordcordT_inter
REGISTER_XYZXYZT_ROPR[OPR_TYPE.UNION] = cordcordT_union

# </editor-fold>

# <editor-fold desc='torch xyp运算'>
REGISTER_XLXLT_ROPR = Register()


def xypT_ropr(xypT1: torch.Tensor, xypT2: torch.Tensor, opr_type=OPR_TYPE.IOU) -> torch.Tensor:
    ropr = REGISTER_XLXLT_ROPR[opr_type]
    return ropr(xypT1, xypT2)


@REGISTER_XLXLT_ROPR.registry(OPR_TYPE.AREAI)
def xypT_areai(xypT1: torch.Tensor, xypT2: torch.Tensor) -> torch.Tensor:
    device = xypT1.device
    areai = xypN_intersect(xypT1.detach().cpu().numpy(), xypT2.detach().cpu().numpy())
    return torch.from_numpy(areai).to(device)


@REGISTER_XLXLT_ROPR.registry(OPR_TYPE.IOU)
def xypT_iou(xypT1: torch.Tensor, xypT2: torch.Tensor) -> torch.Tensor:
    areai = xypT_areai(xypT1, xypT2)
    area1 = xypT2areaT(xypT1)
    area2 = xypT2areaT(xypT2)
    areau = area1 + area2 - areai
    return areai / areau


# </editor-fold>

# <editor-fold desc='torch xywha运算'>
REGISTER_XYWHAT_ROPR = Register()


def xywhaT_ropr(xywhaT1: torch.Tensor, xywhaT2: torch.Tensor, opr_type=OPR_TYPE.KL) -> torch.Tensor:
    ropr = REGISTER_XYWHAT_ROPR[opr_type]
    return ropr(xywhaT1, xywhaT2)


@REGISTER_XYWHAT_ROPR.registry(OPR_TYPE.AREAI)
def xywhaT_areai(xywhaT1: torch.Tensor, xywhaT2: torch.Tensor) -> torch.Tensor:
    xyps1 = xywhaT2xypT(xywhaT1)
    xyps2 = xywhaT2xypT(xywhaT2)
    iarea = xypT_areai(xyps1, xyps2)
    return iarea


@REGISTER_XYWHAT_ROPR.registry(OPR_TYPE.IOU)
def xywhaT_iou(xywhaT1: torch.Tensor, xywhaT2: torch.Tensor) -> torch.Tensor:
    iarea = xywhaT_areai(xywhaT1, xywhaT2)
    area1 = xywhaT1[..., 2] * xywhaT1[..., 3]
    area2 = xywhaT2[..., 2] * xywhaT2[..., 3]
    uarea = area1 + area2 - iarea
    iou = iarea / uarea
    return iou


@REGISTER_XYWHAT_ROPR.registry(OPR_TYPE.KL)
def xywhaT_kl(xywhaT1: torch.Tensor, xywhaT2: torch.Tensor) -> torch.Tensor:
    w1, h1, a1, = xywhaT1[..., 2] + 1e-7, xywhaT1[..., 3] + 1e-7, xywhaT1[..., 4]
    w2, h2, a2, = xywhaT2[..., 2] + 1e-7, xywhaT2[..., 3] + 1e-7, xywhaT2[..., 4]
    x_dt, y_dt = xywhaT2[..., 0] - xywhaT1[..., 0], xywhaT2[..., 1] - xywhaT1[..., 1]
    wr, hr, a_dt = w2 / w1, h2 / h1, a2 - a1
    wh21, wh12, hw12, hw21 = w2 / h1, w1 / h2, h1 / w2, h2 / w1
    cos1, sin1 = torch.cos(a1), torch.sin(a1)
    cos2, sin2 = torch.cos(a2), torch.sin(a2)
    cos_dt, sin_dt = torch.cos(a_dt), torch.sin(a_dt)
    p1 = ((x_dt * cos1 + y_dt * sin1) / w1) ** 2 + ((y_dt * cos1 - x_dt * sin1) / h1) ** 2 \
         + ((x_dt * cos2 + y_dt * sin2) / w2) ** 2 + ((y_dt * cos2 - x_dt * sin2) / h2) ** 2
    p2 = (wr ** 2 + 1 / wr ** 2 + hr ** 2 + 1 / hr ** 2) * cos_dt ** 2 \
         + (wh21 ** 2 + wh12 ** 2 + hw12 ** 2 + hw21 ** 2) * sin_dt ** 2
    kl_dist = p1 + p2 / 4 - 1
    return kl_dist


@REGISTER_XYWHAT_ROPR.registry(OPR_TYPE.KLIOU)
def xywhaT_kliou(xywhaT1: torch.Tensor, xywhaT2: torch.Tensor) -> torch.Tensor:
    kl_dist = xywhaT_kl(xywhaT1, xywhaT2)
    kliou = 1 / (kl_dist + 1)
    return kliou


# </editor-fold>

# <editor-fold desc='torch dl运算'>

REGISTER_DLT_ROPR = Register()


def dpT_ropr(dpT1: torch.Tensor, dpT2: torch.Tensor, opr_type=OPR_TYPE.KL) -> torch.Tensor:
    ropr = REGISTER_DLT_ROPR[opr_type]
    return ropr(dpT1, dpT2)


@REGISTER_DLT_ROPR.registry(OPR_TYPE.AREAI)
def dpT_areai(dpT1: torch.Tensor, dpT2: torch.Tensor) -> torch.Tensor:
    dls_min = torch.where(dpT1 <= dpT2, dpT1, dpT2)
    iarea = torch.mean(dls_min ** 2, dim=-1) * np.pi
    return iarea


@REGISTER_DLT_ROPR.registry(OPR_TYPE.IOU)
def dpT_iou(dpT1: torch.Tensor, dpT2: torch.Tensor) -> torch.Tensor:
    dls_min = torch.where(dpT1 <= dpT2, dpT1, dpT2)
    dls_max = torch.where(dpT1 > dpT2, dpT1, dpT2)
    iou = torch.sum(dls_min ** 2, dim=-1) / (torch.sum(dls_max ** 2, dim=-1) + 1e-9)
    return iou


@REGISTER_DLT_ROPR.registry(OPR_TYPE.DIOU)
def dpT_diou(dpT1: torch.Tensor, dpT2: torch.Tensor) -> torch.Tensor:
    dls_min = torch.where(dpT1 <= dpT2, dpT1, dpT2)
    dls_max = torch.where(dpT1 > dpT2, dpT1, dpT2)
    diou = torch.mean((dls_min / dls_max) ** 2, dim=-1)
    return diou


# </editor-fold>

# <editor-fold desc='torch 边界裁剪'>

def xyxyT_clip(xyxyT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    return cordcordT_inter(xyxyT, torch.from_numpy(xyxyN_rgn).to(xyxyT.device))


def xysT_clip(xysT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    if xyxyN_rgn.shape[0] == 2:
        xyxyN_rgn = np.concatenate([np.zeros(shape=2), xyxyN_rgn], axis=0)
    xyxyT_rgn = torch.from_numpy(xyxyN_rgn).to(xysT.device)
    return torch.maximum(torch.minimum(xysT, xyxyT_rgn[2:4]), xyxyT_rgn[0:2])


def xywhT_clip(xywhT: torch.Tensor, xyxyN_rgn: np.ndarray) -> torch.Tensor:
    xyxyT = xywhT2xyxyT(xywhT)
    xyxyT = xyxyT_clip(xyxyT, xyxyN_rgn=xyxyN_rgn)
    xywhT = xyxyT2xywhT(xyxyT)
    return xywhT

# </editor-fold>
