import json
import numpy as np


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def temporal_offset_2(target_AS, candidate_AS, candidate_END):
    """compute the temporal offset between a target AS and all the test AS
    Parameters:
        target_AS: float 
            tarting 
        condinate_AS: 1d array
            N * [starting]
    """
    # return np.absolute(target_AS - candidate_AS)
    result = np.absolute(candidate_AS - target_AS)
    mask_end = target_AS > candidate_END # if the target_AS > the end of the groundtruth -> TRUE
    mask_start = target_AS < candidate_AS-0.5
    mask = np.logical_or(mask_start, mask_end)
    return result, mask

def temporal_offset(target_AS, candidate_AS):
    """compute the temporal offset between a target AS and all the test AS
    Parameters:
        target_AS: float 
            tarting 
        condinate_AS: 1d array
            N * [starting]
    """
    # return np.absolute(target_AS - candidate_AS)
    result = np.absolute(candidate_AS - target_AS)
    return result