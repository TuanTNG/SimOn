# follow this repo with small adaptation: https://github.com/lq010/Online-Detection-of-Action-Start/blob/master/THUMOS14/Evaluation/get_detection_performance.py
import argparse
import numpy as np

from tools.Evaluation.eval_detection import THUMOSdetection


def main(ground_truth_filename, prediction_filename,
         subset='validation', tOffset_thresholds=np.linspace(1.0, 10.0, 10),
         verbose=True, ontal_gt_from_cls_anno=None):

    anet_detection = THUMOSdetection(ground_truth_filename, prediction_filename,
                                     tOffset_thresholds=np.linspace(
                                         1.0, 10.0, 10),
                                     verbose=verbose, ontal_gt_from_cls_anno=ontal_gt_from_cls_anno)
    anet_detection.evaluate()


def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'detection task which is intended to evaluate the ability '
                   'of  algorithms to temporally localize activities in '
                   'untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--tOffset_thresholds', type=float, default=np.linspace(1.0, 10.0, 10),
                   help='Temporal intersection over union threshold.')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--ontal_gt_from_cls_anno', default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))
