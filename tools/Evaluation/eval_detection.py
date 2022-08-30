import json
import mmcv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils import temporal_offset, interpolated_prec_rec


class THUMOSdetection(object):

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 tOffset_thresholds=np.linspace(1.0, 10.0, 10),
                 verbose=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.tOffset_thresholds = tOffset_thresholds
        self.verbose = verbose

        self.ap = None

        # Import index, ground truth and predictions.
        self.activity_index = {'SoccerPenalty': 0, 'VolleyballSpiking': 1,
                               'Shotput': 2, 'ThrowDiscus': 3, 'JavelinThrow': 4,
                               'HighJump': 5, 'CliffDiving': 6, 'Diving': 7,
                               'HammerThrow': 8, 'CleanAndJerk': 9,
                               'Billiards': 10, 'LongJump': 11, 'TennisSwing': 12,
                               'GolfSwing': 13, 'PoleVault': 14, 'BasketballDunk': 15,
                               'CricketBowling': 16, 'CricketShot': 17,
                               'FrisbeeCatch': 18, 'BaseballPitch': 19}

        self.ground_truth = self.load_gt_seg_from_json(ground_truth_filename)

        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for temporal offset: {}'.format(
                self.tOffset_thresholds))

    def remove_duplicate_annotations(self, ants, tol=1e-3):
        # remove duplicate annotations (same category and starting/ending time)
        valid_events = []
        for event in ants:
            s, e, l = event['segment'][0], event['segment'][1], event['label_id']
            valid = True
            for p_event in valid_events:
                if ((abs(s-p_event['segment'][0]) <= tol)
                        and (abs(e-p_event['segment'][1]) <= tol)
                        and (l == p_event['label_id'])
                    ):
                    valid = False
                    break
            if valid:
                valid_events.append(event)
        return valid_events

    def load_gt_seg_from_json(self, json_file, split='test', label='label_id', label_offset=0):
        # load json file
        with open(json_file, "r", encoding="utf8") as f:
            json_db = json.load(f)
        json_db = json_db['database']

        vids, starts, stops, labels = [], [], [], []
        for k, v in json_db.items():

            # filter based on split
            if (split is not None) and v['subset'].lower() != split:
                continue
            # remove duplicated instances
            ants = self.remove_duplicate_annotations(v['annotations'])
            # video id
            vids += [k] * len(ants)
            # for each event, grab the start/end time and label
            for event in ants:
                starts += [float(event['segment'][0])]
                stops += [float(event['segment'][1])]
                if isinstance(event[label], (tuple, list)):
                    # offset the labels by label_offset
                    label_id = 0
                    for i, x in enumerate(event[label][::-1]):
                        label_id += label_offset**i + int(x)
                else:
                    # load label_id directly
                    label_id = int(event[label])
                labels += [label_id]

        # move to pd dataframe
        gt_base = pd.DataFrame({
            'video-id': vids,
            't-start': starts,
            'label': labels
        })
        return gt_base

    def _import_prediction(self, prediction_filename):
        
        prediction = mmcv.load(prediction_filename)
        if 't-end' in prediction:
            prediction.pop('t-end')
        prediction = pd.DataFrame(prediction)
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print(
                'Warning: No predictions of label <{}> were provdied.'.format(label_name))
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tOffset_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(
                    cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(
                    prediction_by_label, label_name, cidx),
                tOffset_thresholds=self.tOffset_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print('[RESULTS] Performance on Thumos detection task.')
            print('action indexes: {}'.format(self.activity_index))
            for i in range(len(self.mAP)):
                print("temporal offset {}:".format(self.tOffset_thresholds[i]))
                # print('\t{}'.format(self.ap[i]))
                print('\tmap: {}'.format(self.mAP[i]))
            print('\nAverage-mAP: {}'.format(self.average_mAP))


def compute_average_precision_detection(ground_truth, prediction, tOffset_thresholds=np.linspace(1.0, 10.0, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with smallest offset is matches as
    true positive.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 'score']
    tOffset_thresholds : 1darray, optional
        Temporal offset threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tOffset_thresholds))
    if prediction.empty:
        return ap
    # print (prediction)##########3
    npos = float(len(ground_truth))
    # print(npos)
    # print ground_truth ''''''''

    lock_gt = np.ones((len(tOffset_thresholds), len(ground_truth))) * -1

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]

    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tOffset_thresholds), len(prediction)))
    fp = np.zeros((len(tOffset_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(
                this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()

        toff_arr = temporal_offset(this_pred['t-start'],
                                   this_gt['t-start'].values)
        # We would like to retrieve the predictions with smallest offset.
        tOffset_sorted_idx = toff_arr.argsort()
        for tidx, toff_thr in enumerate(tOffset_thresholds):
            for jdx in tOffset_sorted_idx:
                if toff_arr[jdx] > toff_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos
    # print(tp_cumsum[-1][-1])
    # print(fp_cumsum[-1][-1])
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tOffset_thresholds)):
        # print(tidx)
        # print(ground_truth)
        # print("precision: {}".format(precision_cumsum[tidx,:][-1]))

        # print("recall: {}".format(recall_cumsum[tidx,:][-1]))
        # exit(0)
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :])

    return ap
