import argparse
import mmcv
import numpy as np
from tal_metrics import ANETdetection
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_probs_file', type=str)
    parser.add_argument('--ground_truth_cls', type=str)
    parser.add_argument('--video_names_file', type=str)
    parser.add_argument('--oad_gt_file', type=str)
    parser.add_argument('--ambiguos_info', type=str)
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--out_preds_odas',
                        default='./cache/current.pkl', type=str)
    args = parser.parse_args()
    return args


def prob2actionness(classes, num_class=21, threshold=0.3):
    classes = classes.reshape(-1)
    assert len(classes) == num_class
    # ignore index 0
    classes[0] = 0.0
    classes_thr = classes >= threshold
    max_indices = np.nonzero(classes_thr)[0]
    probs = classes[max_indices]
    C = max_indices - 1
    return probs, C


def multi_class_action_grouping_single(all_probls, k_consecutive=6,
                                       current_video_name=None,
                                       fps_file='/data/thumos14_feat/fps_info.pkl',
                                       num_class=20):
    fps_data = mmcv.load(fps_file)

    # Return values
    return_list_dicts = [{
        'video-id': list(),
        't-start': list(),
        't-end': list(),
        'label': list(),
        'score': list()} for _ in range(num_class)]

    clssification_lists = {idx: [] for idx in range(num_class)}
    prob_lists = {idx: [] for idx in range(num_class)}
    frame_count = 1

    previous_classes = np.zeros((num_class, ))
    is_starts = np.zeros((num_class, )) == 1.0
    fps = fps_data[current_video_name]

    for idx in range(len(all_probls)):
        current_output = all_probls[idx]
        current_probs, current_classes = prob2actionness(
            current_output, num_class=num_class+1)

        for class_indx in range(num_class):
            current_d = (class_indx in current_classes)

            if previous_classes[class_indx] == 0 and current_d == 1:
                return_list_dicts[class_indx]['video-id'].append(
                    current_video_name)
                start = (k_consecutive * (frame_count-1))/fps
                return_list_dicts[class_indx]['t-start'].append(start)
                is_starts[class_indx] = True

            if previous_classes[class_indx] == 1 and current_d == 0 \
                    or ((idx == len(all_probls) - 1) and current_d == 1):  # end of video
                if ((idx == len(all_probls) - 1) and current_d == 1) and previous_classes[class_indx] == 0:
                    # clssification_list and prob_list are empty
                    clssification_lists[class_indx].append(
                        current_classes[current_classes == class_indx])
                    prob_lists[class_indx].append(
                        current_probs[current_classes == class_indx])
                end = (k_consecutive * (frame_count-1))/fps

                return_list_dicts[class_indx]['t-end'].append(end)
                return_list_dicts[class_indx]['label'].append(
                    clssification_lists[class_indx][-1])
                return_list_dicts[class_indx]['score'].append(
                    sum(prob_lists[class_indx])/len(prob_lists[class_indx]))

                # reset parameters if end of action in video
                prob_lists[class_indx] = []
                clssification_lists[class_indx] = []
                is_starts[class_indx] = False
                previous_classes[class_indx] = 0.0

            if is_starts[class_indx]:
                assert class_indx >= 0
                clssification_lists[class_indx].append(class_indx)
                prob_lists[class_indx].append(
                    current_probs[current_classes == class_indx])
        # update
        previous_classes[current_classes] = 1
        frame_count = frame_count + 1

    video_id = []
    t_start = []
    t_end = []
    label = []
    score = []
    for result in return_list_dicts:
        video_id.extend(result['video-id'])
        t_start.extend(result['t-start'])
        t_end.extend(result['t-end'])
        label.extend(result['label'])
        score.extend(result['score'])
    return_dict = {
        'video-id': video_id,
        't-start': t_start,
        't-end': t_end,
        'label': label,
        'score': score}
    # check len() of each element in return_dict, it must have same size
    all_keys = list(return_dict)
    for idx in range(len(all_keys)-1):
        assert len(return_dict[all_keys[idx]]) == len(
            return_dict[all_keys[idx+1]])
    return return_dict


def convert_oad2tal(all_probls, video_names, k_consecutive=6,
                    fps_file='/data/thumos14_feat/fps_info.pkl'):
    print('k consecutive: ', k_consecutive)

    # Return values
    return_dict = {
        'video-id': list(),
        't-start': list(),
        't-end': list(),
        'label': list(),
        'score': list()}

    unique_video_names = set(video_names)

    all_probls_array = np.array(all_probls)
    video_names_array = np.array(video_names)

    for idx, video_name in enumerate(tqdm(unique_video_names)):
        if video_name == 'video_test_0001292':
            # follow https://github.com/happyharrycn/actionformer_release
            continue

        video_prob = all_probls_array[video_names_array == video_name]

        # convert oad to tal
        video_result = multi_class_action_grouping_single(
            video_prob, current_video_name=video_name,
            k_consecutive=k_consecutive,
            num_class=args.num_class)

        # update result
        for key, val in video_result.items():
            return_dict[key].extend(val)
    # check len() of each element in return_dict, it must have same size
    all_keys = list(return_dict)
    for idx in range(len(all_keys)-1):
        assert len(return_dict[all_keys[idx]]) == len(
            return_dict[all_keys[idx+1]])

    return_dict['t-start'] = np.array(return_dict['t-start']
                                      ).reshape(-1).astype(np.float64)
    return_dict['t-end'] = np.array(return_dict['t-end']
                                    ).reshape(-1).astype(np.float64)
    return_dict['label'] = np.array(
        return_dict['label']).reshape(-1).astype(np.int64)
    return_dict['score'] = np.array(
        return_dict['score']).reshape(-1).astype(np.float64)

    return return_dict


def remove_ambiguous(all_probs, video_names, ambiguos_info):
    # follow OAD model: https://github.com/wangxiang1230/OadTR
    all_probs = np.array(all_probs)
    video_names = np.array(video_names)
    ambiguos_info = mmcv.load(ambiguos_info)
    for video, indices in ambiguos_info.items():
        current_indices = np.nonzero((video_names == video))[0]
        ambiguous_indices = current_indices[np.array(indices)]
        all_probs[ambiguous_indices] = all_probs[ambiguous_indices] * 0
    return all_probs.tolist(), video_names.tolist()


if __name__ == '__main__':
    args = get_args_parser()
    all_probs = mmcv.load(args.all_probs_file)
    ground_truth_cls = mmcv.load(args.ground_truth_cls)
    video_names = mmcv.load(args.video_names_file)

    assert len(all_probs) == len(ground_truth_cls)
    assert len(all_probs) == len(video_names)

    evaluator = ANETdetection(ant_file='/data/thumos14_feat/thumos14.json',
                              split='test',
                              tiou_thresholds=np.array(
                                  [0.3, 0.4, 0.5, 0.6, 0.7]),
                              label='label_id',
                              label_offset=0,
                              num_workers=8,
                              dataset_name=None,
                              oad_gt_file=args.oad_gt_file)

    if args.ambiguos_info is not None:
        all_probs, video_names = remove_ambiguous(
            all_probs, video_names, args.ambiguos_info)

    preds = convert_oad2tal(all_probs, video_names)

    print('evaluate on On-TAL')
    evaluator.evaluate(preds)

    mmcv.dump(preds, args.out_preds_odas)
    print('evaluation file for odas is saved at: ', args.out_preds_odas)
