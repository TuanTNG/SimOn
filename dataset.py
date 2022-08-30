import os.path as osp
import pickle
import torch
import numpy as np
from tqdm import tqdm
import torch.utils.data as data


START_INDEX = 2


class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase):
        self.pickle_root = args.pickle_root
        self.numclass = args.numclass
        self.training = phase == 'train'

        self.feature_pretrain = args.feature
        self.inputs = []

        self.history_desision = args.history_desision
        self.history_feature = args.history_feature
        assert self.history_desision == self.history_feature
        if self.training:
            self.num_videoframes = args.train_num_videoframes
            self.stride = args.train_stride
        else:
            self.num_videoframes = args.test_num_videoframes
            self.stride = args.test_stride
        self.skip_videoframes = 1

        self.subnet = 'val' if phase == 'train' else 'test'
        print(f'loading {self.subnet} dataset.......................')

        target_all = pickle.load(
            open(osp.join(self.pickle_root, f'thumos_{self.subnet}_anno_multiclass.pickle'), 'rb'))

        self.sessions = sorted(list(set(target_all.keys())))
        feature_file = osp.join(
            self.pickle_root, f'thumos_all_feature_{self.subnet}_V3.pickle')
        if osp.exists(feature_file):
            self.feature_All = pickle.load(open(feature_file, 'rb'))
            print(f'load THUMOS feature {feature_file}')
        else:
            NotImplementedError
        self.origin_video_length_dict = dict()
        self.track_session = []
        self.track_start = []
        self.track_indices = []
        for session in tqdm(self.sessions):
            video_feature_rgb = self.feature_All[session]['rgb']
            video_feature_flow = self.feature_All[session]['flow']

            video_feature = np.concatenate(
                [video_feature_rgb, video_feature_flow], -1)

            target = target_all[session]['anno']
            # pad feature and target
            if len(video_feature) % self.stride != 0:
                tmp_data = np.zeros(
                    (self.stride-len(video_feature) % self.stride, args.dim_feature))
                video_feature = np.concatenate(
                    (video_feature, tmp_data), axis=0)
                # extend gt annotation
                tmp_data_anno = np.full(
                    (self.stride-len(video_feature) % self.stride, target.shape[-1]), -1)
                target = np.concatenate((target, tmp_data_anno), axis=0)

            num_snippet = video_feature.shape[0]

            # save original length of video
            self.origin_video_length_dict[session] = num_snippet

            num_windows = int(
                (num_snippet + self.stride - self.num_videoframes) / self.stride)

            windows_start = [i * self.stride for i in range(num_windows)]

            if num_snippet < self.num_videoframes:
                windows_start = [0]
                # Add on a bunch of zero data if there aren't enough windows.
                tmp_data = np.zeros(
                    (self.num_videoframes - num_snippet, args.dim_feature))
                video_feature = np.concatenate(
                    (video_feature, tmp_data), axis=0)

                # extend gt annotation
                tmp_data_anno = np.full(
                    (self.num_videoframes - num_snippet, target.shape[-1]), -1)
                target = np.concatenate((target, tmp_data_anno), axis=0)

            elif num_snippet - windows_start[-1] - self.num_videoframes > int(self.num_videoframes / self.skip_videoframes):
                windows_start.append(num_snippet - self.num_videoframes)

            for start in windows_start:
                obser_data = video_feature[start:start + self.num_videoframes]
                obser_data = np.array(obser_data).astype(np.float32)
                obser_target = target[start:start + self.num_videoframes]
                self.inputs.append([obser_data, obser_target, session, start])
                self.track_session.append([session]*self.num_videoframes)
                self.track_start.append(start)
                self.track_indices.append(
                    np.arange(start, start+self.num_videoframes))

        self.sos_cls = torch.full(
            (1, self.numclass), 1/(self.numclass-1)).to(torch.float32)

    def __getitem__(self, index):
        return self.prepare_imgs(index)

    def prepare_imgs(self, index):
        obser_data, obser_target, session, start = self.inputs[index]

        class_h_target = torch.from_numpy(obser_target)
        camera_inputs = torch.from_numpy(obser_data)

        return (camera_inputs, class_h_target, torch.tensor([index]))

    def prepare_training_data(self, index,
                              all_video_camera_inputs,
                              all_video_class_h_target):
        taken_decision_indices = np.arange(
            max(0, index-self.history_desision-1), index+1)

        is_start = False
        if len(taken_decision_indices) < self.history_feature + 2:
            is_start = True

        re_camera_inputs = all_video_camera_inputs[:, torch.from_numpy(
            taken_decision_indices)]
        re_class_h_target = all_video_class_h_target[:, torch.from_numpy(
            taken_decision_indices)]

        if is_start:
            re_camera_inputs = torch.cat(
                (re_camera_inputs[:, 0:1], re_camera_inputs), 1)
            # the first class is dummy and not used
            re_class_h_target = torch.cat((self.sos_cls[None].repeat(
                len(re_class_h_target), 1, 1), re_class_h_target), 1)

        return (re_camera_inputs, re_class_h_target,
                torch.tensor([is_start]*len(all_video_camera_inputs)))

    def __len__(self):
        return len(self.inputs)
