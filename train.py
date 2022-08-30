import os
from typing import Iterable
import torch
import utils
import torch.nn.functional as F
import numpy as np
import mmcv

all_class_name = ["BaseballPitch",
                  "BasketballDunk",
                  "Billiards",
                  "CleanAndJerk",
                  "CliffDiving",
                  "CricketBowling",
                  "CricketShot",
                  "Diving",
                  "FrisbeeCatch",
                  "GolfSwing",
                  "HammerThrow",
                  "HighJump",
                  "JavelinThrow",
                  "LongJump",
                  "PoleVault",
                  "Shotput",
                  "SoccerPenalty",
                  "TennisSwing",
                  "ThrowDiscus",
                  "VolleyballSpiking"]


def focal_loss(input, target, criterion,
               is_starts, num_history=3,
               gamma=2, alpha=0.25):
    assert target.shape == input.shape, 'input and target must have same shape'
    input = input[:, 1:-1]
    target = target[:, 1:-1]

    if num_history is None:
        return criterion(input, target)

    # traning
    loss = F.binary_cross_entropy_with_logits(
        input, target, reduction='none')

    # focal loss
    p = torch.sigmoid(input)
    p_t = p * target + (1 - p) * (1 - target)
    loss = loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * loss

    C = input.shape[-1]
    mask = torch.zeros((input.shape[0]//num_history, num_history)).to(input)
    mask[:, -1] = 1.0

    mask[is_starts.view(-1), :] = 1.0
    mask = mask.reshape(-1)[:, None]

    padding_inds = target.sum(-1) < 0
    mask[padding_inds, :] = 0.0

    return (loss*mask).sum()/(mask.sum())


def clean_output(all_probs, all_classes, all_video_names):
    # erase padded data
    all_probs = np.array(all_probs)
    all_classes = np.array(all_classes)
    all_video_names = np.array(all_video_names)
    assert len(all_video_names) == len(all_probs)
    valid_indices = all_classes.sum(-1) >= 0.0
    all_probs = all_probs[valid_indices]
    all_classes = all_classes[valid_indices]
    all_video_names = all_video_names[valid_indices]
    return all_probs.tolist(), all_classes.tolist(), all_video_names.tolist()


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    num_class = args.numclass
    max_length = args.history_feature + 2
    print('max length sequence: ', max_length)
    all_probs = []
    all_classes = []
    all_video_names = []

    # 1. Loop through data loader to take batch of videos
    for (all_video_camera_inputs, all_video_class_h_target, all_video_inds) \
            in metric_logger.log_every(data_loader, print_freq, header):

        length_of_video = all_video_camera_inputs.shape[1]

        video_indices = all_video_inds.reshape(-1).tolist()
        video_name_lists = [data_loader.dataset.inputs[_idx][-2]
                            for _idx in video_indices]

        history_probs = [0.0]

        # 3. train model with sequential data
        for idx in range(length_of_video):
            # construct batch
            camera_inputs, class_h_target, is_starts \
                = data_loader.dataset.prepare_training_data(
                    idx,
                    all_video_camera_inputs,
                    all_video_class_h_target)

            camera_inputs = camera_inputs.to(device)

            class_h_target = class_h_target.to(device)

            is_starts = is_starts.to(device).reshape(-1)

            # log prob
            # do not use p_t in the model, so there is no gt while infer
            if is_starts.sum() > 0:
                history_probs[0] = class_h_target[:,
                                                  0:1][..., :-1]  # start token, not gt

            if len(history_probs) > max_length-1:
                history_probs.pop(0)

            pre_probs = torch.cat(history_probs, 1).to(device)

            # 4. forward model
            cls_scores = model(camera_inputs, is_starts, pre_probs)

            # log output
            dummy_cls = cls_scores[:, -1:][..., :-1].sigmoid().detach()
            dummy_cls[..., 0] = 0.0  # dummy, not use
            history_probs.append(dummy_cls)

            # through away past target
            class_h_target = class_h_target[:, -1:]

            # save output
            for _idx in range(len(class_h_target)):
                all_probs.append(
                    dummy_cls[_idx].cpu().numpy().reshape(num_class-1))
                all_classes.append(
                    class_h_target[:, -1:, :-1][_idx].cpu().numpy().reshape(num_class-1).astype(np.float32))
            all_video_names.extend(video_name_lists)

            assert len(all_video_names) == len(all_probs)

            # CLS loss
            losses = focal_loss(cls_scores.reshape(-1, num_class),
                                class_h_target.reshape(-1,
                                                       num_class),
                                criterion, is_starts,
                                num_history=cls_scores.shape[1])
            # logger
            metric_logger.update(loss_decoder=losses)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm)
            optimizer.step()

    if epoch % 4 == 0:
        # rearrage output
        all_probs, all_classes, all_video_names = clean_output(
            all_probs, all_classes, all_video_names)

        print('start eval accuracy on training set..............................')
        # eval classification
        all_probs = np.asarray(all_probs).T
        print(str(all_probs.shape))

        all_classes = np.asarray(all_classes).T
        print(str(all_classes.shape))
        results = {'probs': all_probs, 'labels': all_classes}

        map, aps, _, _ = utils.frame_level_map_n_cap(results)
        print('-----------------------OAD evaluation results------------------')
        print('[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, args.feature, map))

        for i, ap in enumerate(aps):
            cls_name = all_class_name[i]
            print('{}: {:.4f}'.format(cls_name, ap))
        stats = {k: meter.global_avg for k,
                 meter in metric_logger.meters.items()}
        print(stats)

        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable, device: torch.device,
             logger, args,
             epoch: int, nprocs=None):
    all_probs_dict, all_classes_dict = dict(), dict()
    all_video_names_dict = dict()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    model.eval()
    num_class = args.numclass
    max_length = args.history_feature + 2
    print('max length sequence: ', max_length)

    for (all_video_camera_inputs, all_video_class_h_target, all_video_inds) \
            in metric_logger.log_every(data_loader, print_freq, header):

        length_of_video = all_video_camera_inputs.shape[1]
        video_indices = all_video_inds.reshape(-1).tolist()
        video_name_lists = [data_loader.dataset.inputs[_idx][-2]
                            for _idx in video_indices]

        history_probs = [0.0]

        for idx in range(length_of_video):
            # construct batch
            camera_inputs, class_h_target, is_starts \
                = data_loader.dataset.prepare_training_data(
                    idx,
                    all_video_camera_inputs,
                    all_video_class_h_target)

            camera_inputs = camera_inputs.to(device)

            class_h_target = class_h_target.to(device)

            is_starts = is_starts.to(device).reshape(-1)

            # log prob
            # do not use p_t in the model, so there is no gt while infer

            if is_starts.sum() > 0:
                history_probs[0] = class_h_target[:,
                                                  0:1][..., :-1]  # start token, not gt

            if len(history_probs) > max_length-1:
                history_probs.pop(0)

            pre_probs = torch.cat(history_probs, 1).to(device)

            cls_scores = model(camera_inputs, is_starts, pre_probs)

            # log output
            dummy_cls = cls_scores[:, -1:][..., :-1].sigmoid()
            dummy_cls[..., 0] = 0.0  # dummy, not use
            history_probs.append(dummy_cls)

            # through away past target
            class_h_target = class_h_target[:, -1:]

            # save output
            for _idx, _vid in enumerate(video_indices):
                if _vid in all_probs_dict:
                    all_probs_dict[_vid].append(
                        dummy_cls[_idx].cpu().numpy().reshape(num_class-1))
                    all_classes_dict[_vid].append(
                        class_h_target[:, -1:, :-1][_idx].cpu().numpy().reshape(num_class-1).astype(np.float32))
                    all_video_names_dict[_vid].append(video_name_lists[_idx])
                else:
                    all_probs_dict[_vid] = [
                        dummy_cls[_idx].cpu().numpy().reshape(num_class-1), ]
                    all_classes_dict[_vid] = [
                        class_h_target[:, -1:, :-1][_idx].cpu().numpy().reshape(num_class-1).astype(np.float32), ]
                    all_video_names_dict[_vid] = [video_name_lists[_idx], ]

            assert len(all_video_names_dict) == len(all_probs_dict)
            # CLS loss
            losses = focal_loss(cls_scores.reshape(-1, num_class),
                                class_h_target.reshape(-1,
                                                       num_class),
                                criterion, is_starts,
                                num_history=cls_scores.shape[1])

            metric_logger.update(loss_decoder=losses)
    # convert dict to list
    all_probs = []
    all_classes = []
    all_video_names = []
    for idx in range(len(data_loader.dataset.inputs)):
        all_probs.extend(all_probs_dict[idx])
        all_classes.extend(all_classes_dict[idx])
        all_video_names.extend(all_video_names_dict[idx])

    all_probs, all_classes, all_video_names = clean_output(
        all_probs, all_classes, all_video_names)
    # save for evalue mAP
    save_val_dir = os.path.join(args.output_dir, 'validation_results')
    save_all_prob_file = os.path.join(
        save_val_dir, f'test_{epoch}_all_probs.pkl')
    save_all_classes_file = os.path.join(
        save_val_dir, f'test_{epoch}_ground_truth_cls.pkl')
    save_all_video_name_file = os.path.join(
        save_val_dir, f'test_{epoch}_video_names.pkl')
    mmcv.dump(all_probs, save_all_prob_file)
    mmcv.dump(all_classes, save_all_classes_file)
    mmcv.dump(all_video_names, save_all_video_name_file)
    print('results are saved at: ', save_all_prob_file)
    print(save_all_classes_file)
    print(save_all_video_name_file)

    # eval classification
    all_probs = np.asarray(all_probs).T
    print(str(all_probs.shape))

    all_classes = np.asarray(all_classes).T
    print(str(all_classes.shape))
    results = {'probs': all_probs, 'labels': all_classes}

    map, aps, _, _ = utils.frame_level_map_n_cap(results)
    print('-----------------------OAD evaluation results------------------')
    print('[Epoch-{}] [IDU-{}] mAP: {:.4f}\n'.format(epoch, args.feature, map))

    for i, ap in enumerate(aps):
        cls_name = all_class_name[i]
        print('{}: {:.4f}'.format(cls_name, ap))
    stats = {k: meter.global_avg for k,
             meter in metric_logger.meters.items()}
    model.train()

    return stats
