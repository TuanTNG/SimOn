import argparse
import datetime
import json
import random
import time
from pathlib import Path
from config import get_args_parser
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
import util as utl
import os
import utils

import simon
from dataset import TRNTHUMOSDataLayer
from train import train_one_epoch, evaluate
import torch.nn as nn


def main(args):
    utils.init_distributed_mode(args)
    command = 'python ' + ' '.join(sys.argv)
    this_dir = args.output_dir
    if args.removelog:
        print('remove logs !')
        if os.path.exists(os.path.join(this_dir, 'log_dist.txt')):
            os.remove(os.path.join(this_dir, 'log_dist.txt'))
        if os.path.exists(Path(args.output_dir) / "log_tran&test.txt"):
            os.remove(Path(args.output_dir) / "log_tran&test.txt")
    logger = utl.setup_logger(os.path.join(
        this_dir, 'log_dist.txt'), command=command)
    # logger.output_print("git:\n  {}\n".format(utils.get_sha()))

    # save args
    for arg in vars(args):
        logger.output_print("{}:{}".format(arg, getattr(args, arg)))

    device = args.device

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # SimOn
    model = eval(f'simon.{args.model_name}')(args)
    model.to(device)

    criterior = nn.BCEWithLogitsLoss()

    model_without_ddp = model

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.output_print('number of params: {}'.format(n_parameters))
    # logger.output_print(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, args.lr_drop, gamma=args.gamma)

    if not args.eval:
        dataset_train = TRNTHUMOSDataLayer(args=args, phase='train')
        dataset_val = TRNTHUMOSDataLayer(args=args, phase='val')

        sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       pin_memory=True, num_workers=args.num_workers)
    else:
        dataset_val = TRNTHUMOSDataLayer(args=args, phase='val')

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, pin_memory=True, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print('checkpoint: ', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print('start testing for one epoch !!!')
        with torch.no_grad():
            test_stats = evaluate(
                model, criterior, data_loader_val, device,
                logger, args, epoch=0, nprocs=4)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterior, data_loader_train,
            optimizer, device, epoch, args.clip_max_norm, args=args)

        lr_scheduler.step()
        if args.output_dir:
            # save main model
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths.append(
                output_dir / f'checkpoint{epoch:04}.pth')

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if epoch % 4 == 0:
            test_stats = evaluate(
                model, criterior, data_loader_val, device,
                logger, args, epoch, nprocs=utils.get_world_size())

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log_tran&test.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'On-TAL Transformer', parents=[get_args_parser()])
    args = parser.parse_args()
    # args.dataset = osp.basename(osp.normpath(args.data_root)).upper()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
