import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Optimizer
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr_drop', default=1, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--clip_max_norm', default=1., type=float,
                        help='gradient clipping max norm')

    # decoder
    parser.add_argument('--model_name', type=str, default='SimOn')
    parser.add_argument('--ignore_index', default=21, type=int)
    parser.add_argument('--feature', default='thumosV3', type=str,
                        help="feature type")
    parser.add_argument('--num_decoder_layers', default=4, type=int)
    parser.add_argument('--dim_feature', default=400, type=int,
                        help="input feature dims")
    parser.add_argument('--numclass', default=22, type=int,
                        help="Number of class")
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--feat_forward_dim', default=256, type=int)
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--nhead', default=8, type=int)

    # dataset parameters
    parser.add_argument('--pickle_root', type=str,
                        default='/data/thumos14_feat')
    parser.add_argument('--train_num_videoframes', default=128, type=int)
    parser.add_argument('--train_stride', default=128, type=int)
    parser.add_argument('--test_num_videoframes', default=2048, type=int)
    parser.add_argument('--test_stride', default=2048, type=int)

    # Other
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='models',
                        help='path where to save, empty for no saving')
    parser.add_argument('--removelog', action='store_true',
                        help='remove old log')
    parser.add_argument('--seed', default=20, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--history_desision', default=6, type=int)
    parser.add_argument('--history_feature', default=6, type=int)
    parser.add_argument('--max_lengh_clip', default=5000, type=int)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dataparallel', action='store_true')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:12342',
                        help='url used to set up distributed training')
    return parser
