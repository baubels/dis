
import torch
def parse_command():

    import argparse
    parser = argparse.ArgumentParser(description='Baseline Pose Prediction')

    # model
    parser.add_argument('--model', default='Conv1', type=str, help='one of `resnet50_v1`, `resnet50_v1_frozen`, `resnet18`, `resnet18_frozen`, `Conv1`')
    parser.add_argument('--scheduler', default=0, type=int, help='an int, 0 means False, else True.')
    parser.add_argument('--resume',
                        default=None,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (eg: ./run/run_1/checkpoint-5.pth.tar)')    

    parser.add_argument('--doing_sweeps', default=0, type=int, help='an int, 0 means False, anything else means True')
    parser.add_argument('--project_name', default='Pose Estimation', type=str, help='Wandb name of the sweep project')
    parser.add_argument('--optional', default='', type=str, help='optional description of run arg.')
    parser.add_argument('--dataset', default='starlink_allclose', type=str, help='one of `starlink_allclose`, `starlink_random_small`')


    # parser.add_argument('--weights_init', default='paper', type=str, help='one of `paper`, `wang`')
    parser.add_argument('--criterion', default='MSELoss', type=str, help='one of: MSELoss, Angular1')

    # learning hyperparameters
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run, default is 20.')

    parser.add_argument('--batch_size', '--batch-size', default=16, type=int, help='Batch size, default is 16.')

    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate, default 0.01.')

    parser.add_argument('--lr_patience', default=7, type=int, help='Patience of LR scheduler, default 5.'
                                                                'See documentation of ReduceLROnPlateau.')   

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='SGD momentum, default 0.9.')
    
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='SGD weight decay, default 1e-4.')
    
    # dataset
    # parser.add_argument('--dataset', type=str, default="nyu", help='one of nyu, nyu_small, kitti, or nyu-huggingface (untested)')
    parser.add_argument('--save_model', default=1, type=int, help='an int, 0 means False, anything else means True')

    # training
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--print_freq_train', '-pt', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--print_freq_val', '-pv', default=45, type=int,
                        metavar='N', help='print frequency (default: 10)')

    # CUDA or not
    parser.add_argument('--disable-cuda', default=False, type=bool, help='Disable CUDA')
    
    # run profiler
    parser.add_argument('--profile', default=0, type=int, help='Run profiler, 0 means False, anything else means True')

    # final touch-ups
    args = parser.parse_args()
    args.device = None

    if not args.disable_cuda and torch.cuda.is_available():
        print(f'Cuda? {torch.cuda.is_available()}')
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    return args
