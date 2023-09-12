
from datetime import datetime
import shutil
import time
import torch
from torch.optim import lr_scheduler
import dataloader
from metrics import AverageMeter, Result
import os
import torch.nn as nn
import loss_functions
from tqdm import tqdm
import pose_and_distance
import model_save


# ------------------------------------------------------------------------- #
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yml')

with open(config_path, 'r') as file: 
    args = yaml.load(file, Loader=yaml.FullLoader)

import wandb
if args['logging']['doing_sweeps']:
    with open(args['logging']['doing_sweeps'], 'r') as file: 
        config = yaml.load(file, Loader=yaml.FullLoader)
        print('LOADING SWEEPS')

wandb.init(project=args['logging']['project_name'], config= config if args['logging']['doing_sweeps'] else args)

print('... LOADING THE FOLLOWING CONFIG FILE ...')
print(yaml.dump(args))
print('.'*len('... LOADING THE FOLLOWING CONFIG FILE ...'))
WANDB_AGENT_MAX_INITIAL_FAILURES=100

# ------------------------------------------------------------------------- #


if args['machine']['device'] == 'cuda': 
    args['machine']['device'] = torch.device('cuda')


def create_loader():
    global args

    from torchvision.models import ResNet50_Weights, ResNet18_Weights # this is cumbersome
    if args['model']['architecture'][:8] == 'resnet50' and args['model']['architecture'][:8] != 'resnet18':
        weights = ResNet50_Weights.IMAGENET1K_V1
    elif args['model']['architecture'][:8] == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
    
    else:
        weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms(antialias=True)

    # dir =  "/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-high-res/small"
    if args['data']['dataset'] == 'multi':
        dataset = dataloader.CustomImageDataset(wandb.config.dataset, transform=preprocess, target_transform=None)
    else: 
        dataset = dataloader.CustomImageDataset(args['data']['dataset'], transform=preprocess, target_transform=None)
    print(f'dataset length: {len(dataset)}')

    # perform dataset split (this is where legitimately interesting dataset generation methods can take place.....)
    if args['data']['filters'] == 'none':
        train_sampler, valid_sampler = dataloader.standard_dataset_split(dataset, val_split_ratio=0.2)
    if args['data']['filters'] == 'distances':

        filter_params = args['data']['filter_params']
        filter_params = [(int(filter_params[i].strip('(')), int(filter_params[i+1].strip(')'))) for i in range(0, len(filter_params), 2)]

        train_sampler, valid_sampler = dataloader.dist_dataset_split(dataset, 
                                                                     filter_params, 
                                                                     val_split_ratio=0.2, 
                                                                     seed=args['data']['seed'])

    # construct dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args['training']['batch_size'] if not args['logging']['doing_sweeps'] else wandb.config.batch_size,
        sampler=train_sampler,
        num_workers=args['machine']['workers'], 
        pin_memory=True,)

    val_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,
        sampler=valid_sampler,
        num_workers=args['machine']['workers'], 
        pin_memory=True,)

    return train_loader, val_loader


def update_device():
    if torch.cuda.device_count() >=1:
        print(f"Using {torch.cuda.device_count()} GPUs...")
        args['training']['batch_size'] = args['training']['batch_size'] * torch.cuda.device_count()
    elif args['machine']['device'] == 'cuda':
        print("Using 1 GPU...", torch.cuda.current_device())
    else: print("Using CPU...")

# ------------------------------------------------------------------------- #

def main():
    global args, output_directory

    torch.manual_seed(args['data']['seed'])
    update_device()
    train_loader, val_loader = create_loader()
    model_paths = []

    if args['model']['resume']:
        assert os.path.isfile(args['model']['resume']), \
            "=> no checkpoint found at '{}'".format(args['model']['resume'])
        print("=> loading checkpoint '{}'".format(args['model']['resume']))
        checkpoint  = torch.load(args['model']['resume'])

        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer   = checkpoint['optimizer']

        # solve 'out of memory'
        model = checkpoint['model']

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        if args['machine']['device'] == 'cuda': 
            torch.cuda.empty_cache()

    else: # enum this??
        print(f"***Creating Model ({args['model']['architecture']}) ***")

        if args['model']['architecture']   == 'resnet50':
            model = pose_and_distance.Resnet50PAD()
        elif args['model']['architecture'] == 'resnet18':
            model = pose_and_distance.Resnet18PAD()
        elif args['model']['architecture'] == 'resnet18v2':
            model = pose_and_distance.Resnet18PADv2()
        elif args['model']['architecture'] == 'alexnet':
            model = pose_and_distance.AlexNetPAD()
        elif args['model']['architecture'] == 'custom1':
            model = pose_and_distance.Custom1PAD()
        elif args['model']['architecture'] == 'custom1_v2':
            model = pose_and_distance.Custom1PADv2()

        else:
            print(f"Model provided {args['model']['architecture']} does not exist.")
            raise AssertionError

    if args['model']['architecture'][-6:] == 'frozen': # freeze layers if freezing layers is considered
        print('...freezing resnet layers,')
        for layer in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']:
            try: eval('model' + '.' + layer + '.' + 'requires_grad_(False)')
            except: print(f'unable to freeze layer {layer}')

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args['training']['lr'] if not args['logging']['doing_sweeps'] else wandb.config.lr, 
                                momentum=args['training']['momentum'] if not args['logging']['doing_sweeps'] else wandb.config.momentum,
                                weight_decay=args['training']['weight_decay'] if not args['logging']['doing_sweeps'] else wandb.config.weight_decay)
    
    # model = nn.DataParallel(model, device_ids=list(args['machine']['device'])) # NOTE: may produce an error if you use CPU    
    model = model.to(device=args['machine']['device'])
    print("***Model Created***")
    
    start_epoch = 0

    # when training, use reduceLROnPlateau to reduce learning rate
    if args['model']['scheduler']:
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=args['training']['lr_patience'],
            verbose=True)

    # loss function
    if args['training']['criterion'] == 'RotAndDistLoss':
        loss_fn = loss_functions.RotAndDist(lam=args['training']['rot_importance'])
    elif args['training']['criterion'] == 'RotationDistance':
        loss_fn = loss_functions.RotationDistance()

    wandb.watch(model, log_freq=100)
    for epoch in range(start_epoch, args['training']['epochs'] if not args['logging']['doing_sweeps'] else wandb.config.epochs):

        # remember change of the learning rate and other stats; alternatively redundant seeming logging is for the wandb sweeps
        wandb.log({'lr': args['training']['lr'] if not args['logging']['doing_sweeps'] else wandb.config.lr, 
                'momentum': args['training']['momentum'] if not args['logging']['doing_sweeps'] else wandb.config.momentum, 
                'weight_decay': args['training']['weight_decay'] if not args['logging']['doing_sweeps'] else wandb.config.weight_decay,
                'epochs': args['training']['epochs'] if not args['logging']['doing_sweeps'] else wandb.config.epochs,
                'batch_size': args['training']['batch_size'] if not args['logging']['doing_sweeps'] else wandb.config.batch_size})

        wandb.log({'which_epoch': epoch})
        res = train(train_loader, model, loss_fn, optimizer, epoch)  # logger)   # train for one epoch
        res_test = validate(val_loader, model, epoch)                # logger    # evaluate on validation set

        if args['model']['save_model']:
            # save checkpoint for each epoch if args['model']['save_model'] is True
            print(f'saving checkpoint for epoch {epoch}')
            model_save.save_checkpoint({
                'args': args,
                'epoch': epoch,
                'model': model,
                'wandb_id': wandb.run.id,
                'optimizer': optimizer,
            }, epoch, 
            os.path.join(args['model']['save_model'], wandb.run.id)
            )
        
        # append to list of model_paths 
        model_paths.append(os.path.join(args['model']['save_model'], wandb.run.id))

        # when rml doesn't fall, reduce learning rate
        if args['model']['scheduler']: scheduler.step(res.mae)


        # plot model scaleability
        # import scaling_output_analysis
        # model_path = os.path.join(args['model']['save_model'], wandb.run.id, f'checkpoint-{epoch}.pth.tar')
        # save_dir = os.path.join(model_path[:-8])
        # if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        # scaling_output_analysis.plot_model_scaling_full(model_path, save_dir)

        
        save_path = os.path.join(args['model']['save_model'], wandb.run.id, 'config.yml')
        with open(save_path, 'w') as file:
            yaml.dump(args, file)
            print('saved config file')

    wandb.finish()
    return model_paths

def train(train_loader, model, loss_fn, optimizer, epoch):
    average_meter = AverageMeter()
    model.train()                                                   # switch to train mode
    end = time.time()

    # skip = args['logging']['print_freq_train'] # save images every skip iters
    tqdm.write('', end='\n')
    for i, (input, target) in enumerate(tqdm(train_loader)):

        input, target = input.to(device=args['machine']['device']), target.to(device=args['machine']['device'])
        input, target = input.float(), target.float()
        if -2 in input or -2 in target:
          print('issue in input/target')
          continue

        # print('The input tensor size is: ', input.size())  
        # print('The target tensor size is: ', target.size())
        # print(f'The input tensor values are: max: {input[:,0].max()}, min: {input[:,0].min()}, mean: {input[:,0].mean()}, std: {input[:,0].std()}')
        # print(f'The target tensor values are: max: {target.max()}, min: {target.min()}, mean: {target.mean()}, std: {target.std()}')
        # print(f'input shape: {input.shape}, target shape: {target.shape}')

        # check NaNs, etc.
        assert not torch.isnan(input).any(),  f'warning, NaNs in `input` of train_loader, epoch {i}'
        assert not torch.isnan(target).any(), f'warning, NaNs in `target` of train_loader, epoch {i}'
        
        if args['machine']['device'] == 'cuda': torch.cuda.synchronize()
        data_time = time.time() - end
        end = time.time()
        pred = model(input)  # model pred output


        # ------------------------------------------------- good???
        # PRED MODIFIER (16 +/- options down to 8...)
        import itertools

        # Generate all possible combinations of True and False for 4 variables
        variable_values = [True, False]
        combinations = list(itertools.product(variable_values, repeat=4))

        pred_bools = pred>0
        pred[pred_bools[:,0]==False]*=-1
        # ------------------------------------------------- good???



        # print shapes and check for NaNs
        # print(f'pred size = {pred.size()}, target size = {target.size()}, \
        #       pred is nan?: {torch.isnan(pred.data).any()}, target is nan?: {torch.isnan(target.data).any()}')

        assert not torch.isnan(pred).any(), 'model prediction is NaN'

        # ----------------- loss -----------------
        # with torch.autograd.detect_anomaly(): # <- (optional) detect NaNs
        loss = loss_fn(pred, target)
        loss.backward()                         # compute gradient and do SGD step
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"loss": loss})
        # ----------------------------------------

        if args['machine']['device'] == 'cuda': torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args['logging']['print_freq_train'] == 0:

        #     # do logging
        #     print('=> output: {}'.format(output_directory))
            tqdm.write(f'LOSS:{loss.item()}', end=' ')
            tqdm.write('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'Loss={Loss:.5f} '
                  'Angular={result.ang1:.2f}({average.ang1:.2f}) '
                  'Dist={result.dis1:.2f}({average.dis1:.2f})'.format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))
        
        wandb.log({'Train_Mean_Angular': result.ang1})
        wandb.log({'Train_Mean_Distance': result.dis1})
    
    # wandb: log images, groud truth and prediction
    # images = wandb.Image(input, caption="First 4 training images")
    # targets = wandb.Image(target, caption="First 4 training targets")
    # preds = wandb.Image(pred.data[:4], caption="First 4 training predictions")
    # wandb.log({"Training Images": images,
              #  "Training Targets": targets, 
              #  "Training Predictions": preds})

    avg = average_meter.average()


    

    return avg

# validation
def validate(val_loader, model, epoch):
    average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode
    end = time.time()

    # skip = len(val_loader) // 9  # save images every skip iters

    for i, (input, target) in enumerate(tqdm(val_loader)):
        input, target = input.to(device=args['machine']['device']), target.to(device=args['machine']['device'])
        input = input.float()
        target = target.float()
        if -2 in input or -2 in target:
          # print('issue in input/target')
          continue
        
        if args['machine']['device'] == 'cuda': torch.cuda.synchronize()

        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad(): pred = model(input)
        if args['machine']['device'] == 'cuda': torch.cuda.synchronize()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        wandb.log({'Test_Mean_Angular': result.ang1})
        wandb.log({'Test_Mean_Distance': result.dis1})

        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

    # images = wandb.Image(input.data[:4], caption="First 4 testing images")
    # targets = wandb.Image(target.data[:4], caption="First 4 testing targets")
    # preds = wandb.Image(pred.data[:4], caption="First 4 testing predictions")
    # wandb.log({"Testing Images": images,
    #            "Testing Targets": targets,
    #            "Testing Predictions": preds})

    avg = average_meter.average()
    print('\n*\n'
          'Mean Angular={average.ang1:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    wandb.log({'test_epoch_Mean_Angular': avg.ang1})
    wandb.log({'test_epoch_Mean_Distance': avg.dis1})
  
    return avg


if __name__ == '__main__':
    # run training script
    model_paths = main()
    
    # # perform output analysis
    # import subprocess
    # cmd = ['python', 'output_analysis.py', 
    #        '--path', 'value1', 
    #        '--run_name', 'value2']

    # # Run the command and capture the output
    # subprocess.run(cmd, capture_output=True, text=True)

    # save a gif of the runs
