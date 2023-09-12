
"""
For output analysis:

1) load data
2) create train/val batches (MAKE SURE THE SEED CORRESPONDENCES ARE CORRECT)
3) load in model
4) output preds on data, sort by loss, visualise worst/best preds, AND SAVE!!
"""


def create_dataloaders(dataset_dir:str, batch_size:int=4):
    # load data in

    # do a check for model validity here...
    print('!!! NOTE: this script assumes that the model uses a resnet50 base for data preprocessing !!!')
    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weights.transforms(antialias=True)
    print('#'*50)

    import dataloader
    dataset = dataloader.CustomTestImageDataset(dataset_dir, transform=preprocess, target_transform=None)
    train_sampler, valid_sampler = dataloader.custom_dataset_split(dataset, val_split_ratio=0.2)

    # fetch one data batch
    import torch

    # alternative
    train_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=valid_sampler)
    #
    return train_loader, val_loader


def load_model(model_path:str, model_type:str):
    # load trained resnet
    import pose_and_distance, pose, torch
    print('!!!NOTE: this script assumes the running device is on CPU!!!')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model'].module.state_dict() if hasattr(checkpoint['model'], 'module') else checkpoint['model'].state_dict()
    print('#'*50)

    model_exchange = {'resnet18': 'Resnet18PAD()', 'resnet50': 'Resnet50PAD()', 'alexnet': 'AlexNetPAD()', 'custom1': 'Custom1PAD()'}
    model = eval('pose_and_distance.' + model_exchange[model_type])

    model.load_state_dict(state_dict)
    model.eval()
    del checkpoint

    return model

    # dataset directory
    # dir = f'/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-test-same-dist-up-close/dataset-close/'


def pose_pred(model, loader, dataset_name):
    otps = {}
    import torch
    from tqdm import tqdm
    with torch.no_grad():
        model.eval()
        for batch in tqdm(loader, desc=f"Predicting pose on the {dataset_name} dataset"):

            # fetch batch
            imgs, imgs_transformed, targets, idx = batch
            preds = model(imgs_transformed)
            for i in range(len(batch)):
                otps[idx[i]] = [preds[i], targets[i], imgs[i]]
    return otps


def sort_loss(otps):
    import loss_functions
    from tqdm import tqdm
    
    # compute loss of preds
    criterion = loss_functions.RotAndDist()
    losses = {}
    for k,v in tqdm(otps.items(), desc="Computing losses on (preds, truths)"):
        
        loss = criterion(v[0], v[1])
        losses[k] = [loss]
    
    # sort the dataset and indices by loss values
    sorted_losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[1])}
    sorted_idx = list(sorted_losses.keys())
    return sorted_losses, sorted_idx


def gen_pose_estimates(n_estimates, sorted_idx, otps, dataset_dir, blender_dir, which='best'):
    from tqdm import tqdm
    import quat2sim

    # 3) visualise the best and worst predictions
    # best
    idx = sorted_idx[:n_estimates] if which == 'best' else sorted_idx[-n_estimates:]
    preds   = [otps[i][0] for i in idx]
    targets = [otps[i][1] for i in idx]
    imgs    = [otps[i][2] for i in idx]
    to_visualise = []
    for i in tqdm(range(n_estimates), desc=f"Generating pose estimate renders for {which}-loss preds"):
        try: to_visualise.append([imgs[i], quat2sim.gen_one(preds[i], idx[i], dataset_dir, blender_dir)])
        except Exception as e: print(e)

    return idx, preds, targets, imgs, to_visualise


def plot_and_save(train_loss, val_loss, save_path:str):
    import matplotlib.pyplot as plt
    import os
    plt.figure(figsize=(20,9))
    plt.hist(sum(list({k: v for k, v in train_loss.items()}.values()), []), bins=75, label='train', alpha=0.5)
    plt.hist(sum(list({k: v for k, v in val_loss.items()}.values()), []),   bins=75, label='test',  alpha=0.5)
    plt.title('Item pose loss histogram')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses_histogram.png'))
    plt.close('all')


def main(save_path:str, model_path:str, model_type:str, dataset_dir:str, blender_dir:str):
    print('******* starting output analysis (dataloaders + model load + pose pred + loss sort + pose estimate gen + save)')
    print('\t\tsave path: ',    save_path)
    print('\t\tmodel path:',    model_path)
    print('\t\tdataset path: ', dataset_dir)
    print('\t\tblender path: ', blender_dir)

    # create dataloaders
    batch_size = 4
    # dataset_dir = "/Users/dk/Documents.nosync/msc-project/blender/imgs/linspace-dataset1/dataset"
    
    print('1) creating dataloaders...')
    train_loader, val_loader = create_dataloaders(
        dataset_dir = dataset_dir,
        batch_size=batch_size)
    print('...DONE', end='\n\n')

    # load in a trained model
    model = load_model(model_path = model_path, model_type = model_type)

    # predict poses
    print(f'2) predicting poses on train ({len(train_loader)*batch_size})/val ({len(val_loader)*batch_size}) datasets respectively...')
    train_outputs = pose_pred(model, train_loader, 'training')
    val_outputs   = pose_pred(model, val_loader, 'validation')
    print('...DONE', end='\n\n')

    # compute losses and sort by losses
    print('3) computing losses and sorting by losses')
    train_loss, train_loss_idx = sort_loss(train_outputs)
    val_loss, val_loss_idx = sort_loss(val_outputs)
    print('...DONE', end='\n\n')

    # generate pose estimates
    n_estimates = 16
    print(f'4) generating the best/worst {n_estimates} pose estimates')
    best_train_pose_estimates = gen_pose_estimates(
        n_estimates=n_estimates,
        sorted_idx=train_loss_idx,
        otps=train_outputs,
        dataset_dir=dataset_dir,
        blender_dir=blender_dir,
        which='best')
    worst_train_pose_estimates = gen_pose_estimates(
        n_estimates=n_estimates,
        sorted_idx=train_loss_idx,
        otps=train_outputs,
        dataset_dir=dataset_dir,
        blender_dir=blender_dir,
        which='worst')

    best_val_pose_estimates = gen_pose_estimates(
        n_estimates=n_estimates,
        sorted_idx=val_loss_idx,
        otps=val_outputs,
        dataset_dir=dataset_dir,
        blender_dir=blender_dir,
        which='best')
    worst_val_pose_estimates = gen_pose_estimates(
        n_estimates=n_estimates,
        sorted_idx=val_loss_idx,
        otps=val_outputs,
        dataset_dir=dataset_dir,
        blender_dir=blender_dir,
        which='worst')
    print('...DONE', end='\n\n')

    # save everything - train/val loaders - train/val outputs - train/val losses and indices - {train+val}_pose_estimates
    # import pickle, os
    # print('5) saving overall report.pkl')
    # if not os.path.exists(save_path): os.makedirs(save_path)
    # with open(os.path.join(save_path, 'report.pkl'), 'wb') as f:
    #     pickle.dump({
    #         'train_loader': train_loader,
    #         'val_loader': val_loader,
    #         'train_outputs': train_outputs,
    #         'val_outputs': val_outputs,
    #         'train_loss': train_loss,
    #         'train_loss_idx': train_loss_idx,
    #         'val_loss': val_loss,
    #         'val_loss_idx': val_loss_idx,
    #         'best_train_pose_estimates': best_train_pose_estimates,
    #         'worst_train_pose_estimates': worst_train_pose_estimates,
    #         'best_val_pose_estimates': best_val_pose_estimates,
    #         'worst_val_pose_estimates': worst_val_pose_estimates,
    #     }, f)
    # print('...DONE', end='\n\n')

    # export best train/val pose estimates (for visualisation)
    import torchshow as ts

    # save a batch of inputs+respective preds
    print('6) saving best/worst preds as pngs')

    def run_with_error_handling(statement):
        try: statement
        except Exception: pass

    for i in range(len(best_train_pose_estimates)):
        try: 
            ts.save(best_train_pose_estimates[4][i],  os.path.join(save_path,  f'best-train-preds-{i}.png'))
            ts.save(worst_train_pose_estimates[4][i], os.path.join(save_path, f'worst-train-preds-{i}.png'))
        except Exception: pass 

        try:
            ts.save(best_val_pose_estimates[4][i],    os.path.join(save_path, f'best-val-preds-{i}.png'))
            ts.save(worst_val_pose_estimates[4][i],   os.path.join(save_path, f'worst-val-preds-{i}.png'))
        except Exception: pass
        # run_with_error_handling(ts.save(best_val_pose_estimates[4][i],    os.path.join(save_path, f'best-val-preds-{i}.png')))
        # run_with_error_handling(ts.save(worst_val_pose_estimates[4][i],   os.path.join(save_path, f'worst-val-preds-{i}.png')))

    # save a plot of losses
    print('...DONE', end='\n\n')
    print('7) save a plot of losses')
    plot_and_save(train_loss, val_loss, save_path)
    print('...DONE', end='\n\n')
    # end
    print(f"*!* Saved all data to {save_path }*!*")


if __name__ == "__main__":
    import argparse

    # note: to change

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--model', type=str, help='one of (`resnet50`, `resnet18`, `alexnet`, `custom1`)')
    parser.add_argument('--dataset_dir', type=str, help='dataset directory')
    parser.add_argument('--blender_dir', type=str, help='blender model file (ends with .blend) directory')

    args = parser.parse_args()
    path = args.path
    run_name = args.run_name
    model_type = args.model
    dataset_dir = args.dataset_dir
    blender_dir = args.blender_dir

    # path = '/Users/dk/Documents.nosync/msc-project/code/blender-nets/'
    # run_name = '1pw7vzwn'
    
    import os
    checkpoints = os.listdir(os.path.join(path, 'saved_models', run_name))

    for checkpoint in checkpoints:
        save_path = os.path.join(path, 'results', run_name, checkpoint)
        model_path = os.path.join(path, 'saved_models', run_name, checkpoint)
        print('model path: ', model_path)
        main(save_path, model_path, model_type, dataset_dir, blender_dir)
