import os
import yaml


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.yml')
# config_path = 'config.yml'
with open(config_path, 'r') as file: args = yaml.load(file, Loader=yaml.FullLoader)

import numpy as np
def draw_wedge(ax, r_distances:list[tuple], label:str, t_min = np.pi/4, t_max = 3*np.pi/4, n_pts_range:list[int]=[100], pt_size:float=0.01, color='blue'):
    import numpy as np

    for i in range(len(r_distances)):
        r_min, r_max = r_distances[i]
        n_pts = n_pts_range[i]

        # have some kind of label for the n_objects found??

        R = np.random.rand(n_pts)*(r_max-r_min)+r_min
        T = np.random.rand(n_pts)*(t_max-t_min)+t_min
        
        if i == len(r_distances) - 1:
            ax.scatter(T,R,s=pt_size, c=color, alpha=1, label=label)
            ax.legend()
        else: 
            ax.scatter(T,R,s=pt_size, c=color, alpha=1)
        ax.annotate(f"{n_pts}", xy=[T[0], R[0]], fontsize=12, c=color)
    


def fetch_dataset(dataset, seed):
    from torchvision.models import ResNet50_Weights, ResNet18_Weights # this is cumbersome
    import dataloader

    # fetch dataset according to model's input pre-processing choice
    weights = ResNet50_Weights.IMAGENET1K_V1 if args['model']['architecture'] == 'resnet50' else ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms(antialias=True)
    dir =  args['data']['dataset']
    dataset = dataloader.CustomImageDatasetTest(dir, transform=preprocess, target_transform=None)
    return dataset


def index_sampler(dataset, val_split_ratio:float=0.2, seed:int=42):
    import numpy as np

    """Split a dataset via equal sampling into train and validation indices."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split_ratio * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def count_items_in_distance_range(dataset, indices, distance_range:tuple):
    import numpy as np
    """"""
    distances = np.array(dataset.distances)[indices]
    larger_dists = set(np.argwhere(distances>distance_range[0]).flatten())
    smaller_dists = set(np.argwhere(distances<distance_range[1]).flatten())
    distance_patch = larger_dists.intersection(smaller_dists)
    
    # assert len(distance_patch) > 0, print('nothing found in distance patch provided')
    return max(1, len(distance_patch))


def plot_data_distribution(dataset, plot_title):
    import matplotlib.pyplot as plt
    import numpy as np

    """"""
    ## 
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.set_tight_layout(True)

    ## use camera fov to model camera view angle
    cam_fov = args['data']['fov']
    ax.set_thetamin(-cam_fov/2)
    ax.set_thetamax(+cam_fov/2)

    ## use dataset distances to model camera view range
    max_obj_dist = max(np.array(dataset.distances))
    plot_distances = max_obj_dist + 1
    
    ax.set_ylim([0, plot_distances])
    ax.set_rorigin(-1e-5)
    ax.set_yticks(np.arange(0, plot_distances+1, 1))

    ## set title
    ax.set_title(plot_title, va='bottom')
    fig.canvas.draw()

    # draw scatter plots 
    # plot the data used during training separately to what was used for validation
    seed = args['data']['seed']
    train_idcs, val_idcs = index_sampler(dataset, val_split_ratio=0.2, seed=seed)

    # draw data distribution wedges
    filter_params = args['data']['filter_params']
    filter_params = [(int(filter_params[i].strip('(')), int(filter_params[i+1].strip(')'))) for i in range(0, len(filter_params), 2)]
    n_train_pts_distribution = [count_items_in_distance_range(dataset, train_idcs, drange) for drange in filter_params]
    draw_wedge(ax,
               r_distances=filter_params,
               label='training data',
               t_min=np.deg2rad(-cam_fov/2), 
               t_max=np.deg2rad(+cam_fov/2),
               n_pts_range=n_train_pts_distribution,
               pt_size=0.3, color='blue');

    n_val_pts_distribution = [count_items_in_distance_range(dataset, val_idcs, drange) 
                          for drange in [(min(dataset.distances), max(dataset.distances))]]
    min_val_dist = min(np.array(dataset.distances)[val_idcs])
    max_val_dist = max(np.array(dataset.distances)[val_idcs])

    draw_wedge(ax,
               r_distances=[(min_val_dist, max_val_dist)],
               label='validation data',
               t_min=np.deg2rad(-cam_fov/2),
               t_max=np.deg2rad(+cam_fov/2),
               n_pts_range=n_val_pts_distribution,
               pt_size=0.3, color='green');
    plt.plot()
    # plt.close()


def model_pred(model, loader, dataset_name):
    otps = {}
    import torch
    from tqdm import tqdm
    with torch.no_grad():
        model = model.to(device=args['machine']['device'])
        model.eval()
        for batch in tqdm(loader, desc=f"Predicting on the {dataset_name} dataset"):

            # fetch batch
            imgs, imgs_transformed, targets, idx = batch
            imgs_transformed, targets = imgs_transformed.to(device=args['machine']['device']), targets.to(device=args['machine']['device'])
            if args['machine']['device'] == 'cuda': torch.cuda.synchronize()

            with torch.no_grad(): 
                preds = model(imgs_transformed)
            
            for i in range(len(batch)):
                try: otps[int(idx[i])] = [preds[i].cpu(), targets[i].cpu()]
                except: print(f"couldn't fetch index {i} from this batch...")

    return otps


def compute_rotation_losses(otps:dict):
    from loss_functions import RotationDistance
    """ Compute rotation distances losses of a model's rotation outputs.
    
    otps: a dict with keys of type torch.Tensor, values of type list[tensor, tensor].
          values[i] = [q1, q2, q3, q4, d] (q_j are quaternion values, d is a distance)
    """
    rd = RotationDistance()
    rot_distances = {}
    for k,v in otps.items():
        rot_distances[k] = rd(v[0][:4], v[1][:4])
    return rot_distances


def compute_mse_distances(otps:dict):
    from loss_functions import MSELossDistance
    """ Compute MSE loss of a model's distance outputs.
    
    otps: a dict with keys of type torch.Tensor, values of type list[tensor, tensor].
          values[i] = [q1, q2, q3, q4, d] (q_j are quaternion values, d is a distance)
    """
    mse = MSELossDistance()
    mse_distances = {}
    for k,v in otps.items():
        mse_distances[k] = mse(v[0][4], v[1][4])
    return mse_distances


def plot_hist(pose_losses:dict, plot_title:str, save_dir:str):
    import matplotlib.pyplot as plt
    plt.close() # close any prior figures
    plt.hist(pose_losses.values())
    plt.title(plot_title)
    plt.savefig(save_dir)


def model_contourf(dataset, distance_samples, title_plot, plot_points, save_dest):
    import copy
    import numpy as np
    import matplotlib.pyplot as plt

    ## 
    plt.close()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.set_tight_layout(True)

    ## use camera fov to model camera view angle
    cam_fov = args['data']['fov']
    ax.set_thetamin(-cam_fov/2)
    ax.set_thetamax(+cam_fov/2)

    ## use dataset distances to model camera view range
    max_obj_dist = max(np.array(dataset.distances))
    plot_distances = max_obj_dist + 1
    
    ax.set_ylim([0, plot_distances])
    ax.set_rorigin(-1e-5)
    ax.set_yticks(np.arange(0, plot_distances+1, 1))

    ## set title
    ax.set_title(title_plot, va='bottom')
    fig.canvas.draw()

    # draw scatter plots 
    # plot the data used during training separately to what was used for validation
    seed = args['data']['seed']
    train_idcs, val_idcs = index_sampler(dataset, val_split_ratio=0.2, seed=seed)


    if plot_points:
        # draw data distribution wedges
        filter_params = args['data']['filter_params']
        filter_params = [(int(filter_params[i].strip('(')), int(filter_params[i+1].strip(')'))) for i in range(0, len(filter_params), 2)]
    
        n_train_pts_distribution = [count_items_in_distance_range(dataset, train_idcs, drange) for drange in filter_params]
        draw_wedge(ax,
                    r_distances=filter_params,
                    label='wedge label',
                    t_min=np.deg2rad(-cam_fov/2), 
                    t_max=np.deg2rad(+cam_fov/2),
                    n_pts_range=n_train_pts_distribution,
                    pt_size=0.5, color='blue');

        n_val_pts_distribution = [count_items_in_distance_range(dataset, val_idcs, drange) 
                            for drange in [(min(dataset.distances), max(dataset.distances))]]
        min_val_dist = min(np.array(dataset.distances)[val_idcs])
        max_val_dist = max(np.array(dataset.distances)[val_idcs])

        draw_wedge(ax,
                r_distances=[(min_val_dist, max_val_dist)],
                label='wedge label',
                t_min=np.deg2rad(-cam_fov/2),
                t_max=np.deg2rad(+cam_fov/2),
                n_pts_range=n_val_pts_distribution,
                pt_size=0.5, color='black');

    # get datapoint distances of predictions considered
    train_idx_distance = copy.deepcopy(distance_samples)
    for k,v in train_idx_distance.items(): train_idx_distance[k] = dataset.distances[k]

    # draw contour shadings
    azimuths = np.radians(np.linspace(-cam_fov/2, +cam_fov/2, 100))
    zeniths = sorted(train_idx_distance.values())
    r, theta = np.meshgrid(zeniths, azimuths)

    sorted_train_idx_distance = dict(sorted(train_idx_distance.items(), key=lambda item: item[1]))
    values = [distance_samples[i] for i in sorted_train_idx_distance.keys()]
    values = np.array([values]*azimuths.size)
    pc = ax.contourf(theta, r, values, alpha=0.2, levels=10)    
    fig.colorbar(pc)
    plt.savefig(save_dest)


def plot_model_scaling(model, dataset, save_dir:str):
    """
    
    """
    global args
    from torch.utils.data.sampler import SubsetRandomSampler
    import torch
    import numpy as np
    import os
    
    # 1) plot the model's training and validation data distributions (blue and green resp.)
    print("1. plotting the model's training and validation data distributions")
    plot_data_distribution(dataset, 'model training and validation data distributions')

    # 2) generate dataset dataloaders
    print("2. generating dataset dataloaders")
    train_idcs, val_idcs = index_sampler(dataset, 0.2, args['data']['seed'])
    train_sampler, val_sampler = SubsetRandomSampler(train_idcs), SubsetRandomSampler(val_idcs)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args['training']['batch_size'], sampler=train_sampler)
    val_loader   = torch.utils.data.DataLoader(dataset, batch_size=args['training']['batch_size'], sampler=val_sampler)

    # 3) plot a wedge showing the model's performance on the training set
    print("3. plotting wedges showing training data model performance")
    train_otps = model_pred(model, train_loader, 'training')
    train_rot_distances = compute_rotation_losses(train_otps) # :dict, {idx:loss, ...}
    train_mse_distances = compute_mse_distances(train_otps)   # :dict, {idx:loss, ...}

    # pose and distance histograms
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    # plot_hist(train_rot_distances, 'training rotation estimation losses', save_dir + 'train_rot_dist.png')
    # plot_hist(train_mse_distances, 'training distance estimation losses', save_dir + 'valid_rot_dist.png')

    # dataset, distance_samples, title_plot, plot_points, save_dest

    model_contourf(dataset, 
                   train_rot_distances, 
                   'obj. rotation prediction loss on train',
                   save_dest=os.path.join(save_dir, 'train_contour_rot.png'), plot_points=True)
    model_contourf(dataset, 
                   train_mse_distances, 
                   'obj. distance prediction loss on train',
                   save_dest=os.path.join(save_dir, 'train_contour_dist.png'), plot_points=True)

    # 4) plot a wedge showing the model's performance on the validation set
    print("4. plotting wedges showing validation data model performance")
    val_otps = model_pred(model, val_loader, 'val')
    val_rot_distances = compute_rotation_losses(val_otps)
    val_mse_distances = compute_mse_distances(val_otps)

    # pose and distance histograms
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    # plot_hist(train_rot_distances, 'training rotation estimation losses', save_dir + 'train_rot_dist.png')
    # plot_hist(train_mse_distances, 'training distance estimation losses', save_dir + 'valid_rot_dist.png')

    model_contourf(dataset, val_rot_distances, 'obj. rotation prediction loss on val', 
                   save_dest=os.path.join(save_dir, 'val_contour_rot.png'), plot_points=True)
    model_contourf(dataset, val_mse_distances, 'obj. distance prediction loss on val', 
                   save_dest=os.path.join(save_dir, 'val_contour_dist.png'), plot_points=True)


def load_model(model_path):
    import torch
    import pose_and_distance

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model'].module.state_dict() if hasattr(checkpoint['model'], 'module') else checkpoint['model'].state_dict()

    model_exchange = {'resnet18': 'Resnet18PAD()', 'resnet50': 'Resnet50PAD()', 'alexnet': 'AlexNetPAD()', 'custom1': 'Custom1PAD()'}
    model = eval('pose_and_distance.' + model_exchange[args['model']['architecture']])

    model.load_state_dict(state_dict)
    model.eval()
    del checkpoint
    return model


def load_dataset():
    from torchvision.models import ResNet50_Weights, ResNet18_Weights # this is cumbersome
    import dataloader

    # with the new dataloader
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocess = weights.transforms(antialias=True)
    dir = args['data']['dataset']
    # dir =  "/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-high-res/small/"
    dataset = dataloader.CustomImageDatasetTest(dir, transform=preprocess, target_transform=None)
    return dataset


def plot_model_scaling_full(model_path, save_path):
    model = load_model(model_path)
    dataset = load_dataset()
    plot_model_scaling(model, dataset, save_path)


def main():

    """Usage, assuming models are saved in cwd + saved_models + resnet18_v1:
    python3 scaling_output_analysis.py --model_path 'checkpoint-5.pth.tar'
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    saved_model = args.model_path


    import os
    path = os.getcwd()
    model_path = os.path.join(path, 'saved_models', 'resnet18_v1', saved_model)
    save_path = os.path.join(path, 'results', 'resnet18_v1', saved_model)
    plot_model_scaling_full(model_path, save_path)


if __name__ == "__main__":
    main()
