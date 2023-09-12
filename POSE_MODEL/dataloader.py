
# import json
# import os
# import torch
# import torchvision
# from torchvision.io import read_image
# import numpy as np
# from torch.utils.data.sampler import SubsetRandomSampler
# import re

# import copy


# # class CustomImageDatasetOld(torch.utils.data.Dataset):
# #     def __init__(self, dir, transform=None, target_transform=None):
# #         self.img_dir = dir

# #         self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
# #         self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
# #         self.images = [filename for filename in self.images if "(1)" not in filename]
# #         self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
# #         self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
# #         self.labels = [filename for filename in self.labels if "(1)" not in filename]   # this removes label redundancies

# #         self.transform = transform
# #         self.target_transform = target_transform

# #     def get_label(self, idx): # returns a length-5 tensor of quaternion pose appended by distance
# #         label_path = os.path.join(self.img_dir, f'meta_{idx}.json')
# #         pose_label = json.load(open(label_path))['pose']
# #         dist_label = json.load(open(label_path))['distance']

# #         pose_label.append(dist_label)
# #         return torch.tensor(pose_label)

# #     def __len__(self):
# #         return len(self.images)

# #     def __getitem__(self, idx):
# #         assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"
# #         print(idx, self.__len__())

# #         img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
# #         image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
# #         label = self.get_label(idx)

# #         if self.transform:          image = self.transform(image)
# #         if self.target_transform:   label = self.target_transform(label)
# #         return image, label


# class CustomImageDataset(torch.utils.data.Dataset):
#     def __init__(self, dir, transform=None, target_transform=None):
#         self.img_dir = dir

#         self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
#         self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
#         self.images = [filename for filename in self.images if "(1)" not in filename]
#         self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
#         self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
#         self.labels = [filename for filename in self.labels if "(1)" not in filename]   # this removes label redundancies

#         # print(self.images[:2]) # ['Sat_small_0.png', 'Sat_small_1.png']
#         # print(self.labels[:2]) # ['meta_0.json', 'meta_1.json']

#         self.label_paths = [os.path.join(self.img_dir, f'meta_{idx}.json')   for idx in range(len(self.labels))]
#         self.distances   = [json.load(open(self.label_paths[i]))['distance'] for i in range(len(self.label_paths))]
#         print(f'\nfrom a dataset with item distance range from {np.min(self.distances)} to {np.max(self.distances)}, ')

#         self.poses       = [json.load(open(self.label_paths[i]))['pose']     for i in range(len(self.label_paths))]

#         self.transform = transform
#         self.target_transform = target_transform

#     def get_label(self, idx): # returns a length-5 tensor of quaternion pose appended by distance
#         label_path = os.path.join(self.img_dir, f'meta_{idx}.json')
#         pose_label = json.load(open(label_path))['pose']
#         dist_label = json.load(open(label_path))['distance']

#         pose_label.append(dist_label)
#         return torch.tensor(pose_label)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"
#         print(idx, self.__len__())

#         img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
#         image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
#         label = self.get_label(idx)

#         if self.transform:          image = self.transform(image)
#         if self.target_transform:   label = self.target_transform(label)
#         return image, label


# class CustomTestImageDataset(torch.utils.data.Dataset):
#     def __init__(self, dir, transform=None, target_transform=None):
#         self.img_dir = dir

#         self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
#         self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
#         self.images = [filename for filename in self.images if "(1)" not in filename]
#         self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
#         self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
#         self.labels = [filename for filename in self.labels if "(1)" not in filename]
        
#         self.transform = transform
#         self.target_transform = target_transform

#     def get_label(self, idx):
#         label_path = os.path.join(self.img_dir, f'meta_{idx}.json')
#         pose_label = json.load(open(label_path))['pose']
#         dist_label = json.load(open(label_path))['distance']
#         pose_label.append(dist_label)
#         return torch.tensor(pose_label)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"

#         img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
#         image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
#         label = self.get_label(idx)

#         if self.transform:          
#             image_transformed = self.transform(image)
#         else:
#             image_transformed = image
        
#         if self.target_transform:   label = self.target_transform(label)
#         return image, image_transformed, label, idx


# def index_sampler(dataset, val_split_ratio:float=0.2, seed:int=42):
#     import numpy as np

#     """Split a dataset via equal sampling into train and validation indices."""
#     dataset_size = len(dataset)
#     indices = list(range(dataset_size))
#     split = int(np.floor(val_split_ratio * dataset_size))
#     np.random.seed(seed)
#     np.random.shuffle(indices)

#     train_indices, val_indices = indices[split:], indices[:split]
#     return train_indices, val_indices


# def standard_dataset_split(dataset, val_split_ratio:float=0.2):
#     """Split a dataset into train and validation (indices) and resp. pytorch sampling class."""
#     train_indices, val_indices = index_sampler(dataset, val_split_ratio)
#     train_sampler = SubsetRandomSampler(train_indices)
#     valid_sampler = SubsetRandomSampler(val_indices)
#     return train_sampler, valid_sampler





import json
import os
import torch
import torchvision
from torchvision.io import read_image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import re
import copy


# class CustomImageDatasetOld(torch.utils.data.Dataset):
#     def __init__(self, dir, transform=None, target_transform=None):
#         self.img_dir = dir

#         self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
#         self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
#         self.images = [filename for filename in self.images if "(1)" not in filename]
#         self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
#         self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
#         self.labels = [filename for filename in self.labels if "(1)" not in filename]   # this removes label redundancies


#         # print(self.images[:2]) # ['Sat_small_0.png', 'Sat_small_1.png']
#         # print(self.labels[:2]) # ['meta_0.json', 'meta_1.json']

#         # self.label_paths = [os.path.join(self.img_dir, f'meta_{idx}.json') for idx in range(len(self.labels))]
#         # self.distances   = [json.load(open(self.label_paths[i]))['distance'] for i in range(len(self.label_paths))]
#         # self.poses       = [json.load(open(self.label_paths[i]))['pose'] for i in range(len(self.label_paths))]

#         self.transform = transform
#         self.target_transform = target_transform

#     def get_label(self, idx): # returns a length-5 tensor of quaternion pose appended by distance
#         label_path = os.path.join(self.img_dir, f'meta_{idx}.json')
#         pose_label = json.load(open(label_path))['pose']
#         # pose_label = self.poses[idx]
#         dist_label = json.load(open(label_path))['distance']
#         # dist_label = self.distances[idx]

#         pose_label.append(dist_label)
#         return torch.tensor(pose_label)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"

#         img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
#         image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
#         label = self.get_label(idx)

#         if self.transform:          image = self.transform(image)
#         if self.target_transform:   label = self.target_transform(label)
#         return image, label


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None, target_transform=None):
        self.img_dir = dir

        self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
        self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
        self.images = [filename for filename in self.images if "(1)" not in filename]
        self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
        self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
        self.labels = [filename for filename in self.labels if "(1)" not in filename]   # this removes label redundancies

        # print(self.images[:2]) # ['Sat_small_0.png', 'Sat_small_1.png']
        # print(self.labels[:2]) # ['meta_0.json', 'meta_1.json']

        self.label_paths = [os.path.join(self.img_dir, f'meta_{idx}.json')   for idx in range(len(self.labels))]
        self.distances   = [json.load(open(self.label_paths[i]))['distance'] for i in range(len(self.label_paths))]
        print(f'\nfrom a dataset with item distance range from {np.min(self.distances)} to {np.max(self.distances)}, ')

        self.poses       = [json.load(open(self.label_paths[i]))['pose']     for i in range(len(self.label_paths))]

        self.transform = transform
        self.target_transform = target_transform

    def get_label(self, idx): # returns a length-5 tensor of quaternion pose appended by distance
        label_path = os.path.join(self.img_dir, f'meta_{idx}.json')
        pose_label = json.load(open(label_path))['pose']
        dist_label = json.load(open(label_path))['distance']
        
        pose_label.append(dist_label)
        return torch.tensor(pose_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"

        img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        label = self.get_label(idx)

        if self.transform:          image = self.transform(image)
        if self.target_transform:   label = self.target_transform(label)
        return image, label


class CustomImageDatasetTest(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None, target_transform=None):
        self.img_dir = dir

        self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
        self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
        self.images = [filename for filename in self.images if "(1)" not in filename]
        self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
        self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
        self.labels = [filename for filename in self.labels if "(1)" not in filename]   # this removes label redundancies

        # print(self.images[:2]) # ['Sat_small_0.png', 'Sat_small_1.png']
        # print(self.labels[:2]) # ['meta_0.json', 'meta_1.json']

        self.label_paths = [os.path.join(self.img_dir, f'meta_{idx}.json')   for idx in range(len(self.labels))]
        self.distances   = [json.load(open(self.label_paths[i]))['distance'] for i in range(len(self.label_paths))]
        print(f'\nfrom a dataset with item distance range from {np.min(self.distances)} to {np.max(self.distances)}, ')

        self.poses       = [json.load(open(self.label_paths[i]))['pose']     for i in range(len(self.label_paths))]

        self.transform = transform
        self.target_transform = target_transform

    def get_label(self, idx): # returns a length-5 tensor of quaternion pose appended by distance
        pose_label = copy.deepcopy(self.poses[idx])
        dist_label = self.distances[idx]
        pose_label.append(dist_label)
        return torch.tensor(pose_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"

        img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        label = self.get_label(idx)

        if self.transform:          
            image_transformed = self.transform(image)
        else:
            image_transformed = image
        
        if self.target_transform:   label = self.target_transform(label)
        return image, image_transformed, label, idx


class CustomTestImageDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transform=None, target_transform=None):
        self.img_dir = dir

        self.images = [img for img in os.listdir(dir) if img.endswith(".png")]
        self.images.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
        self.images = [filename for filename in self.images if "(1)" not in filename]
        self.labels = [lab for lab in os.listdir(dir) if lab.endswith(".json")]
        self.labels.sort(key=lambda x: int(re.search(r'\d+', x.split('.')[0]).group()))
        self.labels = [filename for filename in self.labels if "(1)" not in filename]
        
        self.transform = transform
        self.target_transform = target_transform

    def get_label(self, idx):
        label_path = os.path.join(self.img_dir, f'meta_{idx}.json')
        pose_label = json.load(open(label_path))['pose']
        dist_label = json.load(open(label_path))['distance']
        pose_label.append(dist_label)
        return torch.tensor(pose_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        assert idx < self.__len__(), f"Index {idx} out of bounds for dataset of size {self.__len__()}"

        img_path = os.path.join(self.img_dir, f'Sat_{idx}.png')
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        label = self.get_label(idx)

        if self.transform:          
            image_transformed = self.transform(image)
        else:
            image_transformed = image
        
        if self.target_transform:   label = self.target_transform(label)
        return image, image_transformed, label, idx


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


def standard_dataset_split(dataset, val_split_ratio:float=0.2):
    """Split a dataset into train and validation (indices) and resp. pytorch sampling class."""
    train_indices, val_indices = index_sampler(dataset, val_split_ratio)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def distance_masking(dataset, distances_dict:list[tuple], indices:list[int]):
    """ Applies a distance masking to a dataset.

    distances_dict: [(d1, d2), (d3, d4), ...]
                    d_i: Numeric
                    d_{i} < d_{i+1} throughout
                    len(distances_dict) >= 1

    dataset: type: torch.utils.data.Dataset
             the dataset must have a .distances attribute

    indices: type: list[int]
             indices must be valid; min(indices) >= 0, max(indices) < len(dataset)
    """
    assert len(distances_dict) >= 1, print('distances dict must have non-zero length') 
    assert hasattr(dataset, 'distances'), print('dataset must have a .distances method implemented')
    assert max(indices) < len(dataset) and min(indices) >= 0, print('indices out of bounds for the provided dataset')

    filtered_idcs = []
    distances = np.array(dataset.distances)[indices]

    print(f'\nfrom a dataset with item distance range from {np.min(distances)} to {np.max(distances)}, ')
    print(f'fetching distances {distances_dict}\n')

    for drange in distances_dict:
        larger_dists = set(np.argwhere(distances>drange[0]).flatten())
        smaller_dists = set(np.argwhere(distances<drange[1]).flatten())
        distance_patch = list(larger_dists.intersection(smaller_dists))
        if len(distance_patch) > 0: filtered_idcs.append(distance_patch)

    filtered_idcs = [element for sublist in filtered_idcs for element in sublist]
    return filtered_idcs # filtered dataset according to the indices


def dist_dataset_split(dataset, distances_dict:list[tuple], val_split_ratio:float=0.2, seed:int=42):
    """Split a dataset into train and validation (indices) and resp. pytorch sampling class."""
    train_indices, val_indices = index_sampler(dataset, val_split_ratio, seed)

    ## defining data options
    train_indices = distance_masking(dataset, distances_dict, train_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


## USAGE:
# dir =  "/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-high-res/2k/closer/small"
# dataset = CustomImageDataset(dir, preprocess, None)
# dataloader = torch.utils.data.DataLoader(
#     dataset, 
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,      # args.workers,
#     pin_memory=True,)

# # Display image and label.
# import matplotlib.pyplot as plt

# train_features, train_labels = next(iter(dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze().permute(1, 2, 0)
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")



def dist_dataset_split(dataset, distances_dict:list[tuple], val_split_ratio:float=0.2, seed:int=42):
    """Split a dataset into train and validation (indices) and resp. pytorch sampling class."""
    train_indices, val_indices = index_sampler(dataset, val_split_ratio, seed)

    ## defining data options
    train_indices = distance_masking(dataset, distances_dict, train_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


## USAGE:
# dir =  "/Users/dk/Documents.nosync/msc-project/blender/imgs/starlink-high-res/2k/closer/small"
# dataset = CustomImageDataset(dir, preprocess, None)
# dataloader = torch.utils.data.DataLoader(
#     dataset, 
#     batch_size=32,
#     shuffle=True,
#     num_workers=4,      # args.workers,
#     pin_memory=True,)

# # Display image and label.
# import matplotlib.pyplot as plt

# train_features, train_labels = next(iter(dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze().permute(1, 2, 0)
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
