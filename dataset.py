import os
import torch
import cv2
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class StarImageDataset(Dataset):
    '''Star image dataset'''

    def __init__(self, root_dir: str, label_file: str):
        '''
        Args:
            root_dir: directory with all the images
            label_file: name of the csv file with annotations
        '''
        self.label_df = pd.read_csv(os.path.join(root_dir, label_file))
        self.root_dir = root_dir

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.label_df.loc[idx, 'img_name'])
        img = cv2.imread(img_name, 0)
        star_id = self.label_df.loc[idx, 'star_id']

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        
        return transform(img), torch.tensor([star_id])


class LPTDataset(Dataset):
    '''Log-Polar transform based NN dataset'''

    def __init__(self, label_df: pd.DataFrame, gen_cfg: str):
        # number of distance features
        nd = int(gen_cfg.split('_')[-1])
        
        # set length of dataset
        self.len = len(label_df)

        # set data of distance features
        cols = [f'dist{i}' for i in range(nd)]
        self.dists = label_df[cols].to_numpy(np.float32)

        # set data of catalog index(labels)
        self.labels = label_df['cata_idx'].astype(int)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.dists[idx], self.labels[idx]


class RACDataset(Dataset):
    '''Radial and cyclic based NN dataset'''

    def __init__(self, label_df: pd.DataFrame, gen_cfg: str):
        arr_nr, ns, nn = gen_cfg.split('_')[-3:]
        # number of rings
        nr = sum(list(map(int, arr_nr.strip('[]').split(', '))))
        # number of sectors and neighbors
        ns, nn = int(ns), int(nn)

        # set length of dataset
        self.len = len(label_df)

        # set data of rings
        cols = [f'ring{i}' for i in range(nr)]
        self.rings = label_df[cols].to_numpy(np.float32)

        # set data of sectors
        cols = [f'n{i}_sector{j}' for i in range(nn) for j in range(ns)]
        self.sectors = label_df[cols].to_numpy(np.float32)

        # set data of catalog index(labels)
        self.labels = label_df['cata_idx'].astype(int)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return (self.rings[idx], self.sectors[idx]), self.labels[idx]


def create_dataset(method: str, df: pd.DataFrame, gen_cfg: str):
    '''
        Create dataset based on the method
    Args:
        method: the method name
        df: the dataframe with the labels
        
        gen_cfg: the generator configuration
    '''
    method_mapping = {
        'rac_1dcnn': RACDataset,
        'lpt_nn': LPTDataset
    }
    dataset_class = method_mapping.get(method)
    if dataset_class is None:
        raise ValueError(f"Invalid method: {method}")
    return dataset_class(df, gen_cfg)