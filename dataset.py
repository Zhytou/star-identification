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
        self.num_dist = int(gen_cfg.split('_')[-1])
        self.label_df = label_df

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cols = [f'dist{i}' for i in range(self.num_dist)]
        dists = self.label_df.loc[idx, cols].to_numpy(float)
        cata_idx = self.label_df.loc[idx, 'cata_idx'].astype(int)

        return torch.from_numpy(dists).float(), cata_idx


class RACDataset(Dataset):
    '''Radial and cyclic based NN dataset'''

    def __init__(self, label_df: pd.DataFrame, gen_cfg: str):
        arr_nr, ns, nn = gen_cfg.split('_')[-3:]
        self.num_ring = sum(list(map(int, arr_nr.strip('[]').split(', '))))
        self.num_sector, self.num_neighbor = int(ns), int(nn)
        self.label_df = label_df

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cols = [f'ring{i}' for i in range(self.num_ring)]
        rings = self.label_df.loc[idx, cols].to_numpy(float)
        cols = [f'n{i}_sector{j}' for i in range(self.num_neighbor) for j in range(self.num_sector)]
        sectors = self.label_df.loc[idx, cols].to_numpy(float).reshape(self.num_neighbor, -1)
        cata_idx = self.label_df.loc[idx, 'cata_idx'].astype(int)

        return (torch.from_numpy(rings).float(), torch.from_numpy(sectors).float()), cata_idx


class DAADataset(Dataset):
    '''Distance and angular based NN dataset'''

    def __init__(self, label_df: pd.DataFrame, gen_cfg: str):
        arr_n = list(map(int, gen_cfg.split('_')[-1].strip('[]').split(', ')))
        self.tot_n = sum(arr_n)
        self.label_df = label_df

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sequence information: distance/angular feature + stats
        cols = [f's{i}_feat{j}' for i in range(2) for j in range(self.tot_n+4)]
        features = self.label_df.loc[idx, cols].to_numpy(float).reshape(2, -1)
        # catalog index
        cata_idx = self.label_df.loc[idx, 'cata_idx'].astype(int)

        return torch.from_numpy(features).float(), cata_idx


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
        'daa_1dcnn': DAADataset,
        'lpt_nn': LPTDataset
    }
    dataset_class = method_mapping.get(method)
    if dataset_class is None:
        raise ValueError(f"Invalid method: {method}")
    return dataset_class(df, gen_cfg)