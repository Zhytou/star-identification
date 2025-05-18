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

    def __init__(self, label_df: pd.DataFrame, params: list):
        # drop na values
        label_df = label_df.dropna(axis=0)

        # number of distance features
        nd = params[2]
        
        # set length of dataset
        self.len = len(label_df)

        # set data of distance features
        cols = [f'dist{i}' for i in range(nd)]
        self.feats = label_df[cols].to_numpy(np.float32)

        # set data of catalog index(labels)
        self.labels = label_df['cata_idx'].to_numpy(int)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.feats[idx], self.labels[idx]


class RACDataset(Dataset):
    '''Radial and cyclic based NN dataset'''

    def __init__(self, label_df: pd.DataFrame, params: list):
        # drop na values
        label_df = label_df.dropna(axis=0)

        # parameters
        arr_nr, ns, nn = params[2:-1]
        # number of rings
        nr = sum(arr_nr)
        # number of sectors and neighbors
        ns, nn = int(ns), int(nn)

        # set length of dataset
        self.len = len(label_df)

        # set data of rings and sectors
        cols1 = [f'ring{i}' for i in range(nr)]
        cols2 = [f'n{i}_sector{j}' for i in range(nn) for j in range(ns)]
        cols = cols1 + cols2
        self.feats = label_df[cols].to_numpy(np.float32)

        # set data of catalog index(labels)
        self.labels = label_df['cata_idx'].to_numpy(int)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.feats[idx], self.labels[idx]


def create_dataset(method: str, df: pd.DataFrame, params: list):
    '''
        Create dataset based on the method
    Args:
        method: the method name
        df: the dataframe with the labels
        params: the method parameters
    '''
    method_mapping = {
        'rac_nn': RACDataset,
        'lpt_nn': LPTDataset
    }
    dataset_class = method_mapping.get(method)
    return dataset_class(df, params)