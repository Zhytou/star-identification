import os
import torch
import cv2
import pandas as pd
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


class StarPointDataset(Dataset):
    '''Star point dataset'''

    def __init__(self, root_dir: str):
        '''
        Args:
            root_dir: name of the dataset directory
        '''
        label_file_path = os.path.join(root_dir, 'labels.csv')
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f'{label_file_path}does not exist')
        
        dirs = root_dir.split('/')
        print(dirs)
        assert len(dirs) <= 6 and dirs[0] == 'data' and dirs[1] == 'star_points'
        self.num_ring, self.num_sector, self.num_neighbor_limit = list(map(int, dirs[3].split('_')))
        
        self.label_df = pd.read_csv(label_file_path)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cols = [f'ring_{i}' for i in range(self.num_ring)] + [f'neighbor_{i}_sector_{j}' for i in range(self.num_neighbor_limit) for j in range(self.num_sector)]
        points = self.label_df.loc[idx, cols].values
        catalogue_idx = self.label_df.loc[idx, 'catalogue_idx'].astype(int)
        
        return torch.from_numpy(points).float(), catalogue_idx

