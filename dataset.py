import os
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from generate import num_ring, num_sector

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
        
        self.label_df = pd.read_csv(label_file_path)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cols = [f'ring_{i}' for i in range(num_ring)] + [f'sector_{i}' for i in range(num_sector)]
        points = self.label_df.loc[idx, cols].values
        catalogue_idx = self.label_df.loc[idx, 'catalogue_idx'].astype(int)
        
        return torch.from_numpy(points).float(), catalogue_idx


if __name__ == '__main__':
    img_dataset = StarImageDataset('data/star_images/', 'labels.csv')
    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3
    for i in range(1, col * row + 1):
        idx = torch.randint(len(img_dataset), size=(1,)).item()
        img, label = img_dataset[idx]
        fig.add_subplot(row, col, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

    pt_dataset = StarPointDataset('data/star_points/labels.csv')
    pt_loader = DataLoader(pt_dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for points, labels in pt_loader:
        print(points.shape, points.dtype, points.device)
        points.to(device)
        print(points.shape, points.dtype, points.device)
        break
