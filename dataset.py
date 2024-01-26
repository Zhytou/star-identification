import os
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class StarImageDataset(Dataset):
    '''Star image dataset'''

    def __init__(self, root_dir: str, csv_file: str, transform=None):
        '''
        Args:
            root_dir: directory with all the images
            csv_file: name of the csv file with annotations
            transform: optional transform to be applied on a sample
        '''
        self.label_df = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.label_df.iloc[idx, 1])
        img = cv2.imread(img_name)
        star_id = self.label_df.iloc[idx, -1]

        if self.transform:
            img = self.transform(img)

        return torch.from_numpy(img), torch.tensor([star_id])


if __name__ == '__main__':
    dataset = StarImageDataset('data/star_images/', 'labels.csv')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    fig = plt.figure(figsize=(8, 8))
    col, row = 3, 3
    for i in range(1, col * row + 1):
        idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[idx]
        fig.add_subplot(row, col, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
