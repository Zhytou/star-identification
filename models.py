import torch
import torch.nn as nn


class FNN(nn.Module):
    '''
        The feedforward neural network model.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(FNN, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, output_dim),
        )

    def forward(self, x1, x2):
        x = torch.concat((x1, torch.flatten(x2, start_dim=1)), dim=1)
        y = self.fc(x)
        return y


class CNN(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, input1_dim: int, input2_dim: tuple[int, int], output_dim: int):
        super(CNN, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input1_dim, 500),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(input2_dim[0], 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500+32*(input2_dim[1]-4), 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, output_dim),
        )

    def forward(self, x1, x2):
        y1 = self.fc1(x1)
        y2 = self.conv(x2)
        y = self.fc2(torch.concat((y1, y2), dim=1))
        return y
