import torch.nn as nn


class FeedforwardNeuralNetModel(nn.Module):
    '''
        The feedforward neural network model. By the way, model structure refers to https://www.mdpi.com/1424-8220/20/13/3684.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int]=[300, 100]):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class OneDimensionConvNeuralNetModel(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [300, 100]):
        super(OneDimensionConvNeuralNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(1, 8, 3), # 卷积层,输入通道1,输出通道8,kernel_size为3
            nn.ReLU(),
            nn.MaxPool1d(2), # 池化层,窗口大小2
            nn.Conv1d(8, 16, 3),
            nn.ReLU(), 
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(16*35, 120), # 全连接层
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10), # 分类输出
            
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.Dropout1d(0.2),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        out = self.layer(x)
        return out
