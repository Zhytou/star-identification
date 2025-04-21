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
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 1600),
            nn.ReLU(),
            nn.BatchNorm1d(1600),
            nn.Linear(1600, output_dim),
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class RAC_CNN(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, input1_dim: int, input2_dim: tuple[int, int], output_dim: int):
        super(RAC_CNN, self).__init__()
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


class DAA_CNN(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, input_dim: tuple[int, int], output_dim: int):
        super(DAA_CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim[0], 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(((int(input_dim[1]/2))-3)*64, output_dim)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


def create_model(method: str, meth_params: list, num_class: int):
    '''
        Create the model for different method.
    Args:
        method: the method name
        meth_params: the parameters for the method
            rac_1dcnn: [Rb, Rp, [num_ring1, num_ring2, ...], num_sector, num_neighbor]
            daa_1dcnn: [Rb, Rp, [num_ring1, num_ring2, ...]]
            lpt_nn: [Rb, Rp, num_dist]
        num_class: the number of classes
    Returns:
        model: the model
    '''
    method_mapping = {
        #! carefaul!, rac_1dcnn accept (num_ring, num_neighbor, num_sector)
        'rac_1dcnn': (RAC_CNN, lambda params: (sum(params[-3]), (params[-1], params[-2]), num_class)),
        'daa_1dcnn': (DAA_CNN, lambda params: ((2, sum(params[-1])), num_class)),
        'lpt_nn': (FNN, lambda params: (params[-1], num_class))
    }

    model_info = method_mapping.get(method)
    if model_info is None:
        raise ValueError(f"Invalid method: {method}")

    model_class, param_extractor = model_info
    # extract parameters for the model
    model_params = param_extractor(meth_params)

    return model_class(*model_params)