import torch
import torch.nn as nn


DEBUG = False


class FNN(nn.Module):
    '''
        The feedforward neural network model.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(FNN, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),

            nn.Linear(2048, output_dim),
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class RAC_CNN(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(RAC_CNN, self).__init__()
        
        self.conv = nn.Sequential(
            # output batch_size*64*input_dim/2
            ConvBlock(1, 64),
            
            # output batch_size*64*input_dim/4
            ConvBlock(64, 64),
            
            # output batch_size*64*input_dim/8
            ConvBlock(64, 64),
            
            # output batch_size*output_dim*input_dim/8
            nn.Conv1d(64, output_dim, kernel_size=1),

            nn.AvgPool1d(kernel_size=input_dim//8),
        )

    def forward(self, x1, x2):
        # merge the two input, x.shape is [batch_size, num_ring+num_sector*num_neighbor]
        x = torch.concat((x1, x2), dim=1)
        
        # x.shape convert into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        y = self.conv(x).squeeze(-1)

        if DEBUG:
            print(x.shape, y.shape)
        
        return y


class ConvBlock(nn.Module):
    '''
        The block for rac 1dcnn.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(ConvBlock, self).__init__()
        self.conv_3 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//2),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//2),
        )

        self.pool_2 = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        # apply the convolutional layers        
        y1 = self.conv_3(x)
        y2 = self.conv_1(x)
        
        # apply the max pooling layer
        y = self.pool_2(torch.concat((y1, y2), dim=1))

        if DEBUG:
            print(x.shape)
            print(y1.shape, y2.shape, y.shape)

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
        'rac_1dcnn': (RAC_CNN, lambda params: (sum(params[-3])+params[-2]*params[-1], num_class)),
        'lpt_nn': (FNN, lambda params: (params[-1], num_class))
    }

    model_info = method_mapping.get(method)
    if model_info is None:
        raise ValueError(f"Invalid method: {method}")

    model_class, param_extractor = model_info
    # extract parameters for the model
    model_params = param_extractor(meth_params)

    return model_class(*model_params)


if __name__ == '__main__':
    batch_size = 4
    seq_length_1 = 100
    seq_length_2 = 50

    x1 = torch.randn(batch_size, seq_length_1)
    x2 = torch.randn(batch_size, seq_length_2)

    # conv_block = ConvBlock()
    model = RAC_CNN(input_dim=seq_length_1+seq_length_2, output_dim=10)
    output = model(x1, x2)
