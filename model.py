import torch
import torch.nn as nn


DEBUG = False


class FNN(nn.Module):
    '''
        The feedforward neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):
        super(FNN, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_feat),
            
            nn.Linear(num_feat, 512),
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

            nn.Linear(2048, num_class),
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class CNN1(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):

        super(CNN1, self).__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(1, 64),
            
            ConvBlock(64, 64),
            
            ConvBlock(64, 64),
            
            ConvBlock(64, 64),

            ConvBlock(64, 64),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        # x is composed of two input: raidal features and cyclic features
        # convert x.shape from [batch_size, num_ring+num_sector*num_neighbor]
        # into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        y = self.fc(self.conv(x).squeeze(-1))

        if DEBUG:
            print(
                'RAC_CNN',
                '\nX shape', x.shape,
                '\nY shape', y.shape,
            )

        return y


class CNN2(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):
        '''
            input_dim: the dimension of the features
            output_dim: the number of the class(guide star)
        '''

        super(CNN2, self).__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(1, 256),
            
            ConvBlock(256, 256),
            
            ConvBlock(256, 256),

            ConvBlock(256, 256),
            
            nn.Conv1d(256, 512, kernel_size=1),

            # global avg pool
            nn.AdaptiveAvgPool1d(output_size=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),

            nn.Linear(2048, num_class),
        )

    def forward(self, x):
        # x is composed of two input: raidal features and cyclic features
        # convert x.shape from [batch_size, num_ring+num_sector*num_neighbor]
        # into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        # then apply the fully-connected layers
        y = self.fc(self.conv(x).squeeze(-1))

        if DEBUG:
            print(
                'RAC_CNN',
                '\nX shape', x.shape,
                '\nY shape', y.shape,
            )

        return y
    

class CNN3(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):
        '''
            input_dim: the dimension of the features
            output_dim: the number of the class(guide star)
        '''

        super(CNN3, self).__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(1, 256),
            
            ConvBlock(256, 256),
            
            ConvBlock(256, 256),
            
            ConvBlock(256, 256),

            ConvBlock(256, 256),

            nn.Conv1d(256, num_class, kernel_size=1),

            # global avg pool
            # output batch_size*256*1
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        # x is composed of two input: raidal features and cyclic features
        # convert x.shape from [batch_size, num_ring+num_sector*num_neighbor]
        # into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        y = self.conv(x).squeeze(-1)

        if DEBUG:
            print(
                'RAC_CNN',
                '\nX shape', x.shape,
                '\nY shape', y.shape,
            )

        return y
        

class CNN4(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):
        '''
            input_dim: the dimension of the features
            output_dim: the number of the class(guide star)
        '''

        super(CNN4, self).__init__()
        
        self.conv = nn.Sequential(
            ConvBlock(1, 64),
            
            ConvBlock(64, 64),
            
            ConvBlock(64, 64),
            
            ConvBlock(64, 64),

            ConvBlock(64, 64),

            nn.Conv1d(64, num_class, kernel_size=1),

            # global avg pool
            # output batch_size*256*1
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        # x is composed of two input: raidal features and cyclic features
        # convert x.shape from [batch_size, num_ring+num_sector*num_neighbor]
        # into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        y = self.conv(x).squeeze(-1)

        if DEBUG:
            print(
                'RAC_CNN',
                '\nX shape', x.shape,
                '\nY shape', y.shape,
            )

        return y


class ConvBlock(nn.Module):
    '''
        The block for rac 1dcnn.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(ConvBlock, self).__init__()
        self.conv_5 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//4),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//4),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//2),
        )

        self.pool_2 = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        # apply the convolutional layers        
        y1 = self.conv_1(x)
        y2 = self.conv_3(x)
        y3 = self.conv_5(x)

        # apply the max pooling layer
        y = self.pool_2(torch.concat((y1, y2, y3), dim=1))

        if DEBUG:
            print(
                'ConvBlock',
                '\nX shape', x.shape,
                '\nY1 shape', y1.shape,
                '\nY2 shape', y2.shape,
                '\nY3 shape', y3.shape,
                '\nY shape', y.shape
            )

        return y


def create_model(method: str, model_type: str, meth_params: list, num_class: int) -> nn.Module:
    '''
        Create the model for different method.
    Args:
        method: the method name
        meth_params: the parameters for the method
            rac_nn: [Rb, Rp, [num_ring1, num_ring2, ...], num_sector, num_neighbor, use_prob]
            lpt_nn: [Rb, Rp, num_dist, use_prob]
        num_class: the number of classes
    Returns:
        model: the model
    '''
    model_mapping = {
        'fnn': FNN,
        'cnn1': CNN1,
        'cnn2': CNN2,
        'cnn3': CNN3,
        'cnn4': CNN4,
    }

    method_mapping = {
        #! carefaul!, rac_nn accept (num_ring, num_neighbor, num_sector)
        # use lambda to change params dynamically(dict is static)
        'rac_nn': lambda params: (sum(params[-3])+params[-2]*params[-1], num_class),
        'lpt_nn': lambda params: (params[-1], num_class)
    }

    model_class, param_extractor = model_mapping[model_type], method_mapping[method]
    # extract parameters for the model
    model_params = param_extractor(meth_params[:-1])

    return model_class(*model_params)


if __name__ == '__main__':
    batch_size = 4
    seq_length = 100

    x = torch.randn(batch_size, seq_length)

    # conv_block = ConvBlock()
    model = CNN2(seq_length, 10)
    output = model(x)
