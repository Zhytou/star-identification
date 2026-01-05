import torch, torchvision
import torch.nn as nn


DEBUG = False


class MIFNet(nn.Module):
    '''
        A 1D-CONV Net
        (doi:10.1109/TAES.2022.3160134)
    '''

    def __init__(self):
        super(MIFNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(11, 48, 5),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Conv1d(48, 48, 5),
            nn.ReLU(),
            nn.BatchNorm1d(48),
            nn.Conv1d(48, 48, 5),
            nn.ReLU(),
            nn.BatchNorm1d(48),

            nn.Conv1d(48, 96, 5),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, 5),
            nn.ReLU(),
            nn.BatchNorm1d(96),
             nn.Conv1d(96, 96, 5),
            nn.ReLU(),
            nn.BatchNorm1d(96),

            nn.Conv1d(96, 4956, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        y = self.net(x)
        print(y.shape)
        return y
        
    
class RPNet(nn.Module):
    '''
        A representation learning-based model
        (doi:10.1109/ACCESS.2019.2927684)
    '''
    def __init__(self):
        super(RPNet, self).__init__()

        # star pattern generator
        self.spg = nn.Sequential(
            nn.Linear(400, 512), #fc1
            nn.ReLU(),
            nn.Linear(512, 1024), #fc2
        )

        # star pattern classifier
        self.spc = nn.Sequential(
            nn.Linear(1024, 512), #fc3
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 5045), #fc4
        )

    def forward(self, x):
        xx = self.spg(x)
        y = self.spc(xx)
        return y


class SpiderWeb(nn.Module):
    '''
        A hierarchical CNN based on spider-web image
        (doi:10.1109/TAES.2019.2961826)
    '''
    def __init__(self):
        super(SpiderWeb, self).__init__()


class GridVgg(nn.Module):
    '''
        A modified vgg-16 model for star identification.
        (doi:10.1631/FITEE.1900590)
    '''
    def __init__(self):
        super(GridVgg, self).__init__()
        
        original = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            *list(original.features.children())[1:]
        )
        self.classifier = nn.Sequential(
            # 25088 -> 500
            nn.Linear(512 * 7 * 7, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            
            # 500 -> 500
            nn.Linear(500, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),
            
            # number of guide star 
            nn.Linear(500, 4897)
        )

    def forward(self, x):
        xx = self.features(x)
        xx = xx.flatten(1) # reshape(xx.shape[0], -1)
        y = self.classifier(xx)

        return y
        

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
            # output batch_size*1024*1
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        # x is composed of two input: raidal features and cyclic features
        # convert x.shape from [batch_size, num_ring+num_sector*num_neighbor]
        # into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        y = self.conv(x).squeeze(-1)

        return y


class CNN5(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):
        '''
            input_dim: the dimension of the features
            output_dim: the number of the class(guide star)
        '''

        super(CNN5, self).__init__()
        
        self.conv = nn.Sequential(
            NConvBlock(1, 48),
            NConvBlock(48, 48),
            
            NConvBlock(48, 96),
            NConvBlock(96, 96),
            NConvBlock(96, 96),

            nn.Conv1d(96, num_class, kernel_size=1),

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

        return y
    

class NConvBlock(nn.Module):
    '''
        The block for rac 1dcnn.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(NConvBlock, self).__init__()
        self.conv_5 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//3),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//3),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_dim, output_dim//3, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim//3),
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


class LightCNN(nn.Module):
    '''
        The one dimension convolutional neural network model.
    '''
    def __init__(self, num_feat: int, num_class: int):
        '''
            input_dim: the dimension of the features
            output_dim: the number of the class(guide star)
        '''

        super(LightCNN, self).__init__()
        
        self.conv = nn.Sequential(
            LightConvBlock(1, 48),
            
            LightConvBlock(48, 48),

            LightConvBlock(48, 48),
            
            LightConvBlock(48, 96),
            
            LightConvBlock(96, 96),

            LightConvBlock(96, 96),

            nn.Conv1d(96, num_class, kernel_size=1),

            # global avg pool
            # output batch_size*64*1
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        # x is composed of two input: raidal features and cyclic features
        # convert x.shape from [batch_size, num_ring+num_sector*num_neighbor]
        # into [batch_size, 1, num_ring+num_sector*num_neighbor]
        x = x.unsqueeze(1)

        # apply the convolutional layers and remove the last dimension
        y = self.conv(x).squeeze(-1)

        return y


class LightConvBlock(nn.Module):
    '''
        The block for rac 1dcnn.
    '''
    def __init__(self, input_dim: int, output_dim: int):
        super(LightConvBlock, self).__init__()
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
        y1 = self.conv_1(x)
        y2 = self.conv_3(x)

        # apply the max pooling layer
        y = self.pool_2(torch.concat((y1, y2), dim=1))

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
        'cnn5': CNN5,
        'lcnn': LightCNN,
    }

    method_mapping = {
        #!NOTE: rac_nn accept (num_ring, num_neighbor, num_sector)
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
