import os
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import StarPointDataset
from generate import star_num_per_sample, num_classes, dataset_root_path, point_dataset_sub_path


class FeedforwardNeuralNetModel(nn.Module):
    '''
        The feedforward neural network model. By the way, model structure refers to https://www.mdpi.com/1424-8220/20/13/3684.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = [300, 100]):
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


def train(model: nn.Module, optimizer: optim.Optimizer, num_epochs: int, loader: DataLoader, test_loader: DataLoader=None):
    '''
        Train the model.
    Args:
        model: the model to be trained
        optimizer: the optimizer
        num_epochs: the number of epochs
        loader: the data loader for training
        test_loader: the data loader for testing
    '''
    for epoch in range(num_epochs):
        # set the model into train model
        model.train()

        for points, labels in loader:
            # points.to(device)
            # labels.to(device)
            # print(points.shape, points.device, points.is_cuda)
            # print(next(model.parameters()).device) 

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            scores = model(points)
            # Calculate Loss: softmax --> cross entropy loss
            loss = F.cross_entropy(scores, labels)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        
        if test_loader:
            accuracy = check_accuracy(model, test_loader)
            print(f'Epoch: {epoch}, Accuracy: {accuracy}%')
        
        
def check_accuracy(model: nn.Module, loader: DataLoader):
    '''
        Check the accuracy of the model.
    Args:
        model: the model to be checked
        loader: the data loader for validation or testing
    '''
    # set the model into evaluation model
    model.eval()
    # initialize the number of correct predictions
    correct = 0
    total = 0
    # Iterate through test dataset
    for points, labels in loader:
        # points.to(device)
        # labels.to(device)
        # Forward pass only to get logits/output
        outputs = model(points)
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        # Total number of labels
        total += labels.size(0)
        # Total correct predictions
        correct += (predicted == labels).sum().item()
    return round(100.0 * correct / total, 2)


if __name__ == '__main__':
    # training setting
    batch_size = 100
    num_epochs = 15
    learning_rate = 0.1
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # define datasets for training & validation
    dataset_path = os.path.join(dataset_root_path, point_dataset_sub_path)
    train_dataset, validate_dataset, test_dataset = [StarPointDataset(os.path.join(dataset_path, name)) for name in ['train', 'validate', 'test']]
    # print datasets' sizes
    print(f'Training set: {len(train_dataset)}, Validation set: {len(validate_dataset)}, Test set: {len(test_dataset)}')

    # create data loaders for our datasets
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # load old model
    best_model = FeedforwardNeuralNetModel(star_num_per_sample*2, num_classes)
    best_model.load_state_dict(torch.load('model/fnn_model.pth'))
    best_accuracy = check_accuracy(best_model, validate_loader)
    print(f'Original model accuracy {best_accuracy}%')

    # tune hyperparameters
    hidden_dimss = [[10, 20]]
    for hidden_dims in hidden_dimss:  
        model = FeedforwardNeuralNetModel(star_num_per_sample*2, num_classes)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)  

        print(f'train model with {hidden_dims}')
        train(model, optimizer, num_epochs, train_loader, test_loader)
        val_accuracy = check_accuracy(model, validate_loader)
        print(f'validate accurracy {val_accuracy}')

        if val_accuracy > best_accuracy:
            best_model = model
            best_accuracy = val_accuracy
    
    print(f'Best model validate accuracy: {best_accuracy}%')
    test_accuray = check_accuracy(best_model, test_loader)
    print(f'Best model test accuracy: {test_accuray}%')
    torch.save(best_model.state_dict(), 'model/fnn_model.pth')
