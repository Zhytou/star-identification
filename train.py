import os
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import StarPointDataset
from generate import num_input, num_class, point_dataset_path, config_name
from models import FeedforwardNeuralNetModel, OneDimensionConvNeuralNetModel
from test import check_accuracy


def train(model: nn.Module, optimizer: optim.Optimizer, num_epochs: int, loader: DataLoader, test_loader: DataLoader=None, device=torch.device('cpu')):
    '''
        Train the model.
    Args:
        model: the model to be trained
        optimizer: the optimizer
        num_epochs: the number of epochs
        loader: the data loader for training
        test_loader: the data loader for testing
        device: the device to run the model
    '''
    for epoch in range(num_epochs):
        # set the model into train model
        model.train()

        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)

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
            accuracy = check_accuracy(model, test_loader, device)
            print(f'Epoch: {epoch}, Accuracy: {accuracy}%')
        
        
if __name__ == '__main__':
    # training setting
    batch_size = 100
    num_epochs = 30
    learning_rate = 0.03
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # define datasets for training & validation
    train_dataset, validate_dataset, test_dataset = [StarPointDataset(os.path.join(point_dataset_path, name)) for name in ['train', 'validate', 'test']]
    # print datasets' sizes
    print(f'Training set: {len(train_dataset)}, Validation set: {len(validate_dataset)}, Test set: {len(test_dataset)}')

    # create data loaders for our datasets
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # load best model of last train
    model_dir = f'model/{config_name}/fnn'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'best_model.pth')
    if os.path.exists(model_path):
        best_model = FeedforwardNeuralNetModel(num_input, num_class)
        best_model.load_state_dict(torch.load(model_path))
        best_model.to(device)
        best_val_accuracy = check_accuracy(best_model, validate_loader, device)
        print(f'Original model accuracy {best_val_accuracy}%')

    # tune hyperparameters
    hidden_dimss = [[100, 200]]
    for hidden_dims in hidden_dimss:  
        model = FeedforwardNeuralNetModel(num_input, num_class)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)  

        print(f'train model with {hidden_dims}')
        train(model, optimizer, num_epochs, train_loader, test_loader, device)
        val_accuracy = check_accuracy(model, validate_loader, device)
        print(f'validate accurracy {val_accuracy}')

        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy
    
    test_accuray = check_accuracy(best_model, test_loader, device)
    print(f'Best model validate accuracy: {best_val_accuracy}%, test accuracy: {test_accuray}%')
    torch.save(best_model.state_dict(), model_path)
