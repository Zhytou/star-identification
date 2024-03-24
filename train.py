import os
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import StarPointDataset
from generate import num_class, sim_cfg, point_dataset_path
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

            # forward pass to get output/logits
            scores = model(points)
            # calculate Loss: softmax --> cross entropy loss
            loss = F.cross_entropy(scores, labels)
            # clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # getting gradients w.r.t. parameters
            loss.backward()
            # updating parameters
            optimizer.step()
        
        if test_loader:
            accuracy = check_accuracy(model, test_loader, device)
            print(f'Epoch: {epoch}, Accuracy: {accuracy}%')
        
        
if __name__ == '__main__':
    # training setting
    batch_size = 100
    num_epochs = 10
    learning_rate = 0.1
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # iterate different generate configs under same simulate config
    gen_cfgs = os.listdir(point_dataset_path)
    for gen_cfg in gen_cfgs:
        num_ring, num_sector, num_neighbor_limit = list(map(int, gen_cfg.split('_')))
        num_input = num_ring+num_sector*num_neighbor_limit
        # define datasets for training & validation
        train_dataset, validate_dataset, = [StarPointDataset(os.path.join(point_dataset_path, gen_cfg, type)) for type in ['train', 'validate']]
        test_dataset = StarPointDataset(os.path.join(point_dataset_path, gen_cfg, 'test', 'pos0_mv0_fs0_test'))
        # print datasets' sizes
        print(f'Training set: {len(train_dataset)}, Validation set: {len(validate_dataset)}, Test set: {len(test_dataset)}')
        # create data loaders for our datasets
        train_loader, validate_loader, test_loader = [DataLoader(dataset, batch_size, shuffle=True) for dataset in [train_dataset, validate_dataset, test_dataset]]

        # load best model of last train
        for model_type in ['fnn', '1dcnn']:
            model_dir = f'model/{sim_cfg}/{gen_cfg}/{model_type}'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if model_type == 'fnn':
                best_model = FeedforwardNeuralNetModel(num_input, num_class)
            else:
                best_model = OneDimensionConvNeuralNetModel(num_input, num_class)
            model_path = os.path.join(model_dir, 'best_model.pth')
            if os.path.exists(model_path):
                best_model.load_state_dict(torch.load(model_path))
            best_model.to(device)
            best_val_accuracy = check_accuracy(best_model, validate_loader, device)
            print(f'Original {model_type} model accuracy {best_val_accuracy}%')

            # tune hyperparameters
            hidden_dimss = [[100, 200]]
            for hidden_dims in hidden_dimss:
                if model_type == 'fnn':
                    model = FeedforwardNeuralNetModel(num_input, num_class)
                else:
                    model = OneDimensionConvNeuralNetModel(num_input, num_class)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

                print(f'train {model_type} model with {hidden_dims}')
                train(model, optimizer, num_epochs, train_loader, test_loader, device)
                val_accuracy = check_accuracy(model, validate_loader, device)
                print(f'validate accurracy {val_accuracy}')

                if val_accuracy > best_val_accuracy:
                    best_model = model
                    best_val_accuracy = val_accuracy
    
            test_accuray = check_accuracy(best_model, test_loader, device)
            print(f'Best {model_type} model validate accuracy: {best_val_accuracy}%, test accuracy: {test_accuray}%')
            torch.save(best_model.state_dict(), model_path)
