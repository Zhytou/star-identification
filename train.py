import os
import pandas as pd
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import StarPointDataset
from generate import num_class, sim_cfg, dataset_path
from models import FNN, CNN


def weights_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)


def check_accuracy(model: nn.Module, loader: DataLoader, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader. The accuracy is calculated based on the model's ability to correctly identify at least three stars in each star image.
    Args:
        model: the model to be checked
        loader: the data loader for validation or testing
        device: the device to run the model
    Returns:
        the accuracy of the model
    '''
    # set the model into evaluation model
    model.eval()

    # num of correctly predicted samples
    correct = 0
    total = 0

    # iterate through test dataset
    for idxs, rings, sectors, labels in loader:
        idxs, rings, sectors, labels = idxs.to(device), rings.to(device), sectors.to(device), labels.to(device)
        
        # forward pass only to get logits/output
        outputs = model(rings, sectors)
        # get predictions from the maximum value
        predicted = torch.argmax(outputs.data, 1)
        # correctly predicted sample indexes
        idxs = idxs[predicted == labels].tolist()
        
        correct += len(idxs)
        total += len(labels)

    acc = round(100.0*correct/total, 2)

    return acc


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
    Returns:
        ls: the loss of each epoch
        accs: the accuracy of each epoch
    '''
    ls, accs = [], []
    for epoch in range(num_epochs):
        # set the model into train model
        model.train()

        for _, rings, sectors, labels in loader:
            rings, sectors, labels = rings.to(device), sectors.to(device), labels.to(device)

            # forward pass to get output/logits
            scores = model(rings, sectors)
            # calculate Loss: softmax --> cross entropy loss
            loss = F.cross_entropy(scores, labels)
            # clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # getting gradients w.r.t. parameters
            loss.backward()
            # updating parameters
            optimizer.step()
        
        ls.append(loss.item())

        if test_loader:
            accuracy = check_accuracy(model, test_loader, device)
            print(f'Epoch: {epoch+1}, Accuracy: {accuracy}%')
            accs.append(accuracy)

    with open(os.path.join(model_dir, 'train.log'), 'a+') as f:
        f.write(f'Loss: {ls}\nAccuracy: {accs}\n')


if __name__ == '__main__':
    # training setting
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # iterate different generate configs under same simulate config
    gen_cfgs = os.listdir(dataset_path)
    for gen_cfg in gen_cfgs:
        if gen_cfg != 'SAO5.6_15_20_0_6_50_16_3':
            continue
        print(f'Generate config: {gen_cfg}')
        num_ring, num_sector, num_neighbor = list(map(int, gen_cfg.split('_')[-3:]))
        num_input = num_ring+num_sector*num_neighbor
        # define datasets for train validate and test
        train_dataset, val_dataset, test_dataset = [StarPointDataset(os.path.join(dataset_path, gen_cfg, type), gen_cfg) for type in ['train', 'validate', 'test']]
        # print datasets' sizes
        print(f'Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}, Test set: {len(test_dataset)}')
        # create data loaders for our datasets
        train_loader, val_loader, test_loader = [DataLoader(dataset, batch_size, shuffle=True) for dataset in [train_dataset, val_dataset, test_dataset]]

        # load best model of last train
        model_dir = f'model/{sim_cfg}/{gen_cfg}/1dcnn'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model = CNN(num_ring, (num_neighbor, num_sector), num_class)
        model_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(model_path):
          model.apply(weights_init)  
        else:
            model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)  
        train(model, optimizer, num_epochs, train_loader, test_loader, device=device)
        torch.save(model.state_dict(), model_path)
    