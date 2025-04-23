import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import create_dataset
from model import create_model


def check_accuracy(method: str, model: nn.Module, loader: DataLoader, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader.
    '''
    # total number of samples
    tot = 0
    # count of correct predictions
    cnt = 0

    # set the model into evaluation mode
    model.eval()

    # move the model to the device
    model.to(device)

    with torch.no_grad():
        for feats, labels in loader:
            # move the features and labels to the device
            feats = (feats[0].to(device), feats[1].to(device)) if method == 'rac_1dcnn' else feats.to(device)
            labels = labels.to(device)

            # forward pass to get output logits
            scores = model(*feats) if method == 'rac_1dcnn' else model(feats)  

            # get the predicted class
            preds = torch.argmax(scores, dim=1)

            # accumulate correct and total number of samples
            cnt += (preds == labels).sum().item()
            tot += labels.size(0)

    acc = round(100.0*cnt/tot, 2)

    return acc


def train(method: str, model: nn.Module, optimizer: optim.Optimizer, num_epochs: int, loader: DataLoader, val_loader: DataLoader, device=torch.device('cpu')):
    '''
        Train the model.
    '''
    ls, accs = [], []

    # move the model to the device
    model.to(device)

    for epoch in range(num_epochs):
        # set the model into train model, because check_accuracy is called for every epoch
        model.train()

        # epoch loss
        epoch_loss = 0.0
        
        for feats, labels in loader:
            # move the features and labels to the device
            feats = (feats[0].to(device), feats[1].to(device)) if method == 'rac_1dcnn' else feats.to(device)
            labels = labels.to(device)
            
            # forward pass to get output/logits
            scores = model(*feats) if method == 'rac_1dcnn' else model(feats)
            
            # calculate Loss: softmax --> cross entropy loss
            loss = F.cross_entropy(scores, labels)

            # clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # getting gradients w.r.t. parameters
            loss.backward()
            
            # updating parameters
            optimizer.step()

            # accumulate loss
            epoch_loss += loss.item()

        # calculate average loss
        epoch_loss = epoch_loss / len(loader)

        # check accuracy on validation set
        acc = check_accuracy(method, model, val_loader, device)
        print(f'Epoch: {epoch+1}, Loss: {epoch_loss}, Validate Accuracy: {acc}%')
        
        ls.append(loss.item())
        accs.append(acc)

    return ls, accs


def do_train(meth_params: dict, simu_params: dict, gcata_path: str, batch_size: int=256, num_epochs: int=10, learning_rate: float=0.01, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    '''
        Train the model for different method.
    '''

    # simulation config
    sim_cfg = f"{simu_params['h']}_{simu_params['w']}_{simu_params['fovx']}_{simu_params['fovy']}_{simu_params['limit_mag']}"
    
    # guide star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=["Star ID", "Ra", "De", "Magnitude"])
    num_class = len(gcata)

    # print the training setting
    print('Train')
    print('-----------')
    print('Batch size:', batch_size, 'Num epochs:', num_epochs, 'Learning rate:', learning_rate)
    print('Using device:', device)
    print('Number of class', num_class)
    print('Simulation config:', sim_cfg)

    for method in meth_params:
        # generation config for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        print('Method:', method, 'Generating config:', gen_cfg)

        # define model
        model = create_model(method, meth_params[method], num_class)

        # define dataset
        df = pd.read_csv(os.path.join('dataset', sim_cfg, method, gen_cfg, 'labels.csv'))
        dataset = create_dataset(method, df, gen_cfg)
        dataset_size = len(dataset)

        # define data loaders
        train_size, val_size = int(0.8 * dataset_size), int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        train_loader, val_loader, test_loader = [DataLoader(sub_dataset, batch_size, shuffle=True) for sub_dataset in random_split(dataset, [train_size, val_size, test_size])]
        # print datasets' sizes
        print(f'Training set: {train_size}, Validation set: {val_size}, Test set: {test_size}')

        # check model directory exsitence
        model_dir = os.path.join('model', sim_cfg, method, gen_cfg)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # load best model of last train
        model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            
        # do training
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)  
        ls, accs = train(method, model, optimizer, num_epochs, train_loader, val_loader, device=device)
        
        # check accuracy on test set
        test_acc = check_accuracy(method, model, test_loader, device)
        print(f'Test Accuracy: {test_acc}%')

        # save the best model and training log
        torch.save(model.state_dict(), model_path)
        with open(os.path.join(model_dir, 'train.log'), 'a+') as f:
            f.write(f'Loss: {ls}\nAccuracy: {accs}\n')
    

if __name__ == '__main__':
    if False:
        do_train(
            {
                # 'lpt_nn': [0.5, 6, 50],
                'rac_1dcnn': [0.1, 6, [25, 50], 16, 3],
            },
            {
                'h': 512,
                'w': 512,
                'fovx': 12,
                'fovy': 12,
                'limit_mag': 6,
            },
            gcata_path='catalogue/sao6.0_d0.03_12_15.csv',
            num_epochs=30,
            batch_size=512,
            learning_rate=0.01
        )

    if True:
        do_train(
            {
                'rac_1dcnn': [0.1, 4.5, [50, 100], 16, 3],
            },
            {
                'h': 1024,
                'w': 1280,
                'fovx': 11.398822251559647,
                'fovy': 9.129887427521604,
                'limit_mag': 5.5,
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            num_epochs=200,
            batch_size=256,
            learning_rate=0.001
        )