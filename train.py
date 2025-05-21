import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generate import setup
from dataset import create_dataset
from model import create_model


def check_accuracy(model: nn.Module, loader: DataLoader, device=torch.device('cpu')):
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
            feats, labels= feats.to(device), labels.to(device)
            
            # forward pass to get output/logits
            scores = model(feats)

            # get the predicted class
            preds = torch.argmax(scores, dim=1)

            # accumulate correct and total number of samples
            cnt += (preds == labels).sum().item()
            tot += labels.size(0)

    acc = round(100.0*cnt/tot, 2)

    return acc


def train(model: nn.Module, optimizer: optim.Optimizer, num_epochs: int, loader: DataLoader, device=torch.device('cpu')):
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
            feats, labels= feats.to(device), labels.to(device)
            
            # forward pass to get output/logits
            scores = model(feats)
            
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
        acc = check_accuracy(model, loader, device)
        print(f'Epoch: {epoch+1}, Loss: {epoch_loss}, Train Accuracy: {acc}%')
        
        ls.append(epoch_loss)
        accs.append(acc)

    return ls, accs


def do_train(meth_params: dict, simu_params: dict, model_types: dict, gcata_path: str, batch_size: int=256, num_epochs: int=10, learning_rate: float=0.01, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    '''
        Train the model for different method.
    '''

    # setup
    sim_cfg, _, gcata_name, gcata = setup(simu_params, gcata_path)
    num_class = len(gcata)

    for method in meth_params:
        # generation config for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))

        # load train data
        df = pd.read_csv(os.path.join('dataset', sim_cfg, method, gen_cfg, 'labels.csv'))

        # print the training setting
        print(
            'Train',
            '\n------------------------------',
            '\nMETHOD INFO',
            '\nMethod:', method, 
            '\nSimulation config:', sim_cfg,
            '\nGeneration config:', gen_cfg,
            '\n------------------------------',
            '\nTRAIN INFO',
            '\nModel type:', model_types[method],
            '\nDataset size:', len(df),
            '\nNumber of class', num_class,
            '\nBatch size:', batch_size, 
            '\nNum epochs:', num_epochs, 
            '\nLearning rate:', learning_rate,
            '\nUsing device:', device,
            '\n------------------------------',
        )

        # define model
        model = create_model(method, model_types[method], meth_params[method], num_class)

        # define dataset
        dataset = create_dataset(method, df, meth_params[method])

        # define data loaders
        loader = DataLoader(dataset, batch_size, shuffle=True)

        # check model directory exsitence
        model_dir = os.path.join('model', sim_cfg, method, gen_cfg, model_types[method])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # load best model of last train
        model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            
        # do training
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)  
        ls, accs = train(model, optimizer, num_epochs, loader, device=device)
        
        # save the best model and training log
        torch.save(model.state_dict(), model_path)
        with open(os.path.join(model_dir, 'train.log'), 'a+') as f:
            f.write(f'Batch size: {batch_size}\nNumber of epochs: {num_epochs}\nLearning rate: {learning_rate}\nLoss: {ls}\nAccuracy: {accs}\n')
    

if __name__ == '__main__':
    if False:
        do_train(
            {
                'lpt_nn': [0.5, 6, 55, 0],
                # 'lpt_nn': [0.5, 6, 55, 1],
                # 'rac_nn': [0.5, 6, [15, 35, 55], 18, 3, 0],
                # 'rac_nn': [0.5, 6, [15, 35, 55], 18, 3, 1],
                # 'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 0],
                # 'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 1],
            },
            {
                'h': 1024,
                'w': 1282,
                'fovy': 12,
                'fovx': 14.9925,
                'limit_mag': 6,
                'rot': 1
            },
            {
                'lpt_nn': 'fnn',
                'rac_nn': 'cnn2',
            },
            gcata_path='catalogue/sao6.0_d0.03_12_15.csv',
            num_epochs=30,
            batch_size=512,
            learning_rate=0.01
        )

    if False:
        do_train(
            {
                'rac_nn': [0.1, 4.5, [25, 55, 85], 18, 3, 0],
            },
            {
                'h': 1024,
                'w': 1280,
                'fovx': 11.398822251559647,
                'fovy': 9.129887427521604,
                'limit_mag': 5.5,
                'rot': 1
            },
            {
                'rac_nn': 'cnn3'
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            num_epochs=10,
            batch_size=512,
            learning_rate=0.00001
        )
    
    if False:
        do_train(
            {
                'rac_nn': [0.1, 5.7, [25, 55, 85], 18, 3, 0],
            },
            {
                'h': 1024,
                'w': 1280,
                'fovy': 11.522621164995503,
                'fovx': 14.37611786938476,
                'limit_mag': 5.5,
                'rot': 1
            },
            {
                'rac_nn': 'cnn3'
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            num_epochs=30,
            batch_size=512,
            learning_rate=0.01
        )

    if True:
        do_train(
            {
                'rac_nn': [0.5, 7.7, [35, 75, 115], 18, 3, 0],
            },
            {
                'h': 1040,
                'w': 1288,
                'fovx': 18.97205141393946,
                'fovy': 15.36777053565561,
                'limit_mag': 6,
                'rot': 1
            },
            {
                'rac_nn': 'cnn2',
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            num_epochs=30,
            batch_size=512,
            learning_rate=0.01
        )