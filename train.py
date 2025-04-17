import os
import pandas as pd
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import create_dataset
from models import create_model


def check_accuracy(method: str, model: nn.Module, loader: DataLoader, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader. The accuracy is calculated based on the model's ability to correctly identify at least three stars in each star image.
    Args:
        method: the method to be checked
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

    if method == 'rac_1dcnn':
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

    elif method == 'daa_1dcnn' or method == 'lpt_nn':
        for idxs, feats, labels in loader:
            idxs, feats, labels = idxs.to(device), feats.to(device), labels.to(device)
            # forward pass only to get logits/output
            outputs = model(feats)
            # get predictions from the maximum value
            predicted = torch.argmax(outputs.data, 1)
            # correctly predicted sample indexes
            idxs = idxs[predicted == labels].tolist()

            correct += len(idxs)
            total += len(labels)
    else:
        return 0.0
    
    acc = round(100.0*correct/total, 2)
    return acc


def train(method: str, model: nn.Module, optimizer: optim.Optimizer, num_epochs: int, loader: DataLoader, val_loader: DataLoader=None, device=torch.device('cpu')):
    '''
        Train the model.
    Args:
        method: the method to be trained
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

    model.to(device)
    for epoch in range(num_epochs):
        # set the model into train model
        model.train()

        if method == 'rac_1dcnn':
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
        elif method == 'daa_1dcnn' or method == 'lpt_nn':
            for _, feats, labels in loader:
                feats, labels = feats.to(device), labels.to(device)
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
        else:
            return

        val_acc = check_accuracy(method, model, val_loader, device) if val_loader else 0.0
        print(f'Epoch: {epoch+1},  Loss: {loss.item()}, Validation Accuracy: {val_acc}%')
        
        ls.append(loss.item())
        accs.append(val_acc)

    return ls, accs


def do_train(meth_params: dict, simu_params: dict, gcata_path: str, batch_size: int=256, num_epochs: int=10, learning_rate: float=0.01, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    '''
        Train the model for different method.
    Args:
        meth_params: the parameters for the method
    '''

    # simulation config
    sim_cfg = f"{simu_params['h']}_{simu_params['w']}_{simu_params['fovx']}_{simu_params['fovy']}_{simu_params['limit_mag']}"
    
    # guide star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=["Star ID", "Ra", "De", "Magnitude"])
    num_class = len(gcata)

    # noise config
    noise_cfg = f"{simu_params['sigma_pos']}_{simu_params['sigma_mag']}_{simu_params['num_fs']}_{simu_params['num_ms']}"

    # print the training setting
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

        # define datasets for train, validate and test
        dataset_dir = os.path.join('dataset', sim_cfg, method, gen_cfg, noise_cfg)
        dataset = create_dataset(method, dataset_dir, gen_cfg)
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
        model.to(device)
            
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
    do_train(
        {
            # 'lpt_nn': [6, 50],
            'rac_1dcnn': [0, 5.5, [20, 50, 80], 16, 3],
        },
        {
            'h': 1024,
            'w': 1280,
            'fovx': 14,
            'fovy': 11,
            'limit_mag': 5.2,
            'sigma_pos': 3,
            'sigma_mag': 0.5,
            'num_fs': 7,
            'num_ms': 0
        },
        gcata_path='./catalogue/sao4.5.csv',
        num_epochs=20,
        batch_size=512,
        learning_rate=0.001
    )