import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generate import database_path, pattern_path, point_dataset_path, sim_cfg, num_class
from dataset import StarPointDataset
from models import FNN, CNN


def check_pattern_match_accuracy(method: int, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30, pos_noise_std: int = 0, mv_noise_std: float = 0, num_false_star: int = 0):
    '''
        Check the accuracy of the pattern match method.
    Args:
        method: the method to generate the pattern
                1: grid algorithm
                2: radial and cyclic algorithm
        grid_len: the length of the grid
        num_ring: the number of rings
        num_sector: the number of sectors
    '''
    # wrong method
    if method > 2:
        return
    
    # patterns in database
    if method == 1:
        db = pd.read_csv(f"{database_path}/db{method}_{grid_len}x{grid_len}.csv")
    else:
        db = pd.read_csv(f"{database_path}/db{method}_{num_ring}_{num_sector}.csv")
    db_patterns = db['pattern'].values
    db_patterns = [np.array(list(map(int, pattern))) for pattern in db_patterns]

    # count the number of correct predictions & multiple match
    correct = 0
    multi_match = 0

    # iterate test cases
    if method == 1:
        test = pd.read_csv(f"{pattern_path}/db{method}_{grid_len}x{grid_len}_pos{pos_noise_std}_mv{mv_noise_std}_fs{num_false_star}_test/test.csv")
    else:
        test = pd.read_csv(f"{pattern_path}/db{method}_{num_ring}_{num_sector}_pos{pos_noise_std}_mv{mv_noise_std}_fs{num_false_star}_test/test.csv")
    for i in range(len(test)):
        pattern = test.loc[i, 'pattern']
        id = test.loc[i, 'id']
        pattern = np.array(list(map(int, pattern)))

        # find the closest pattern in the database
        match_result = np.sum(pattern&db_patterns, axis=1)
        max_val = np.max(match_result)
        idxs = np.where(match_result == max_val)[0]
        if id in db.loc[idxs, 'id'].values:
            if len(idxs) == 1:
                correct += 1
            else:
                multi_match += 1

    return 100.0*correct/len(test)


def check_accuracy(model: nn.Module, loader: DataLoader, device=torch.device('cpu')):
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
    for rings, sectors, labels in loader:
        rings, sectors, labels = rings.to(device), sectors.to(device), labels.to(device)
        # forward pass only to get logits/output
        outputs = model(rings, sectors)
        # get predictions from the maximum value
        predicted = torch.argmax(outputs.data, 1)
        # total number of labels
        total += labels.size(0)
        # total correct predictions
        correct += (predicted == labels).sum().item()
    return round(100.0 * correct / total, 2)


if __name__ == '__main__':
    batch_size = 100
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # load best model
    gen_cfgs = os.listdir(point_dataset_path)
    for gen_cfg in gen_cfgs:
        num_ring, num_sector, num_neighbor_limit = list(map(int, gen_cfg.split('_')))
        num_input = num_ring+num_sector*num_neighbor_limit
        # iterate different model types
        for model_type in ['fnn', '1dcnn']:
            model_path = f'model/{sim_cfg}/{gen_cfg}/{model_type}/best_model.pth'
            if not os.path.exists(model_path):
                print(f'{model_path} does not exist!')
                continue
            if model_type == 'fnn':
                best_model = FNN(num_input, num_class)
            else:
                best_model = CNN(num_input, num_class)
            best_model.load_state_dict(torch.load(model_path))
            best_model.to(device)

            pos_accs, mv_accs, fs_accs = [], [], []
            # define pos_noise_test datasets
            for pos_noise_std in [0, 1, 2]:
                pos_test_dataset = StarPointDataset(os.path.join(point_dataset_path, gen_cfg, 'test', f'pos{pos_noise_std}_mv0_fs0_test'))
                # print datasets' sizes
                print(f'Positional noise set: {len(pos_test_dataset)}')
                # define data loaders
                pos_test_loader = DataLoader(pos_test_dataset, batch_size)
                pos_acc = check_accuracy(best_model, pos_test_loader, device)
                pos_accs.append(pos_acc)

            # define mv_noise_test datasets
            for mv_noise_std in [0, 0.1, 0.2]:
                mv_test_dataset = StarPointDataset(os.path.join(point_dataset_path, gen_cfg, 'test', f'pos0_mv{mv_noise_std}_fs0_test'))
                # print datasets' sizes
                print(f'Magnitude noise set: {len(mv_test_dataset)}')
                # define data loaders
                mv_test_loader = DataLoader(mv_test_dataset, batch_size)
                mv_acc = check_accuracy(best_model, mv_test_loader, device)
                mv_accs.append(mv_acc)

            # define false_star_test datasets
            for num_false_star in [0, 1, 2, 3, 4, 5]:
                fs_test_dataset = StarPointDataset(os.path.join(point_dataset_path, gen_cfg, 'test', f'pos0_mv0_fs{num_false_star}_test'))
                # print datasets' sizes
                print(f'False star set: {len(fs_test_dataset)}')
                # define data loaders
                fs_test_loader = DataLoader(fs_test_dataset, batch_size)
                fs_acc = check_accuracy(best_model, fs_test_loader, device)   
                fs_accs.append(fs_acc)
            
            print(pos_accs, mv_accs, fs_accs)
            
    # conventional pattern match methods' accuracy
    for method in [1, 2]:
        pos_accs, mv_accs, fs_accs = [], [], []
        for pos_noise_std in [0, 1, 2]:
            acc = check_pattern_match_accuracy(method, grid_len=60, pos_noise_std=pos_noise_std)
            pos_accs.append(acc)
        
        for mv_noise_std in [0, 0.1, 0.2]:
            acc = check_pattern_match_accuracy(method, grid_len=60, mv_noise_std=mv_noise_std)
            mv_accs.append(acc)

        for num_false_star in [0, 1, 2, 3, 4, 5]:
            acc = check_pattern_match_accuracy(method, grid_len=60, num_false_star=num_false_star)
            fs_accs.append(acc)
        
        print(pos_accs, mv_accs, fs_accs)

