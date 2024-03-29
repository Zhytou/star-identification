import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generate import database_path, pattern_path, dataset_path, sim_cfg, num_class
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


def check_accuracy(model: nn.Module, loader: DataLoader, df: pd.Series, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader. The accuracy is calculated based on the model's ability to correctly identify at least three stars in each star image.
    Args:
        model: the model to be checked
        loader: the data loader for validation or testing
        df: the series that maps sample indices to image ids
        device: the device to run the model
    Returns:
        the accuracy of the model
    '''
    # set the model into evaluation model
    model.eval()
    # dict to store the number of correctly predicted samples for each star image
    freqs = {}

    # iterate through test dataset
    for idxs, rings, sectors, labels in loader:
        idxs, rings, sectors, labels = idxs.to(device), rings.to(device), sectors.to(device), labels.to(device)
        
        # forward pass only to get logits/output
        outputs = model(rings, sectors)
        # get predictions from the maximum value
        predicted = torch.argmax(outputs.data, 1)
        # correctly predicted sample indexes
        idxs = idxs[predicted == labels].tolist()
        df[idxs].apply(lambda x: freqs.update({x: freqs.get(x, 0)+1}))

    # the number of star images that have at least three correctly predicted samples
    cnt = sum(v >= 3 for v in freqs.values())
    
    

    return round(100.0*cnt/len(freqs), 2)


if __name__ == '__main__':
    batch_size = 100
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # load best model
    gen_cfgs = os.listdir(dataset_path)
    for gen_cfg in gen_cfgs:
        num_ring, num_sector, num_neighbor = list(map(int, gen_cfg.split('_')))
        num_input = num_ring+num_sector*num_neighbor
        # iterate different model types
        for model_type in ['fnn', '1dcnn']:
            model_path = f'model/{sim_cfg}/{gen_cfg}/{model_type}/best_model.pth'
            if not os.path.exists(model_path):
                print(f'{model_path} does not exist!')
                continue
            if model_type == 'fnn':
                best_model = FNN(num_input, num_class)
            else:
                best_model = CNN(num_ring, (num_neighbor, num_sector), num_class)
            best_model.load_state_dict(torch.load(model_path))
            best_model.to(device)

            pos_accs, mv_accs, fs_accs = [], [], []
            # define pos_noise_test datasets
            for pns in [1, 2]:
                pos_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', f'pos{pns}'))
                pos_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', f'pos{pns}', 'labels.csv'))
                # print datasets' sizes
                print(f'Positional noise set: {len(pos_dataset)}')
                # define data loaders
                pos_loader = DataLoader(pos_dataset, batch_size)
                pos_acc = check_accuracy(best_model, pos_loader, pos_df['img_id'], device)
                pos_accs.append(pos_acc)

            # define mv_noise_test datasets
            for mns in [0.1, 0.2]:
                mv_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', f'mv{mns}'))
                mv_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', f'mv{mns}', 'labels.csv'))  
                # print datasets' sizes
                print(f'Magnitude noise set: {len(mv_dataset)}')
                # define data loaders
                mv_loader = DataLoader(mv_dataset, batch_size)
                mv_acc = check_accuracy(best_model, mv_loader, mv_df['img_id'], device)
                mv_accs.append(mv_acc)

            # define false_star_test datasets
            for nfs in [1, 2, 3, 4, 5]:
                fs_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', f'fs{nfs}'))
                fs_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', f'fs{nfs}', 'labels.csv'))
                # print datasets' sizes
                print(f'False star set: {len(fs_dataset)}')
                # define data loaders
                fs_loader = DataLoader(fs_dataset, batch_size)
                fs_acc = check_accuracy(best_model, fs_loader, fs_df['img_id'], device)   
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

