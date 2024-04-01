import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from generate import database_path, pattern_path, dataset_path, sim_cfg, num_class
from dataset import StarPointDataset
from models import FNN, CNN


def check_pm_accuracy(db: pd.DataFrame, df: pd.DataFrame, method:int=1):
    '''
        Evaluate the pattern match method's accuracy on the provided patterns. The accuracy is calculated based on the method's ability to correctly identify the closest pattern in the database.
    Args:
        db: the database of patterns
        df: the patterns to be tested
    Returns:
        the accuracy of the pattern match method
    '''
    # convert patterns to numpy array
    db_patterns = db['pattern'].values
    db_patterns = np.array([np.array(list(map(int, pattern))) for pattern in db_patterns])

    # number of multi-match samples
    multi_match = 0 
    # num of correctly predicted samples
    correct = 0

    # dict to store the number of correctly predicted samples for each star image
    freqs = {}

    for i in range(len(df)):
        img_id, pattern, id = df.loc[i, ['img_id', 'pattern', 'id']]
        pattern = np.array(list(map(int, pattern)))

        # find the closest pattern in the database
        match_result = np.sum(pattern & db_patterns, axis=1)
        max_val = np.max(match_result)
        idxs = np.where(match_result == max_val)[0]
        if id in db.loc[idxs, 'id'].values:
            if len(idxs) == 1:
                if method == 1:
                    correct += 1
                else:
                    freqs.update({img_id: freqs.get(img_id, 0)+1})
            else:
                multi_match += 1

    acc = 0
    if method == 1:
        acc = round(100.0*correct/len(df), 2)
    else: 
        # the number of star images that have at least three correctly predicted samples
        cnt = sum(v >= 3 for v in freqs.values())
        acc = round(100.0*cnt/len(freqs), 2)

    return acc


def check_nn_accuracy(model: nn.Module, loader: DataLoader, df: pd.Series=None, device=torch.device('cpu')):
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
        if df != None:
            df[idxs].apply(lambda x: freqs.update({x: freqs.get(x, 0)+1}))
        
        correct += len(idxs)
        total += len(labels)

    acc = 0

    if df != None:
        # the number of star images that have at least three correctly predicted samples
        cnt = sum(v >= 3 for v in freqs.values())
        acc = round(100.0*cnt/len(freqs), 2)
    else:
        acc = round(100.0*correct/total, 2)

    return acc


if __name__ == '__main__':
    test_pm, test_nn = False, True
    
    # conventional pattern match method accuracy
    if test_pm:
        gen_cfgs = os.listdir(pattern_path)
        for gen_cfg in gen_cfgs:
            db = pd.read_csv(os.path.join(database_path, gen_cfg, 'db.csv'))
            test = pd.read_csv(os.path.join(pattern_path, gen_cfg, 'default', 'patterns.csv'))
            defualt_acc = check_pm_accuracy(db, test)
            
            pos_accs, mv_accs, fs_accs = [], [], []
            for pns in [0.5, 1, 1.5, 2]:
                test = pd.read_csv(os.path.join(pattern_path, gen_cfg, f'pos{pns}', 'patterns.csv'))
                acc = check_pm_accuracy(db, test)
                pos_accs.append(acc)
            
            for mns in [0.1, 0.2]:
                test = pd.read_csv(os.path.join(pattern_path, gen_cfg, f'mv{mns}', 'patterns.csv'))
                acc = check_pm_accuracy(db, test)
                mv_accs.append(acc)

            for nfs in [1, 2, 3, 4, 5]:
                test = pd.read_csv(os.path.join(pattern_path, gen_cfg, f'fs{nfs}', 'patterns.csv'))
                acc = check_pm_accuracy(db, test)
                fs_accs.append(acc)
            
            print(gen_cfg, defualt_acc, pos_accs, mv_accs, fs_accs)

    # nn model accuracy
    if test_nn:
        batch_size = 100
        # use gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}')
        gen_cfgs = os.listdir(dataset_path)
        for gen_cfg in gen_cfgs:
            _, num_ring, num_sector, num_neighbor = list(map(int, gen_cfg.split('_')))
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
                # load best model
                best_model.load_state_dict(torch.load(model_path))
                best_model.to(device)

                # default_test accuracy
                default_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', 'default'))
                defualt_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', 'default', 'labels.csv'))
                default_loader = DataLoader(default_dataset, batch_size)
                defualt_acc = check_nn_accuracy(best_model, default_loader, device=device)

                pos_accs, mv_accs, fs_accs = [], [], []
                # pos_noise_test accuracy
                for pns in [0.5, 1, 1.5, 2, 3, 5]:
                    pos_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', f'pos{pns}'))
                    pos_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', f'pos{pns}', 'labels.csv'))
                    # print datasets' sizes
                    print(f'Positional noise set: {len(pos_dataset)}')
                    # define data loaders
                    pos_loader = DataLoader(pos_dataset, batch_size)
                    pos_acc = check_nn_accuracy(best_model, pos_loader, device=device)
                    pos_accs.append(pos_acc)

                # mv_noise_test accuracy
                for mns in [0.1, 0.2]:
                    mv_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', f'mv{mns}'))
                    mv_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', f'mv{mns}', 'labels.csv'))  
                    # print datasets' sizes
                    print(f'Magnitude noise set: {len(mv_dataset)}')
                    # define data loaders
                    mv_loader = DataLoader(mv_dataset, batch_size)
                    mv_acc = check_nn_accuracy(best_model, mv_loader, device=device)
                    mv_accs.append(mv_acc)

                # false_star_test accuracy
                for nfs in [1, 2, 3, 4, 5]:
                    fs_dataset = StarPointDataset(os.path.join(dataset_path, gen_cfg, 'test', f'fs{nfs}'))
                    fs_df = pd.read_csv(os.path.join(dataset_path, gen_cfg, 'test', f'fs{nfs}', 'labels.csv'))
                    # print datasets' sizes
                    print(f'False star set: {len(fs_dataset)}')
                    # define data loaders
                    fs_loader = DataLoader(fs_dataset, batch_size)
                    fs_acc = check_nn_accuracy(best_model, fs_loader, device=device)   
                    fs_accs.append(fs_acc)
                
                print(gen_cfg, defualt_acc, pos_accs, mv_accs, fs_accs)


