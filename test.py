import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simulate import get_neighbor_num
from generate import database_path, test_path, sim_cfg, num_class
from dataset import StarPointDataset
from models import CNN


def check_pm_accuracy(method: str, db: pd.DataFrame, df: pd.DataFrame):
    '''
        Evaluate the pattern match method's accuracy on the provided patterns. The accuracy is calculated based on the method's ability to correctly identify the closest pattern in the database.
    Args:
        method: the pattern match method
        db: the database of patterns
        df: the patterns to be tested
    Returns:
        the accuracy of the pattern match method
    '''
    
    # convert str patterns to list
    db_patterns = db['pattern'].values
    db_patterns = [list(map(int, pattern.split(' '))) for pattern in db_patterns]
    
    # preprocess the db patterns based on method
    if method == 'pm1':
        # fill -1 to make sure each db_pattern has the same length so that they can be stored in np.array
        max_len = max(map(len, db_patterns))
        db_patterns = np.array([pattern+[-1]*(max_len-len(pattern)) for pattern in db_patterns])
    elif method == 'pm2':
        db_patterns = np.array(db_patterns)
    else:
        print(f'Invalid method {method}')
        return

    # dict to store the number of correctly predicted samples for each star image
    freqs = {}

    for i in range(len(df)):
        img_id, pattern, star_id = df.loc[i, ['img_id', 'pattern', 'id']]
        pattern = np.array(list(map(int, pattern.split(' '))))

        if method == 'pm1':
            # find the closest pattern in the database
            res = np.sum(np.isin(db_patterns, pattern), axis=1)
            idxs = np.where(res == np.max(res))[0]
            if len(idxs) == 1 and star_id == db.loc[idxs[0], 'id']:
                freqs.update({img_id: freqs.get(img_id, 0)+1})
        else:
            # initial match based on radial features
            res1 = np.sum(pattern[:-8] & db_patterns[:, :-8], axis=1)
            idxs1 = np.where(res1 == np.max(res1))[0]
            # do follow-up matches only if initial match success
            if star_id not in db.loc[idxs1, 'id'].values:
                continue
            if len(idxs1) == 1:
                freqs.update({img_id: freqs.get(img_id, 0)+1})
                continue
            # cyclic match
            res2 = np.sum(pattern[-8:] & db_patterns[:, -8:], axis=1)
            idxs2 = np.where(res2 == np.max(res2))[0]
            idxs = np.intersect1d(idxs1, idxs2)        
            if len(idxs) == 1 and star_id == db.loc[idxs[0], 'id']:
                freqs.update({img_id: freqs.get(img_id, 0)+1})
                continue
            # FOV constraint
            ids = db.loc[idxs, 'id'].values
            if star_id not in ids:
                continue
            # remove the candidates that do not have enough neighbor stars
            nums = [get_neighbor_num(id) for id in ids]
            id = ids[np.argmax(nums)]
            if star_id == id:
                freqs.update({img_id: freqs.get(img_id, 0)+1})


    # the number of star images that have at least three correctly predicted samples
    cnt = sum(v >= 3 for v in freqs.values())
    test_info = df['img_id'].value_counts()
    tot = len(test_info)-len(test_info[test_info<3])
    acc = round(100.0*cnt/tot, 2) if tot > 0 else 0

    return acc


def check_nn_accuracy(model: nn.Module, loader: DataLoader, img_ids: pd.Series=None, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader. The accuracy is calculated based on the model's ability to correctly identify at least three stars in each star image.
    Args:
        model: the model to be checked
        loader: the data loader for validation or testing
        img_ids: the series that maps sample indices to image ids
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
        # idxs, rings, sectors, labels = idxs.to(device), rings.to(device), sectors.to(device), labels.to(device)
        
        # forward pass only to get logits/output
        outputs = model(rings, sectors)
        # get predictions from the maximum value
        predicted = torch.argmax(outputs.data, 1)
        # correctly predicted sample indexes
        idxs = idxs[predicted == labels].tolist()
        # update number of successfully predicted star images
        img_ids[idxs].apply(lambda x: freqs.update({x: freqs.get(x, 0)+1}))

    # the percentage of star images that have at least three correctly predicted samples
    test_info = img_ids.value_counts()
    cnt = sum(v >= 3 for v in freqs.values())
    tot = len(test_info) - len(test_info[test_info<3])
    acc = round(100.0*cnt/tot, 2)

    return acc


if __name__ == '__main__':
    test_pm, test_nn = True, False
    
    # conventional pattern match method accuracy
    if test_pm:
        methods = ['pm1']
        for method in methods:
            gen_cfgs = os.listdir(os.path.join(test_path, method))
            for gen_cfg in gen_cfgs:
                db = pd.read_csv(os.path.join(database_path, gen_cfg, f'{method}.csv'))
                test = pd.read_csv(os.path.join(test_path, method, gen_cfg, 'default', 'labels.csv'))
                default_acc = check_pm_accuracy(method, db, test)
                
                pos_accs, mv_accs, fs_accs = [], [], []
                for pns in [1, 2, 3, 4]:
                    test = pd.read_csv(os.path.join(test_path, method, gen_cfg, f'pos{pns}', 'labels.csv'))
                    acc = check_pm_accuracy(method, db, test)
                    pos_accs.append(acc)
                
                for mns in [0.05, 0.1, 0.15, 0.2]:
                    test = pd.read_csv(os.path.join(test_path, method, gen_cfg, f'mv{mns}', 'labels.csv'))
                    acc = check_pm_accuracy(method, db, test)
                    mv_accs.append(acc)

                for nfs in [1, 2, 3, 4, 5]:
                    test = pd.read_csv(os.path.join(test_path, method, gen_cfg, f'fs{nfs}', 'labels.csv'))
                    acc = check_pm_accuracy(method, db, test)
                    fs_accs.append(acc)
            
                print(method, gen_cfg, default_acc, pos_accs, mv_accs, fs_accs)

    # nn model accuracy
    if test_nn:
        batch_size = 100
        # use gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        gen_cfgs = os.listdir(os.path.join(test_path, 'nn'))
        for gen_cfg in gen_cfgs:
            # parse gen_cfg
            num_ring, num_sector, num_neighbor = list(map(int, gen_cfg.split('_')[-3:]))
            num_input = num_ring+num_sector*num_neighbor

            # load best model
            model_path = f'model/{sim_cfg}/{gen_cfg}/1dcnn/best_model.pth'            
            best_model = CNN(num_ring, (num_neighbor, num_sector), num_class)
            best_model.load_state_dict(torch.load(model_path))
            # best_model.to(device)

            # default_test accuracy
            default_dataset = StarPointDataset(os.path.join(test_path, 'nn', gen_cfg, 'default'), gen_cfg)
            defualt_df = pd.read_csv(os.path.join(test_path, 'nn', gen_cfg, 'default', 'labels.csv'))
            default_loader = DataLoader(default_dataset, batch_size)
            defualt_acc = check_nn_accuracy(best_model, default_loader, defualt_df['img_id'], device=device)

            pos_accs, mv_accs, fs_accs = [], [], []
            # pos_noise_test accuracy
            for pns in [1, 2, 3, 4]:
                pos_dataset = StarPointDataset(os.path.join(test_path, 'nn', gen_cfg, f'pos{pns}'), gen_cfg)
                pos_df = pd.read_csv(os.path.join(test_path, 'nn', gen_cfg, f'pos{pns}', 'labels.csv'))
                # define data loaders
                pos_loader = DataLoader(pos_dataset, batch_size)
                pos_acc = check_nn_accuracy(best_model, pos_loader, pos_df['img_id'], device=device)
                pos_accs.append(pos_acc)

            # mv_noise_test accuracy
            for mns in [0.05, 0.1, 0.15, 0.2]:
                mv_dataset = StarPointDataset(os.path.join(test_path, 'nn', gen_cfg, f'mv{mns}'), gen_cfg)
                mv_df = pd.read_csv(os.path.join(test_path, 'nn', gen_cfg, f'mv{mns}', 'labels.csv'))  
                # define data loaders
                mv_loader = DataLoader(mv_dataset, batch_size)
                mv_acc = check_nn_accuracy(best_model, mv_loader, mv_df['img_id'], device=device)
                mv_accs.append(mv_acc)

            # false_star_test accuracy
            for nfs in [1, 2, 3, 4, 5]:
                fs_dataset = StarPointDataset(os.path.join(test_path, 'nn', gen_cfg, f'fs{nfs}'), gen_cfg)
                fs_df = pd.read_csv(os.path.join(test_path, 'nn', gen_cfg, f'fs{nfs}', 'labels.csv'))
                # define data loaders
                fs_loader = DataLoader(fs_dataset, batch_size)
                fs_acc = check_nn_accuracy(best_model, fs_loader, fs_df['img_id'], device=device)   
                fs_accs.append(fs_acc)
            
            print('nn', gen_cfg, defualt_acc, pos_accs, mv_accs, fs_accs)

