import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from generate import database_path, test_path, sim_cfg, num_class
from dataset import LPTDataset, RACDataset, DAADataset
from models import RAC_CNN, DAA_CNN, FNN


def cal_match_score(pat1: np.ndarray, pat2: np.ndarray, L: int, K: int=0, sim: float=0.0):
    '''
        Calculate the match score between two patterns based on the specified similarity metric.
    Args:
        pat1/2: pattern vector, each element is the index of 1 value in the 0-1 pattern matrix
        L: the length of 0-1 pattern matrix
        K: the search window size for soft matching
        sim: the similarity coefficient for soft matching, if similarity == 0, then use hard matching
    Returns:
        the match score between the two patterns
    '''

    # convert the pattern index to coordinates
    rows1, cols1 = pat1//L, pat1%L
    rows2, cols2 = pat2//L, pat2%L    

    # calculate the difference between the two patterns
    rows1, cols1 = rows1[:, np.newaxis], cols1[:, np.newaxis]
    rows_diff, cols_diff = np.abs(rows1-rows2), np.abs(cols1-cols2)

    score = 0

    # exact match | hard match
    exact_match = (rows_diff == 0) & (cols_diff == 0)
    # calculate the score
    score += np.sum(exact_match)

    # close match | soft match
    if K > 0 and sim > 0:
        # normalized L2 distance from pat1 to pat2
        dist = np.sqrt(rows_diff**2 + cols_diff**2)/K
        close_match = ((dist <= 1) & (dist > 0)).astype(float)
        if True:
            # nonlinear attenuation
            atten = np.exp(-dist**2)
        else:
            atten = 1-dist
        # calculate the score
        score += np.sum(close_match*atten)*sim

    return score


def check_pm_accuracy(method: str, db: pd.DataFrame, df: pd.DataFrame, L: int, K: int=3, sim: float=0.8):
    '''
        Evaluate the pattern match method's accuracy on the provided patterns. The accuracy is calculated based on the method's ability to correctly identify the closest pattern in the database.
    Args:
        method: the pattern match method
        db: the database of patterns
        df: the patterns to be tested
    Returns:
        the accuracy of the pattern match method
    '''

    if method != 'grid' and method != 'lpt':
        return 0.0
    
    # dict to store the number of correctly predicted samples for each star image
    freqs = {}

    db_pats = db['pattern'].to_numpy()
    db_pats = [np.array(pat.split(' '), dtype=int) for pat in db_pats]
    
    for i in range(len(df)):
        img_id, pat, star_id = df.loc[i, ['img_id', 'pattern', 'id']]
        pat = np.array(pat.split(' '), dtype=int)
        
        # find the closest pattern in the database
        res = [cal_match_score(pat, db_pat, L, 2, 0.7) for db_pat in db_pats]
        idxs = np.where(res == np.max(res))[0]
        if method == 'grid' and np.max(res) >= 5 and len(idxs) == 1 and star_id == db.loc[idxs[0], 'id']:
            freqs.update({img_id: freqs.get(img_id, 0)+1})
        elif method == 'lpt' and np.max(res) >= 6 and len(idxs) == 1 and star_id == db.loc[idxs[0], 'id']:
            freqs.update({img_id: freqs.get(img_id, 0)+1})

    # the number of star images that have at least three correctly predicted samples
    cnt = sum(v >= 3 for v in freqs.values())
    test_info = df['img_id'].value_counts()
    tot = len(test_info)-len(test_info[test_info<3])
    acc = round(100.0*cnt/tot, 2) if tot > 0 else 0

    return acc


def check_nn_accuracy(method: str, model: nn.Module, loader: DataLoader, img_ids: pd.Series=None, device=torch.device('cpu')):
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

    if method == 'rac_1dcnn':
        for idxs, rings, sectors, labels in loader:
            idxs, rings, sectors, labels = idxs.to(device), rings.to(device), sectors.to(device), labels.to(device)
            # forward pass only to get logits/output
            outputs = model(rings, sectors)
            # get predictions from the maximum value
            predicted = torch.argmax(outputs.data, 1)
            # correctly predicted sample indexes
            idxs = idxs[predicted == labels].tolist()
            # update number of successfully predicted star images
            img_ids[idxs].apply(lambda x: freqs.update({x: freqs.get(x, 0)+1}))
    elif method == 'daa_1dcnn' or method == 'lpt_nn':
        for idxs, feats, labels in loader:
            idxs, feats, labels = idxs.to(device), feats.to(device), labels.to(device)
            # forward pass only to get logits/output
            outputs = model(feats)
            # get predictions from the maximum value
            predicted = torch.argmax(outputs.data, 1)
            # correctly predicted sample indexes
            idxs = idxs[predicted == labels].tolist()
            # update number of successfully predicted star images
            img_ids[idxs].apply(lambda x: freqs.update({x: freqs.get(x, 0)+1}))
    else:
        return 0.0

    # the percentage of star images that have at least three correctly predicted samples
    test_info = img_ids.value_counts()
    cnt = sum(v >= 4 for v in freqs.values())
    tot = len(test_info) - len(test_info[test_info<4])
    acc = round(100.0*cnt/tot, 2)

    return acc


if __name__ == '__main__':
    res = {}

    # conventional pattern match method accuracy
    for method in ['grid']:  
        res[method] = {}
        for gen_cfg in os.listdir(os.path.join(test_path, method)):
            # parse gen_cfg
            L = int(gen_cfg.split('_')[-1])

            db = pd.read_csv(os.path.join(database_path, gen_cfg, f'{method}.csv'))
            for s in os.listdir(os.path.join(test_path, method, gen_cfg)):
                # use regex to parse test parameters
                match = re.match('(pos|mag|fs)([0-9]+\.?[0-9]*)', s)
                if match is None:
                    name, x = 'default', 0
                else:
                    name, x = match.groups()
                    x = float(x)
                # calculate accuracy
                df = pd.read_csv(os.path.join(test_path, method, gen_cfg, s, 'labels.csv'))
                y = check_pm_accuracy(method, db, df, L)
                if name == 'default':
                    res[method]['default'] = y
                    continue
                if name not in res[method].keys():
                    res[method][name] = [(x, y)]
                else:
                    res[method][name].append((x, y))
            
    # nn model accuracy
    for method in []:
        res[method] = {}
        batch_size = 100
        # use gpu if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for gen_cfg in os.listdir(os.path.join(test_path, method)):
            # parse gen_cfg
            if method == 'rac_1dcnn':
                arr_nr, num_sector, num_neighbor = gen_cfg.split('_')[-3:]
                num_sector, num_neighbor = int(num_sector), int(num_neighbor)
                arr_nr = list(map(int, arr_nr.strip('[]').split(', ')))
                num_ring = sum(arr_nr)
                best_model = RAC_CNN(num_ring, (num_neighbor, num_sector), num_class)
            elif method == 'daa_1dcnn':
                arr_n = list(map(int, gen_cfg.split('_')[-1].strip('[]').split(', ')))
                num_feat = sum(arr_n) + 4
                best_model = DAA_CNN((2, num_feat), num_class)
            elif method == 'lpt_nn':
                num_dist = int(gen_cfg.split('_')[-1])
                best_model = FNN(num_dist, num_class)
            else:
                print('Invalid method')
                continue
            # load best model
            best_model.load_state_dict(torch.load(os.path.join('model', sim_cfg, method, gen_cfg, 'best_model.pth')))
            best_model.to(device)

            path = os.path.join(test_path, method, gen_cfg)
            for s in os.listdir(path):
                # use regex to parse test parameters
                match = re.match('(pos|mag|fs)([0-9]+\.?[0-9]*)', s)
                if match is None:
                    name, x = 'default', 0
                else:
                    name, x = match.groups()
                    x = float(x)
                # calculate accuracy
                df = pd.read_csv(os.path.join(path, s, 'labels.csv'))
                if method == 'rac_1dcnn':
                    dataset = RACDataset(os.path.join(path, s), gen_cfg)
                elif method == 'daa_1dcnn':
                    dataset = DAADataset(os.path.join(path, s), gen_cfg)
                elif method == 'lpt_nn':
                    dataset = LPTDataset(os.path.join(path, s), gen_cfg)
                loader = DataLoader(dataset, batch_size)
                y = check_nn_accuracy(method, best_model, loader, df['img_id'], device=device)
                # store the results
                if name == 'default':
                    res[method]['default'] = y
                    continue
                if name not in res[method].keys():
                    res[method][name] = [(x, y)]
                else:
                    res[method][name].append((x, y))
    
    print(res)
