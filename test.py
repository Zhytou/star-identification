import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from generate import database_path, test_path, sim_cfg, num_class, gcatalogue
from dataset import LPTDataset, RACDataset, DAADataset
from models import RAC_CNN, DAA_CNN, FNN


def check_pm_accuracy(db: pd.DataFrame, df: pd.DataFrame, size: tuple[int, int], T: int, Rp: float, sim: float=0.8):
    '''
        Evaluate the pattern match method's accuracy on the provided patterns. The accuracy is calculated based on the method's ability to correctly identify the closest pattern in the database.
    Args:
        db: the database of patterns
        df: the patterns to be tested
        size: the size of 0-1 pattern matrix
        T: the score threshold for pattern matching
        Rp: the radius in rad for pattern region
        sim: the similarity coefficient for soft match, if similarity == 0, then use hard match
    Returns:
        the accuracy of the pattern match method
    '''

    def compress_db_row(row: pd.Series):
        '''
            Compress the database row.
        Args:
            row: the database row
        Returns:
            the guide star id
        '''
        assert row.dropna().nunique() == 1, 'The row should have only one unique value'
        return row.dropna().iloc[0]

    def cal_match_score(db: pd.DataFrame, pat: np.ndarray):
        '''
            Calculate the match score between two patterns based on the specified similarity metric.
        Args:
            db: the database of patterns
            pat: input pattern vector, each element is the index of 1 value in the 0-1 pattern matrix
        Returns:
            the match score between the db patterns and the input pattern
        ''' 
        # exact match | hard match
        scores = db[pat].notna().sum(axis=1)

        # close match | soft match
        if sim > 0:
            sim_pat = []
            for coord in pat:
                row, col = coord // size[1], coord % size[1]
                for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nrow, ncol = row + d[0], col + d[1]
                    if 0 <= nrow < size[0] and 0 <= ncol < size[1]:
                        sim_pat.append(nrow * size[1] + ncol)
            scores += db[sim_pat].notna().sum(axis=1)*sim

        return scores
    
    def filter_star_in_fov(cata: pd.DataFrame, ids: list, r: float):
        '''
            Filter the stars in the field of view.
        Args:
            cata: the catalogue of stars
            ids: the ids of the star(center of circle fov)
            r: the radius in rad
        Returns:
            the ids of star in the fov
        '''
        cata['X'] = np.cos(cata['De']) * np.cos(cata['Ra'])
        cata['Y'] = np.cos(cata['De']) * np.sin(cata['Ra'])
        cata['Z'] = np.sin(cata['De'])
        pos1 = cata[['X', 'Y', 'Z']].to_numpy()

        stars = cata[cata['Star ID'].isin(ids)]
        pos2 = stars[['X', 'Y', 'Z']].to_numpy()
        
        angles = np.dot(pos1, pos2.T)
        filter_mask = np.all(angles > np.cos(np.radians(r)), axis=1)

        return cata.loc[filter_mask, 'Star ID'].to_numpy()

    def identify_img_pats(img_pats: pd.DataFrame):
        '''
            Identify the patterns in the image.
        Args:
            img_pats: the patterns in the image
            db: the database of patterns
        Returns:
            True if the image patterns are identified, False otherwise
        '''
        n = len(img_pats)
        if n < 4:
            return False
        
        # match results
        matched_ids, potential_ids = [], []
        for _, val in img_pats.iterrows():
            # pattern and id for each star in the image
            pat, id = np.array(val['pattern'].split(' '), dtype=int), val['id']
            
            # calculate the match scores
            scores = cal_match_score(db, pat)
            
            # max score guide star id
            ids = gstar_ids[scores==scores.max()].values
            if len(ids) == 1 and ids[0] == id:
                matched_ids.append(id)
                continue

            # other possible guide star ids
            ids = gstar_ids[scores>=T].values
            if id in ids:
                potential_ids.append(ids)

        # count of correctly identified patterns in the image
        cnt = len(matched_ids)
        if cnt < 3:
            # use the identified patterns and fov restriction to exclude spurious matches
            filtered_ids = filter_star_in_fov(gcatalogue, matched_ids, Rp)
            # print(len(filtered_ids), filtered_ids)
            for ids in potential_ids:
                common_ids = np.intersect1d(filtered_ids, ids)
                # print(len(common_ids), common_ids)
                if len(common_ids) >= 1:
                    cnt += 1
                if cnt >= 3:
                    break
   
        return cnt >= 3

    # rename the columns of the database
    db.columns = db.columns.astype(int)
    # get the id of guide stars in database
    gstar_ids = db.apply(compress_db_row, axis=1)
    # get identify results
    res = df.groupby('img_id', as_index=True).apply(identify_img_pats, include_groups=False)

    # calculate the accuracy
    df = df['img_id'].value_counts()
    tot = df[df >= 4].count()
    # print('total:', tot, len(df))
    acc = round(np.sum(res)/tot*100.0, 2)

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

            if method == 'grid':
                Rp = int(gen_cfg.split('_')[-2])
                L = int(gen_cfg.split('_')[-1])
                size = (L, L)
            else:
                Rp = int(gen_cfg.split('_')[-3])
                L1, L2 = gen_cfg.split('_')[-2:]
                size = (int(L1), int(L2))
            db = pd.read_csv(os.path.join(database_path, gen_cfg, f'{method}.csv'))
            print(method, 'avg db pat cnt', np.sum(db.notna().values)/len(db))

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
                y = check_pm_accuracy(db, df, size, Rp=Rp, T=15, sim=0.8)
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
