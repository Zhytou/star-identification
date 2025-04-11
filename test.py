import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

from generate import sim_cfg, num_class, gcatalogue, gcata_name
from dataset import create_dataset
from models import create_model


def check_pm_accuracy(db: pd.DataFrame, df: pd.DataFrame, size: tuple[int, int], T: float, Rp: float, sim: float=0.8):
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
            ids = gstar_ids[np.logical_and(scores==scores.max(), scores>=T)].values
            if len(ids) == 1 and ids[0] == id:
                matched_ids.append(id)
                continue

            # other possible guide star ids
            ids = gstar_ids[scores>=T].values
            if id in ids:
                potential_ids.append(ids)

        # count of correctly identified patterns in the image
        cnt = len(matched_ids)
        if cnt < 3 and cnt > 0:
            # use the identified patterns and fov restriction to exclude spurious matches
            filtered_ids = filter_star_in_fov(gcatalogue, matched_ids, Rp)
            # print('filtered ids:', len(filtered_ids), 'potential ids:', len(potential_ids))
            for ids in potential_ids:
                common_ids = np.intersect1d(filtered_ids, ids)
                print('before', len(ids), 'after', len(common_ids))
                if len(common_ids) == 1:
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
    print('total:', tot, len(df))
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


def draw_results(res: dict, save: bool=False):
    '''
        Draw the results of the accuracy.
    Args:
        res: the results to be drawn
            {
                'method1': {
                    'test_type1': [(x, y), ...],
                    'test_type2': [(x, y), ...],
                },
                'method2': {
                    'test_type1': [(x, y), ...],
                    'test_type2': [(x, y), ...],
                },
            }
    '''

    # method abbreviation to full name
    abbr_2_name = {
        'rac_1dcnn': 'Proposed algorithm',
        'lpt_nn': 'Polestar NN algorithm',
        'grid': 'Grid algorithm',
        'lpt': 'Log-polar transform algorithm'
    }
    # test type abbreviation to full name
    type_2_name = {
        'pos': 'Position noise',
        'mag': 'Magnitude noise',
        'fs': 'Number of false stars',
        'ms': 'Number of missing stars'
    }

    # set timestamp as sub directory
    subdir = time.time()
    if not os.path.exists(f'res/chapter4/{subdir}'):
        os.makedirs(f'res/chapter4/{subdir}')
    # save the results
    if save:
        with open(f'res/chapter4/{subdir}/res.txt', 'w') as f:
            json.dump(res, f, indent=4)

    # change abbreviation to full name
    for mabbr, mname in abbr_2_name.items():
        if mabbr not in res:
            continue
        res[mname] = res.pop(mabbr)
        for tabbr, tname in type_2_name.items():
                if tabbr not in res[mname]:
                    continue
                res[mname][tname] = res[mname].pop(tabbr)

    # draw the results
    for name in type_2_name.values():
        fig, ax = plt.subplots()
        ax.set_xlabel(name)
        ax.set_ylabel('Accuracy (%)')

        ymin = 90
        for method in abbr_2_name.values():
            if method not in res or name not in res[method]:
                continue

            y = res[method]['default']
            res[method][name].sort(key=lambda x: x[0])
            xs, ys = zip(*res[method][name])
            xs, ys = [0]+list(xs), [y]+list(ys)
            
            # avoid 100% accuracy
            # ys = [y-0.1 for y in ys]
            
            # calculate the minimum y value
            if ymin > ys[-1]:
                ymin = np.floor(ys[-1]/10)*10

            # plot the results
            ax.plot(xs, ys, label=method, marker='o')
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(ymin, 100)
            ax.legend()

        fig.savefig(f'res/chapter4/{subdir}/{name}.png')
    plt.show()


def do_test(meth_params: dict, test_params: dict, num_thd: int=20):
    '''
        Do test.
    Args:
        meth_params: the parameters for the test sample generation, possible methods include:
        test_params: the parameters for the test sample generation
    '''
    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thd)
    # tasks for later aggregation
    tasks = {}

    # aggregate test params
    test_names = []
    for test_type in test_params:
        if test_type == 'default':
            test_names.append(test_type)
        else:
            test_names.extend(f'{test_type}{val}' for val in test_params[test_type])

    # add each test task to the threadpool
    for method in meth_params:
        # generation config for each method
        gen_cfg = f'{gcata_name}_0_'+'_'.join(map(str, meth_params[method]))
        print(gen_cfg)
        if method in ['grid', 'lpt']:
            # load the database
            db = pd.read_csv(os.path.join('database', sim_cfg, gen_cfg, f'{method}.csv'))
            # average count of 1 in guide star pattern
            avg_cnt = np.sum(db.notna().values)/len(db)
            print(method, 'avg db pat cnt', avg_cnt)
            # parse method parameters
            if method == 'grid':
                _, Rp, L = meth_params[method]
                size = (L, L)
                T = avg_cnt/3
            else:
                _, Rp, L1, L2 = meth_params[method]
                size = (int(L1), int(L2))
                T = avg_cnt/3.5
        elif method in ['rac_1dcnn', 'daa_1dcnn', 'lpt_nn']:
            batch_size = 100
            # use gpu if available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
            # initialize a default model
            best_model = create_model(method, meth_params[method], num_class)
            # load best model
            best_model.load_state_dict(torch.load(os.path.join('model', sim_cfg, method, gen_cfg, 'best_model.pth')))
            best_model.to(device)
        else:
            print('Wrong Method!')
            continue

        tasks[method] = {}
        for test_name in test_names:
            # directory path storing the labels.csv for each test
            test_dir = os.path.join('test', sim_cfg, method, gen_cfg, test_name)
            df = pd.read_csv(os.path.join(test_dir, 'labels.csv'))

            if method in ['grid', 'lpt']:
                tasks[method][test_name] = pool.submit(check_pm_accuracy, db, df, size, Rp, T, 0.8)
            else:
                dataset = create_dataset(method, test_dir, gen_cfg)
                loader = DataLoader(dataset, batch_size)
                tasks[method][test_name] = pool.submit(check_nn_accuracy, method, best_model, loader, df['img_id'], device=device)
    

    # aggregate the results
    res = {}
    for method in tasks:
        res[method] = {}
        for test_name in tasks[method]:
            # get the accuracy
            y = tasks[method][test_name].result()

            # use regex to parse test parameters
            match = re.match('(pos|mag|fs)([0-9]+\.?[0-9]*)', test_name)
            if match is None:
                name, x = 'default', 0
            else:
                name, x = match.groups()
                x = float(x)
            
            # store the results
            if name == 'default':
                res[method]['default'] = y
                continue
            if name not in res[method].keys():
                res[method][name] = [(x, y)]
            else:
                res[method][name].append((x, y))

    return res


if __name__ == '__main__':
    res = do_test(
        # {'lpt_nn': [6, 50]},
        {'rac_1dcnn': [6, [20, 50, 80], 16, 3]},
        # {'grid': [0.3, 6, 50]},
        {'default': True, 'pos': [1, 2], 'mag': [0.2, 0.4], 'fs': [3, 5]}
    )

    print(res)
