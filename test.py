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
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

from dataset import create_dataset
from models import create_model


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


def cal_match_score(db: pd.DataFrame, pat: np.ndarray, size: tuple[int, int]):
    '''
        Calculate the match score between two patterns based on the specified similarity metric.
    Args:
        db: the database of patterns
        pat: input pattern vector, each element is the index of 1 value in the 0-1 pattern matrix
        size: the size of 0-1 pattern matrix
    Returns:
        the match score between the db patterns and the input pattern
    ''' 

    # intersect the pattern with the database
    pat = np.intersect1d(pat, db.columns.astype(int))

    # exact match | hard match
    scores = db[pat].notna().sum(axis=1)
    assert scores.size == len(db), 'The size of scores should be equal to the size of db'

    # close match | soft match
    dds = {
        0.4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
        0.2: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        # 0.2: [(-2, 0), (2, 0), (0, -2), (0, 2)],
        # 0.1: [(-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)],
        # 0.05: [(-2, -2), (-2, 2), (2, -2), (2, 2)],
    }

    for sim, dd in dds.items():
        sim_pat = []
        for coord in pat:
            row, col = coord // size[1], coord % size[1]
            for d in dd:
                nrow, ncol = row + d[0], col + d[1]
                if 0 <= nrow < size[0] and 0 <= ncol < size[1]:
                    sim_pat.append(nrow * size[1] + ncol)
        sim_pat = np.intersect1d(sim_pat, db.columns.astype(int))
        scores += db[sim_pat].notna().sum(axis=1)*sim

    return scores


def cluster_by_angle(cata: pd.DataFrame, ids: np.ndarray, r: float):
    '''
        Cluster the stars by angle distance.
    Args:
        cata: the catalogue of stars
        ids: the ids of the star(center of circle fov)
        r: the radius in rad
    Returns:
        the ids of stars in the biggest cluster
    '''

    stars = cata[cata['Star ID'].isin(ids)]
    ra_des = stars[['Ra', 'De']].to_numpy()
    # distance matrix
    dis_mat = haversine_distances(ra_des, ra_des)

    # angle distance threshold
    eps = np.radians(r)

    # cluster labels
    labels = DBSCAN(
        eps=eps, 
        min_samples=1,
        metric='precomputed',
    ).fit_predict(dis_mat)

    # get the label of biggest cluster
    ulabels, cnts = np.unique(labels, return_counts=True)
    # verify failure, if cannot determine the biggest cluster
    if np.sum(np.max(cnts) == cnts) > 1:
        return np.full_like(ids, -1)

    # get the ids of stars not in the biggest cluster
    max_label = ulabels[np.argmax(cnts)]
    suprious_ids = stars['Star ID'][labels != max_label]

    # set all the suprious ids to -1
    mask = np.isin(ids, suprious_ids)
    ids[mask] = -1

    return ids


def check_pm_accuracy(db: pd.DataFrame, df: pd.DataFrame, size: tuple[int, int], T: float, Rp: float, gcata: pd.DataFrame, by_img: bool=False):
    '''
        Evaluate the pattern match method's accuracy on the provided patterns. The accuracy is calculated based on the method's ability to correctly identify the closest pattern in the database.
    Args:
        db: the database of guide star patterns
        df: the patterns of each test star image
        size: the size of 0-1 pattern matrix
        T: the score threshold for pattern matching
        Rp: the radius in degree for pattern region
        by_img: calculate the accuracy by image or by pattern
    Returns:
        the accuracy of the pattern match method
    '''

    # the proportion of unsuccessfully identified stars
    multi_max = 0 # 多个最大匹配
    lower_thd = 0 # 低于阈值
    error_id = 0 # 错误id

    def identify_image(img_df: pd.DataFrame):
        '''
            Identify the image by its patterns.
        '''
        n = len(img_df)
        if n < 4:
            return False
        
        # real star ids
        real_ids = img_df['star_id'].to_numpy()

        #! the identification step
        # estimated star ids
        esti_ids = []
        for pat in img_df['pat']:
            # convert str pat into numpy array
            pat = np.array(pat.split(' '), dtype=int)
            
            # calculate the match scores
            scores = cal_match_score(db, pat, size)
            
            # max score guide star id
            ids = gstar_ids[np.logical_and(scores==scores.max(), scores>=T)].to_numpy()
            if len(ids) == 1:
                esti_ids.append(ids[0])
            else:
                esti_ids.append(-1)

        esti_ids = np.array(esti_ids)

        #! the verification step
        if np.sum(esti_ids != -1) >= 3:
            # do fov restriction by clustering and take the biggest cluster as final result
            esti_ids = cluster_by_angle(gcata, esti_ids, Rp)

        #! the check step
        # count the successfully identified stars
        cnt = np.sum(np.logical_and(esti_ids == real_ids, real_ids != -1))

        return cnt >= 3

    def identify_pattern(df: pd.DataFrame):
        '''
            Identify the pattern by its patterns.
        '''
        real_ids = df['star_id'].to_numpy()
        pats = df['pat'].str.split(' ').apply(lambda x: np.array(x, dtype=int)).to_numpy()
        
        esti_ids = []
        for pat in pats:
            # calculate the match scores
            scores = cal_match_score(db, pat, size)
        
            # max score guide star id
            max_score = scores.max()
            idxs = np.where(scores == max_score)[0]
            if max_score < T or len(idxs) > 1:
                nonlocal multi_max, lower_thd
                multi_max += 1 if len(idxs) > 1 else 0
                lower_thd += 1 if max_score < T else 0
                esti_ids.append(-1)
            else:
                esti_ids.append(gstar_ids.iloc[idxs[0]])

        res = np.sum(np.logical_and(esti_ids == real_ids, real_ids != -1))
        return res

    # rename the columns of the database
    db.columns = db.columns.astype(int)
    # get the id of guide stars in database
    gstar_ids = db.apply(compress_db_row, axis=1)

    if by_img:
        # get identify results
        res = df.groupby('img_id', as_index=True).apply(identify_image, include_groups=False)
        # calculate the accuracy
        df = df['img_id'].value_counts()
        tot = np.sum(df >= 4)
        acc = round(np.sum(res)/tot*100.0, 2)
    else:
        # get the pattern id
        res = identify_pattern(df)
        # calculate the accuracy
        tot = len(df)
        acc = round(res/tot*100.0, 2)

    if True:
        print(
            '--------------'
            'Total patterns:', tot,
            '\nSuccessfully identified patterns:', res,
            '\nError id:', error_id,
            '\nMulti max:', multi_max,
            '\nLower thd:', lower_thd,
            '\nAccuracy:', acc,
            '--------------'
        )

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
        # 'ms': 'Number of missing stars'
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


def do_test(meth_params: dict, simu_params: dict, test_params: dict, gcata_path: str, num_thd: int=20):
    '''
        Do test.
    '''
    if meth_params == {}:
        return

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thd)
    # tasks for later aggregation
    tasks = {}

    # aggregate test params
    test_names = []
    for test_type in test_params:
        test_names.extend(f'{test_type}{val}' for val in test_params[test_type])

    # simulation config
    sim_cfg = f'{simu_params["h"]}_{simu_params["w"]}_{simu_params["fovx"]}_{simu_params["fovy"]}_{simu_params["limit_mag"]}'

    # noise config
    noise_cfg = f'{simu_params["sigma_pos"]}_{simu_params["sigma_mag"]}_{simu_params["num_fs"]}_{simu_params["num_ms"]}'

    # read the guide star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])
    num_class = len(gcata)

    print('Test')
    print('------------------')
    print('Simulation config:', sim_cfg)
    print('Test names:', test_names)

    # add each test task to the threadpool
    for method in meth_params:
        # generation config for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        print('Method:', method, '\nGeneration config:', gen_cfg)
        
        if method in ['grid', 'lpt']:
            # load the database
            db = pd.read_csv(os.path.join('database', sim_cfg, method, gen_cfg, noise_cfg, 'db.csv'))
        
            # database information
            db_info = np.sum(db.notna().to_numpy(), axis=1)
            max_cnt, min_cnt, avg_cnt = np.max(db_info), np.min(db_info), np.sum(db_info)/len(db)
            print(
                'Max count of 1 in pattern matrix', max_cnt, 
                '\nMin count of 1 in pattern matrix', min_cnt, 
                '\nAvg count of 1 in pattern matrix', avg_cnt
            )

            # parse method parameters
            if method == 'grid':
                _, Rp, L = meth_params[method]
                size = (L, L)
                T = 0
            else:
                _, Rp, L1, L2 = meth_params[method]
                size = (int(L1), int(L2))
                T = 0
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
                tasks[method][test_name] = pool.submit(check_pm_accuracy, db, df, size, T=T, Rp=Rp, gcata=gcata)
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
            assert match is not None, 'Cannot parse the test name'
            name, x = match.groups()
            x = float(x)
            
            # store the results
            if name not in res[method].keys():
                res[method][name] = [(x, y)]
            else:
                res[method][name].append((x, y))

    pool.shutdown()

    return res


if __name__ == '__main__':
    res = do_test(
        {
            # 'lpt_nn': [6, 50],
            # 'rac_1dcnn': [6, [20, 50, 80], 16, 3],
            # 'grid': [0.1, 6, 50], 
            'lpt': [0.1, 6, 25, 50]
        },
        {
            'h': 512,
            'w': 512,
            'fovx': 12,
            'fovy': 12,
            'limit_mag': 6,
            'sigma_pos': 0,
            'sigma_mag': 0,
            'num_fs': 0,
            'num_ms': 0,
        },
        {
            'pos': [0, 0.5, 1, 1.5, 2],
            # 'mag': [0, 0.1, 0.2, 0.3, 0.4],
            # 'fs': [0, 1, 2, 3, 4]
        },
        './catalogue/sao6.0_d0.03_12_15.csv',
    )

    print(res)
    # draw_results(res)
