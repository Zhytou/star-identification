import os
import re
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances

from dataset import create_dataset
from model import create_model


# Chinese font setting
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEBUG = True

# A single image is considered to have been successfully identified only when min_cnt stars are successfully identified within it.
# minimum count of successfully identified stars
min_sis_cnt = 3


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
        r: the angle distance threshold in radians
    Returns:
        the ids of stars in the biggest cluster
    '''

    stars = cata[cata['Star ID'].isin(ids)]
    # right ascension and declination
    ra_des = stars[['Ra', 'De']].to_numpy()
    # distance matrix
    dis_mat = haversine_distances(ra_des, ra_des)
    
    # cluster labels
    labels = DBSCAN(
        eps=r, 
        min_samples=1,
        metric='precomputed',
    ).fit_predict(dis_mat)

    # get the label of biggest cluster
    ulabels, cnts = np.unique(labels, return_counts=True)
    # verify failure, if biggest cluster is smaller than min_sis_cnt or biggest cluster cannot be determined
    if np.max(cnts) < min_sis_cnt or np.sum(np.max(cnts) == cnts) > 1:
        return np.full_like(ids, -1)

    # get the ids of stars not in the biggest cluster
    max_label = ulabels[np.argmax(cnts)]
    suprious_ids = stars['Star ID'][labels != max_label]

    # set all the suprious ids to -1
    mask = np.isin(ids, suprious_ids)
    ids[mask] = -1

    return ids


def identify_pattern(db: pd.DataFrame, pats: list[np.ndarray], size: tuple[int, int], T: float):
    '''
        Identify the star by pattern matching.
    '''

    def compress_db_row(row: pd.Series):
        '''
            Compress the database row.
        '''
        assert row.dropna().nunique() == 1, 'The row should have only one unique value(guide star id)'
        return row.dropna().iloc[0]

    # the proportion of unsuccessfully identified stars
    multi_max = 0 # 多个最大匹配
    lower_thd = 0 # 低于阈值

    # rename the columns of the database
    db.columns = db.columns.astype(int)
    # get the id of guide stars in database
    gstar_ids = db.apply(compress_db_row, axis=1).to_numpy()

    esti_ids = []
    for pat in pats:
        # calculate the match scores
        scores = cal_match_score(db, pat, size)
    
        # max score guide star id
        ids = gstar_ids[np.logical_and(scores==scores.max(), scores>=T)]
        if len(ids) == 1:
            esti_ids.append(ids[0])
        else:
            multi_max += 1 if len(ids) > 1 else 0
            lower_thd += 1 if len(ids) == 0 else 0
            esti_ids.append(-1)

    if DEBUG:
        print(
            'Total patterns:', len(pats),
            '\nMulti max:', multi_max,
            '\nLower thd:', lower_thd,
        )

    return np.array(esti_ids)


def check_pm_accuracy(db: pd.DataFrame, df: pd.DataFrame, size: tuple[int, int], T: float, Rp: float, gcata: pd.DataFrame, method: str='', test_name: str=''):
    '''
        Evaluate the pattern match method's accuracy on the provided patterns. The accuracy is calculated based on the method's ability to correctly identify the closest pattern in the database.
    Args:
        db: the database of guide star patterns
        df: the patterns of each test star image
        size: the size of 0-1 pattern matrix
        T: the score threshold for pattern matching
        Rp: the radius in degree for pattern region
    Returns:
        the accuracy of the pattern match method
    '''
    res = []

    #! the identification step
    # patterns
    pats = df['pat'].str.split(' ').apply(lambda x: np.array(x, dtype=int)).to_numpy()

    # estimated star ids
    esti_ids = identify_pattern(db, pats, size, T)

    # real star ids
    real_ids = df['star_id'].to_numpy()

    if DEBUG:
        print(
            'Correct pattern', np.sum(esti_ids == real_ids),
            '\nMethod:', method,
            '\nTest name:', test_name,
            '\n--------------'
        )

    # image ids
    img_ids = df['img_id'].to_numpy()

    # identify each image
    for img_id in np.unique(img_ids):
        mask = img_ids == img_id

        # skip if the number of stars in the image is less than min_sis_cnt
        if np.sum(mask) < min_sis_cnt or np.sum(esti_ids[mask] != -1) < min_sis_cnt:
            res.append(False)
            continue
        
        #! the verification step
        # do fov restriction by clustering and take the biggest cluster as final result for each image
        esti_ids[mask] = cluster_by_angle(gcata, esti_ids[mask], 2*Rp)
       
        #! the check step
        if np.sum(np.logical_and(esti_ids[mask] == real_ids[mask], real_ids[mask] != -1)) >= min_sis_cnt:
            res.append(True)
        else:
            res.append(False)

    # calculate the accuracy
    df = df['img_id'].value_counts()
    tot = np.sum(df >= min_sis_cnt)
    acc = round(np.sum(res)/tot*100.0, 2)

    return acc


def predict(model: nn.Module, loader: DataLoader, T: float=0, device=torch.device('cpu')):
    '''
        Predict the labels of the test data.
    '''
    res = []

    # set the model into evaluation mode
    model.eval()

    # move the model to the device
    model.to(device)

    with torch.no_grad():
        for feats, _ in loader:
            # move the features and labels to the device
            feats = feats.to(device)
            
            # forward pass to get output/logits
            scores = model(feats)

            # get the probabilities
            probs = F.softmax(scores, dim=1)

            # get predictions from the maximum value
            vals, idxs = torch.max(probs, dim=1)

            # filter the predictions by threshold
            mask = vals < T
            idxs[mask] = -1

            # append the predicted labels
            res.extend(idxs.tolist())
    
    return np.array(res)


def check_nn_accuracy(model: nn.Module, df: pd.DataFrame, method: str, gen_cfg: str, Rp: float, gcata: pd.DataFrame, batch_size: int=2048, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader.
    '''

    res = []

    # create the dataset and loader
    dataset = create_dataset(method, df, gen_cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    #! the identification step
    # get the predicted catalogue index
    esti_idxs = predict(model, loader, device=device)

    # get the guide star ids by non -1 idxs
    esti_ids = np.full_like(esti_idxs, -1)
    mask = np.array(esti_idxs) != -1    
    esti_ids[mask] = gcata.loc[esti_idxs[mask], 'Star ID'].to_numpy()

    # get the real star ids
    real_ids = df['star_id'].to_numpy()

    # get the image ids
    img_ids = df['img_id'].to_numpy()

    # identify each image
    for img_id in np.unique(img_ids):
        mask = img_ids == img_id

        if np.sum(mask) < min_sis_cnt or np.sum(esti_ids[mask] != -1) < min_sis_cnt:
            res.append(False)
            continue

        #! the verification step
        # do fov restriction by clustering and take the biggest cluster as final result
        esti_ids[mask] = cluster_by_angle(gcata, esti_ids[mask], 2*Rp)

        #! the check step
        if np.sum(np.logical_and(esti_ids[mask] == real_ids[mask], real_ids[mask] != -1)) >= min_sis_cnt:
            res.append(True)
        else:
            res.append(False)

    # calculate the accuracy
    df = df['img_id'].value_counts()
    tot = len(df[df >= min_sis_cnt])
    acc = round(np.sum(res)/tot*100.0, 2)

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
        'rac_1dcnn': '本文方法',
        'lpt_nn': '基于Polestar模式的神经网络算法',
        'grid': '栅格算法',
        'lpt': '改进的LPT算法'
    }
    # test type abbreviation to full name
    type_2_name = {
        'pos': '位置噪声',
        'mag': '亮度噪声',
        'fs': '伪星噪声',
        'ms': '缺失星噪声'
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
        ax.set_ylabel('识别率')

        ymin = 90
        for method in abbr_2_name.values():
            if method not in res or name not in res[method]:
                continue

            res[method][name].sort(key=lambda x: x[0])
            xs, ys = zip(*res[method][name])
            
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


def do_test(meth_params: dict, simu_params: dict, model_types: dict, test_params: dict, gcata_path: str, num_thd: int=20):
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
    sim_cfg = f'{simu_params["h"]}_{simu_params["w"]}_{simu_params["fovy"]}_{simu_params["fovx"]}_{simu_params["limit_mag"]}_{simu_params["rot"]}'

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
        # pattern radius in radians
        Rp = np.radians(meth_params[method][1])

        # generation config for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        print('Method:', method, '\nGeneration config:', gen_cfg)
        
        if method in ['grid', 'lpt']:
            # parse method parameters
            size = [meth_params[method][-1]]*2 if method == 'grid' else meth_params[method][-2:]

            # load the database
            db = pd.read_csv(os.path.join('database', sim_cfg, method, gen_cfg, noise_cfg, 'db.csv'))
        
            # database information
            db_info = np.sum(db.notna().to_numpy(), axis=1)
            max_cnt, min_cnt, avg_cnt = np.max(db_info), np.min(db_info), np.sum(db_info)/len(db)
            print(
                'Database information:',
                '\nSize of database:', len(db),
                '\nTheoretical size of database:', len(gcata),
                '\nMax count of 1 in pattern matrix', max_cnt, 
                '\nMin count of 1 in pattern matrix', min_cnt, 
                '\nAvg count of 1 in pattern matrix', avg_cnt
            )
        elif method in ['rac_nn', 'lpt_nn']:
            # device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # initialize a default model
            model = create_model(method, model_types[method], meth_params[method], num_class)

            # load best model
            model.load_state_dict(torch.load(os.path.join('model', sim_cfg, method, gen_cfg, model_types[method], 'best_model.pth')))

            print('Device:', device)
        else:
            print('Wrong Method!')
            continue

        tasks[method] = {}
        for test_name in test_names:
            # directory path storing the labels.csv for each test
            test_dir = os.path.join('test', sim_cfg, method, gen_cfg, test_name)
            df = pd.read_csv(os.path.join(test_dir, 'labels.csv'))
                        
            if method in ['grid', 'lpt']:
                tasks[method][test_name] = pool.submit(
                    check_pm_accuracy, 
                    db, 
                    df, 
                    size, 
                    T=0, 
                    Rp=Rp, 
                    gcata=gcata,
                    method=method,
                    test_name=test_name
                )
            else:
                tasks[method][test_name] = pool.submit(
                    check_nn_accuracy,
                    model,
                    df, 
                    method,
                    gen_cfg,
                    Rp=Rp,
                    gcata=gcata,
                    device=device,
                )
    
    # aggregate the results
    res = {}
    for method in tasks:
        res[method] = {}
        for test_name in tasks[method]:
            # get the accuracy
            y = tasks[method][test_name].result()

            # use regex to parse test parameters
            match = re.match('(pos|mag|fs|ms)([0-9]+\.?[0-9]*)', test_name)
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
            # 'lpt_nn': [0.5, 6, 55],
            'rac_nn': [0.5, 6, [15, 35, 55], 18, 3],
            # 'grid': [0.5, 6, 100], 
            # 'lpt': [0.5, 6, 50, 36]
        },
        {
            'h': 1024,
            'w': 1282,
            'fovy': 12,
            'fovx': 14.9925,
            'limit_mag': 6,
            'sigma_pos': 0,
            'sigma_mag': 0,
            'num_fs': 0,
            'num_ms': 0,
            'rot': 1
        },
        {
            'rac_nn': 'fnn',
        },
        {
            # 'pos': [0, 0.5, 1, 1.5, 2],
            'mag': [0, 0.1, 0.2, 0.3, 0.4],
            # 'fs': [0, 1, 2, 3, 4],
            'ms': [0, 1, 2, 3, 4]
        },
        './catalogue/sao6.0_d0.03_12_15.csv',
    )

    print(res)
    # draw_results(res)
