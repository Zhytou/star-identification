import os, re, json
from datetime import datetime
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

from generate import setup
from dataset import create_dataset
from model import create_model
from utils import get_angdist, get_attitude_matrix


# Chinese font setting
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEBUG = False

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


def cluster_by_angle(ra_des: np.ndarray, ids: np.ndarray, probs: np.ndarray, r: float):
    '''
        Cluster the stars by angle distance.
    Args:
        ra_des: right ascension and declination of valid ids(non -1)
        ids: the ids of the star(center of circle fov)
        r: the angle distance threshold in radians
    Returns:
        the ids of stars in the biggest cluster
    '''
    # some of ids may be -1, because of its low probability
    mask = ids != -1

    # distance matrix
    dis_mat = haversine_distances(ra_des, ra_des)
    
    # cluster labels
    labels = DBSCAN(
        eps=r, 
        min_samples=1,
        metric='precomputed',
    ).fit_predict(dis_mat)

    # get the unique label and counts of each cluster
    clabels, cnts = np.unique(labels, return_counts=True)
    
    # get each cluster probabiliy sum
    cprobs = np.zeros_like(clabels, dtype=np.float32)
    for i, label in enumerate(clabels):
        cprobs[i] = np.sum(probs[mask][labels == label])
    
    # get the the cluster labels with the maximum count
    mc_labels = clabels[cnts == cnts.max()]
    # get the the cluster labels with the maximum probability sum
    mp_labels = clabels[cprobs == cprobs.max()]

    # intersect mc_labels with mp_labels
    mcp_labels = np.intersect1d(mc_labels, mp_labels)

    # get the label id of the biggest cluster
    if len(mc_labels) == 1:
        max_label = mc_labels[0]
    elif len(mcp_labels) == 1:
        max_label = mcp_labels[0]
    else:
        # verify failure, biggest cluster cannot be determined by both counts and probabilities
        return np.full_like(ids, -1)

    # get the ids of stars not in the biggest cluster
    suprious_ids = ids[labels != max_label]

    # set all the suprious ids to -1
    ids[np.isin(ids, suprious_ids)] = -1

    return ids


def match_pattern(db: pd.DataFrame, pats: list[np.ndarray], size: tuple[int, int], T: float):
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


def check_pm_accuracy(db: pd.DataFrame, df: pd.DataFrame, method: str, meth_params: list, gcata: pd.DataFrame, h: int, w: int, f: float, T: float=0, test_name: str=''):
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

    # parse parameters
    size = meth_params[2:] if method == 'lpt' else (meth_params[2], meth_params[2])
    rp = np.radians(meth_params[1])

    #! the identification step
    # patterns
    df = df.dropna(axis=0)
    pats = df['pat'].str.split(' ').apply(lambda x: np.array(x, dtype=int)).to_numpy()

    # estimated star ids
    esti_ids = match_pattern(db, pats, size, T)

    # real star ids
    real_ids = df['star_id'].to_numpy()

    if DEBUG:
        print(
            'Correct pattern', np.sum(esti_ids == real_ids),
            '\nMethod:', method,
            '\nTest name:', test_name,
        )

    # image ids
    img_ids = df['img_id'].to_numpy()

    # identify each image
    for img_id in np.unique(img_ids):
        mask = img_ids == img_id

        #! the early check step
        if np.sum(np.logical_and(esti_ids[mask] == real_ids[mask], real_ids[mask] != -1)) >= min_sis_cnt:
            res.append(True)
            continue
        elif np.sum(np.logical_and(esti_ids[mask] == real_ids[mask], real_ids[mask] != -1)) <= 1:
            res.append(False)
            continue
        
        #! the verification and postprocess step
        # esti_ids[mask], att_mat = verify(
        #     gcata, 
        #     coords[mask], 
        #     esti_ids[mask], 
        #     probs=probs[mask],
        #     fov=2*rp,
        #     h=h,
        #     w=w,
        #     f=f,
        # )

        # esti_ids[mask] = postprocess(
        #     gcata, 
        #     coords[mask], 
        #     esti_ids[mask], 
        #     att_mat=att_mat,
        #     fov=2*rp,
        #     h=h,
        #     w=w,
        #     f=f,
        # )
       
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


def predict(model: nn.Module, loader: DataLoader, T: float, device=torch.device('cpu')):
    '''
        Predict the labels of the test data.
    '''
    # catalogue indexs and confidence interval
    labels, cis = [], []

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
            vals[mask] = 0
            idxs[mask] = -1

            # append the predicted labels
            labels.extend(idxs.tolist())
            cis.extend(vals.tolist())
    
    return np.array(labels), np.array(cis)


def verify(cata: pd.DataFrame, coords: np.ndarray, ids: np.ndarray, probs: np.ndarray, fov: float, h: int, w: int, f: int, eps: float=1e-4):
    '''
        Verify the estimated results by angular distances and calculate the attitude matrix with the highest confidence star.  
    '''
    assert len(coords) == len(ids)
    mask = ids != -1

    if not np.any(mask):
        return ids, None

    # get the concerning stars
    #! careful! use index to sort the stars, because the output of isin is not sorted by the given list
    stars = cata[cata['Star ID'].isin(ids)].copy()
    stars = stars.set_index('Star ID').loc[ids[mask]].reset_index()

    ### 1.do fov restriction by clustering and take the biggest cluster as output
    #! careful! fov is in radians
    ids = cluster_by_angle(stars[['Ra', 'De']].to_numpy(), ids, probs, fov)

    # update mask
    mask = ids != -1
    n = np.sum(mask)
    if n <= 1:
        return ids, None

    ### 2.do angular distnace validation
    # get the view vectors
    vvs = np.full((n, 3), f)
    vvs[:, 0] = coords[mask, 1]-w/2
    vvs[:, 1] = coords[mask, 0]-h/2

    # get the refernce vectors
    stars['X'] = np.cos(stars['Ra'])*np.cos(stars['De'])
    stars['Y'] = np.sin(stars['Ra'])*np.cos(stars['De'])
    stars['Z'] = np.sin(stars['De'])
    rvs = stars[['X', 'Y', 'Z']].to_numpy()[mask] # use mask, because ids may change after cluster_by_angle

    # get the angular distances
    vagds, ragds = get_angdist(vvs), get_angdist(rvs)

    # compare the angular distnaces
    match = np.isclose(vagds, ragds, atol=eps)
    scores = np.sum(match, axis=1)

    # at least the angular distance between one pair of stars is verified, the image can be considered as correctly identification    
    # !careful! minus one to exclude itself
    if np.max(scores)-1 < 1:
        # if no star pairs are verified, leave to postprocess    
        # ids[probs != probs.max()] = -1
        return ids, None
    
    # get the star pair with the highest match
    if probs is None:
        idx1, idx2 = np.argsort(scores)[-2:]
    else:
        # if more than one maximum pair, use probability/score to determine
        # in other words, sort by scores first, if same score, then use probs to sort
        idx1, idx2 = np.lexsort((probs[mask], scores))[-2:]    

    # keep results that satisfy both the idx1-th and idx2-th angular distance constraints
    cstr = match[idx1] & match[idx2]
    idxs = np.where(mask)[0] 
    ids[idxs[~cstr]] = -1

    ### 3.calculate the attitude matrix
    att_mat = get_attitude_matrix(vvs[cstr].T, rvs[cstr].T)

    return ids, att_mat


def postprocess(cata: pd.DataFrame, coords: np.ndarray, ids: np.ndarray, att_mat: np.ndarray, h: int, w: int, f: int, eps1: float=5e-5, eps2: float=1e-3):
    '''
        Use the estimated and verified ids(attitude info) to predict unidentified ids.
    '''
    assert len(coords) == len(ids)
    n = len(ids)

    mask = ids != -1
    if not np.any(mask):
        return ids
    
    # get the view vectors
    vvs = np.full((n, 3), f)
    vvs[:, 0] = coords[:, 1]-w/2
    vvs[:, 1] = coords[:, 0]-h/2

    # normalize
    vvs = vvs/np.linalg.norm(vvs, axis=1, keepdims=True)

    # if no attitude infomation is offerred, do angular distance match directly
    if att_mat is None:
        # cosine cangular distances between view vectors
        vagds = get_angdist(vvs[mask], vvs) # (m, n) (m is np.sum(mask), n is len(ids)) 

        # get the all potential reference vectors in guide star database
        rvs = np.zeros((len(cata), 3))
        rvs[:, 0] = np.cos(cata['Ra'])*np.cos(cata['De'])
        rvs[:, 1] = np.sin(cata['Ra'])*np.cos(cata['De'])
        rvs[:, 2] = np.sin(cata['De'])
        
        # get the idxs of non -1 ids in cata
        idxs = cata.set_index('Star ID').index.get_indexer(ids[mask])
        cata.reset_index()

        # cosine angular distances between all reference vectors
        ragds = get_angdist(rvs[idxs], rvs) # (m, k) (k is len(cata))

        # calculate the differences between view vectors and reference vectors
        diffs = np.abs(vagds[..., None] - ragds[:, None, :]) # (m, n, k)

        # count the valid match for each non -1
        valid_counts = np.sum(
            np.any(diffs < eps1, axis=2), # (m, n)
            axis=1
        )  # (m,)

        # no reference star is matched        
        if np.max(valid_counts) == 1:
            return ids
        
        # take the maximum match count as match result
        diffs = diffs[np.argmax(valid_counts)]  # (n, k)
        match = np.argmin(diffs, axis=1) # (n,)

        # only when differences less than eps, match is valid
        valid_mask = np.min(diffs, axis=1) < eps1

        # get the matched star id
        ids[valid_mask] = cata.loc[match[valid_mask], 'Star ID'].to_numpy()
        
        return ids

    # get the ideal reference vectors
    rvs = vvs @ att_mat
    
    # calaculate the ra and de
    ras = np.mod(np.arctan2(rvs[:, 1], rvs[:, 0]), 2*np.pi)
    des = np.arcsin(rvs[:, 2])

    # find the most similar points
    query = np.column_stack((ras, des))
    refer = cata[['Ra', 'De']].to_numpy()
    dists = haversine_distances(query, refer) # n * m

    min_dists = np.min(dists, axis=1) # n
    idxs = np.argmin(dists, axis=1) # n, each element is in [0, m-1]

    # keep the outputs, if similarity is higher than threshold.
    mask = (ids == -1) & (min_dists < eps2)
    ids[mask] = cata.loc[idxs, 'Star ID'][mask]

    return ids


def check_nn_accuracy(model: nn.Module, df: pd.DataFrame, method: str, meth_params: list, gcata: pd.DataFrame, h: int, w: int, f:float, T: float=0, batch_size: int=512, device=torch.device('cpu')):
    '''
        Evaluate the model's accuracy on the provided data loader.
    '''

    res = []

    # parse parameters
    rp = np.radians(meth_params[1])

    # create the dataset and loader
    dataset = create_dataset(method, df, meth_params)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    #! the identification step
    # get the predicted catalogue index
    esti_idxs, probs = predict(model, loader, T, device=device)

    # get the guide star ids by non -1 idxs
    esti_ids = np.full_like(esti_idxs, -1)
    mask = np.array(esti_idxs) != -1    
    esti_ids[mask] = gcata.loc[esti_idxs[mask], 'Star ID'].to_numpy()

    # get the real star ids
    real_ids = df['star_id'].to_numpy()

    # get the image ids
    img_ids = df['img_id'].to_numpy()

    # get the coordinates of each star
    coords = df[['row', 'col']].to_numpy()

    # identify each image
    for img_id in np.unique(img_ids):
        mask = img_ids == img_id

        # ! the early check step
        if np.sum(np.logical_and(esti_ids[mask] == real_ids[mask], real_ids[mask] != -1)) >= min_sis_cnt:
            res.append(True)
            continue
        elif np.sum(np.logical_and(esti_ids[mask] == real_ids[mask], real_ids[mask] != -1)) <= 1:
            res.append(False)
            continue

        #! the verification and postprocess step
        # esti_ids[mask], att_mat = verify(
        #     gcata, 
        #     coords[mask], 
        #     esti_ids[mask], 
        #     probs=probs[mask],
        #     fov=2*rp,
        #     h=h,
        #     w=w,
        #     f=f,
        # )

        # esti_ids[mask] = postprocess(
        #     gcata, 
        #     coords[mask], 
        #     esti_ids[mask], 
        #     att_mat=att_mat,
        #     fov=2*rp
        #     h=h,
        #     w=w,
        #     f=f,
        # )

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
        'rac_nn': '本文方法',
        'lpt_nn': '基于Polestar模式的神经网络算法',
        'grid': '栅格算法',
        'lpt': '改进的LPT算法'
    }
    # test type abbreviation to full name
    type_2_name = {
        'pos': '位置噪声(pixel)',
        'mag': '亮度噪声(Mv)',
        'fs': '伪星数目',
        'ms': '缺失星数目'
    }

    # set timestamp as sub directory
    now = datetime.now()
    subdir = now.strftime("%Y%m%d_%H%M%S")
    dir = f'res/chapter4/sim/{subdir}'
    os.makedirs(dir, exist_ok=True)
    
    # save the results
    if save:
        with open(f'{dir}/res.txt', 'w') as f:
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
        ax.set_ylabel('识别率(%)')

        for method in abbr_2_name.values():
            if method not in res or name not in res[method]:
                continue

            res[method][name].sort(key=lambda x: x[0])
            xs, ys = zip(*res[method][name])
            
            # avoid 100% accuracy
            # ys = [y-0.1 for y in ys]
            
            # calculate the minimum y value
            # if ymin > ys[-1]:
            #     ymin = np.floor(ys[-1]/10)*10

            # plot the results
            ax.plot(xs, ys, label=method, marker='o')
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(80, 100)
            ax.set_xticks(xs)
            ax.legend()

        fig.savefig(f'{dir}/{name}.png')
    plt.show()


def do_test(meth_params: dict, simu_params: dict, model_types: dict, test_params: dict, gcata_path: str, num_thd: int=20):
    '''
        Do test.
    '''
    if meth_params == {}:
        return

    # set threshold
    Ts = {
        'grid': 3.4,
        'lpt': 3.8,
        'lpt_nn': 0.3,
        'rac_nn': 0.7,
    }

    # setup
    sim_cfg, noise_cfg, gcata_name, gcata = setup(simu_params, gcata_path)
    num_class = len(gcata)

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thd)
    # tasks for later aggregation
    tasks = {}

    # aggregate test params
    test_names = []
    for test_type in test_params:
        test_names.extend(f'{test_type}{val}' for val in test_params[test_type])

    print(
        # 'Test',
        # '\n------------------------------',
        # '\nTEST INFO',
        '\nNoise config:', noise_cfg,
        '\nTest name:', test_names,
        # '\n------------------------------',
    )

    # add each test task to the threadpool
    for method in meth_params:

        # generation config for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        
        # print info
        print(
            # 'METHOD INFO',
            '\nMethod:', method, 
            # '\nSimulation config:', sim_cfg,
            # '\nGeneration config:', gen_cfg,
            # '\n------------------------------',
        )
        
        if method in ['grid', 'lpt']:
            # load the database
            #? always use 0_0_0_0 db to test
            db = pd.read_csv(os.path.join('database', sim_cfg, method, gen_cfg, '0_0_0_0', 'db.csv'))
                                    
            # database information
            db_info = np.sum(db.notna().to_numpy(), axis=1)
            max_cnt, min_cnt, avg_cnt = np.max(db_info), np.min(db_info), np.sum(db_info)/len(db)

            print(
                # 'DATABASE INFO',
                '\nThreshold:', Ts[method],
                # '\nSize of database:', len(db),
                # '\nTheoretical size of database:', len(gcata),
                # '\nMax count of 1 in pattern matrix', max_cnt, 
                # '\nMin count of 1 in pattern matrix', min_cnt, 
                # '\nAvg count of 1 in pattern matrix', avg_cnt,
                # '\n------------------------------',
            )
        elif method in ['rac_nn', 'lpt_nn']:
            # device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # initialize a default model
            model = create_model(method, model_types[method], meth_params[method], num_class)

            # load best model
            model.load_state_dict(torch.load(os.path.join('model', sim_cfg, method, gen_cfg, model_types[method], 'best_model.pth')))

            print(
                # 'MODEL INFO'
                '\nThreshold:', Ts[method],
                '\nModel type:', model_types[method],
                # '\nDevice:', device,
                # '\n------------------------------',
            )
        else:
            print('Wrong Method!')
            continue

        tasks[method] = {}
        for test_name in test_names:
            # directory path storing the labels.csv for each test
            test_dir = os.path.join('test', sim_cfg, method, gen_cfg, test_name, noise_cfg)
            df = pd.read_csv(os.path.join(test_dir, 'labels.csv'))
                        
            if method in ['grid', 'lpt']:
                tasks[method][test_name] = pool.submit(
                    check_pm_accuracy, 
                    db, 
                    df, 
                    method,
                    meth_params[method],
                    gcata=gcata,
                    h=simu_params['h'],
                    w=simu_params['w'],
                    f=simu_params['f'],
                    T=Ts[method], 
                    test_name=test_name
                )
            else:
                tasks[method][test_name] = pool.submit(
                    check_nn_accuracy,
                    model,
                    df, 
                    method,
                    meth_params[method],
                    gcata=gcata,
                    h=simu_params['h'],
                    w=simu_params['w'],
                    f=simu_params['f'],
                    T=Ts[method],
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
    if True:
        res = do_test(
            {
                # 'lpt_nn': [0.5, 6, 55, 0],
                'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 0],
                # 'grid': [0.5, 6, 100], 
                # 'lpt': [0.5, 6, 50, 50]
            },
            {
                'h': 1024,
                'w': 1282,
                'fovy': 12,
                'fovx': 14.9925,
                'limit_mag': 6,
                'sigma_pos': 0.5,
                'sigma_mag': 0.1,
                'num_fs': 0,
                'num_ms': 0,
                'rot': 1
            },
            {
                'lpt_nn': 'fnn',
                'rac_nn': 'cnn2',
            },
            {
                # 'pos': [0, 0.5, 1, 1.5, 2],
                # 'mag': [0, 0.1, 0.2, 0.3, 0.4],
                'fs': [0, 1, 2, 3, 4],
                # 'ms': [0, 1, 2, 3, 4]
            },
            './catalogue/sao6.0_d0.03_12_15.csv',
        )

    if False:
        res = do_test(
            {
                'rac_nn': [0.5, 7.7, [35, 75, 115], 18, 3, 0],
            },
            {
                'h': 1040,
                'w': 1288,
                'fovx': 18.97205141393946,
                'fovy': 15.36777053565561,
                'limit_mag': 5.5,
                'sigma_pos': 3,
                'sigma_mag': 0,
                'num_fs': 0,
                'num_ms': 0,
                'rot': 1
            },
            {
                'rac_nn': 'cnn2',
            },
            {
                'pos': [0, 0.5, 1, 1.5, 2], 
                'mag': [0, 0.1, 0.2, 0.3, 0.4], 
                'fs': [0, 1, 2, 3, 4],
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
        )

    print(res)
    # draw_results(res)
