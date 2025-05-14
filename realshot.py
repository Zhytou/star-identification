import os
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from simulate import cata
from denoise import filter_image, denoise_image
from detect import cal_threshold, get_seed_coords, group_star
from extract import get_star_centroids
from generate import gen_real_sample, setup
from dataset import create_dataset
from model import create_model
from test import predict, verify, postprocess
from utils import get_angdist, label_star_image, get_attitude_matrix


DEBUG = True


# 验证降噪\二值化\连通域等算法和matlab实现一致性
if False:
    # dir = './example/xie/20161227225347/'
    file = './example/3P0/00001010_00000000019CFBA2/'

    img0 = cv2.imread(file+'o.bmp', cv2.IMREAD_GRAYSCALE)
    print(img0.shape)

    # 验证中值滤波正确性
    img1 = filter_image(img0, 'MEDIAN')
    img2 = cv2.imread(file+'f.bmp', cv2.IMREAD_GRAYSCALE)
    assert np.sum(img1!=img2) == 0, 'Wrong median filter!'

    # 验证阈值计算以及二值化正确性
    T = cal_threshold(img1, 'Liebe5')
    print('Threashold:', T)
    bimg2 = np.zeros_like(img2)
    bimg2[img2 >= T] = 1
    img3 = cv2.imread(file+'v.bmp', cv2.IMREAD_GRAYSCALE)
    assert np.sum(bimg2!=img3) == 0, 'Wrong segementation!'

    # 验证连通性标记正确性
    num_label, limg3 = cv2.connectedComponents(img3, connectivity=4)
    img4 = cv2.imread(file+'bw.bmp', cv2.IMREAD_GRAYSCALE)
    assert num_label == len(np.unique(img4)), 'Wrong connected compononets labeling!'

    coords = np.array(get_star_centroids(
        img0, 
        'MEDIAN',
        'Liebe5',
        'CCL',
        'MCoG',
        5,
    ))
    print(coords.shape)


# 展示降噪-提取各阶段效果
if False:
    img0 = cv2.imread('./example/xie/20161227225347/20161227225347.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img0', img0)

    h, w = img0.shape
    y0, x0 = h/2, w/2
    f0 = 34000/6.7
    
    img1 = denoise_image(img0, 'BLF')
    cv2.imshow('img1', img1)

    T = cal_threshold(img1, 'Liebe3')
    print('mean+3*std', T)

    coords = get_seed_coords(img1, 5, T, 10100, 1.2*T)
    img2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for coord in coords:
        cv2.circle(img2, (int(coord[1]), int(coord[0])), 5, (255, 0, 0), 1)
    cv2.imshow('img2', img2)

    cv2.waitKey(-1)


# 验证提取算法有效性——分别计算恒星在星敏感器坐标系以及天球坐标系下角距，比较对应值大小
if False:
    name = 'cdata'

    h, w = 1024, 1280

    # 角距35mm/像元尺寸5.5um
    f = 35269.52/5.5

    data = np.load(f'./example/xie/{name}/{name}.npz', allow_pickle=True)

    # idxs需要减一，因为matlab中序号计数从1开始
    idxs = data['idxs']-1
    ids = data['ids']

    # 天球坐标系下矢量
    stars = cata[cata['Star ID'].isin(ids)].copy()
    stars['Star ID'] = pd.Categorical(stars['Star ID'], categories=ids, ordered=True)
    stars = stars.sort_values('Star ID') # 将stars按照ids排序
    stars['X'] = np.cos(stars['Ra'])*np.cos(stars['De'])
    stars['Y'] = np.sin(stars['Ra'])*np.cos(stars['De'])
    stars['Z'] = np.sin(stars['De'])
    V1 = stars[['X', 'Y', 'Z']].to_numpy()

    # 星敏感器坐标系下矢量
    # matlab中计算结果
    V2 = data['points'][idxs]
    V2[:, 0] = V2[:, 0] - h/2
    V2[:, 1] = V2[:, 1] - w/2
    V2[:, 2] = f

    print(np.allclose(get_angdist(V1), get_angdist(V2), atol=1e-4))

    # 本方法计算结果
    # coords = get_star_centroids(
    #     cv2.imread(f'./example/xie/{name}/{name}.bmp', cv2.IMREAD_GRAYSCALE),
    #     'MEDIAN', 
    #     'Liebe3', 
    #     'CCL', 
    #     'CoG', 
    #     pixel_limit=3
    # )


# 使用单张图片验证识别算法有效性，并在原图中标出恒星ID
if False:
    # test image
    img_path = './example/xie/cdata/cdata.bmp'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # test image config
    h, w, f = 1024, 1280, 35269.52/5.5

    # guide star catalogue
    gcata_path = 'catalogue/sao5.5_d0.03_9_10.csv'
    
    # generation parameters
    method = 'rac_nn'

    # parameters
    simu_params = {
        'h': h,
        'w': w,
        'fovy': 2*np.degrees(np.arctan(h/(2*f))),
        'fovx': 2*np.degrees(np.arctan(w/(2*f))),
        'limit_mag': 5.5,
        'rot': 1
    }
    meth_params = {
        'rac_nn': [
            0.1,            # Rb
            4.5,            # Rp
            [25, 55, 85],   # arr_ring
            18,             # num_sector
            3,              # num_neighbor
            0,              # use_prob
        ],
    }
    extr_params = {
        'den': 'NLM_BLF',   # denoise
        'thr': 'Liebe3',    # threshold
        'seg': 'RG',        # segmentation
        'cen': 'MCoG',      # centroid
        'pixel': 3,         # pixel number limit
        'T1': None,         # optional for RG
        'T2': 0,            # optional for RG
        'T3': None,         # optional for RG
    }
    # get the pattern radius for later fov restriction
    rp = np.radians(meth_params[method][1])

    # setup all the configs
    sim_cfg, _, gcata_name, gcata = setup(simu_params, gcata_path)
    gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
    ext_cfg = '_'.join(map(str, extr_params.values()))

    # model parameters
    model_type = 'cnn3'
    model_path = os.path.join('model', sim_cfg, method, gen_cfg, model_type, 'best_model.pth')
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # test data
    df_dict = gen_real_sample(
        [img_path],
        meth_params,
        extr_params,
        f=35269.52/5.5,
    )
    df = df_dict[method]

    # dataset
    dataset = create_dataset(
        method,
        df,
        meth_params[method],
    )
    loader = DataLoader(dataset, batch_size=20, shuffle=False) # shuffle must be set to False

    # best model
    model = create_model(
        method,
        model_type,
        meth_params[method],
        len(gcata),
    )
    model.load_state_dict(torch.load(model_path))

    # predict the star ids
    esti_idxs, probs = predict(model, loader, 0, device)
    esti_ids = np.full_like(esti_idxs, -1)
    mask = esti_idxs != -1    
    esti_ids[mask] = gcata.loc[esti_idxs[mask], 'Star ID'].to_numpy()

    # get the star coordinates
    coords = df[['row', 'col']].to_numpy()

    # verify the results
    esti_ids, att_mat = verify(
        gcata,
        coords, 
        esti_ids, 
        probs=probs,
        fov=2*rp, 
        h=1024, 
        w=1280, 
        f=35269.52/5.5,
    )

    # postprocess
    esti_ids = postprocess(
        gcata,
        coords, 
        esti_ids, 
        att_mat,
        h=1024, 
        w=1280, 
        f=35269.52/5.5,
    )
    
    # sort the coords by row for label
    esti_ids = esti_ids[np.argsort(coords[:, 0])]
    coords = coords[np.argsort(coords[:, 0])]
    mask = esti_ids != -1
    
    # label star image
    # label_star_image(img, coords, auto_label=True, circle=True, axis_on=False)
    label_star_image(img, coords[mask], esti_ids[mask], circle=True, axis_on=False)


def load_h5data(dir: str, name: str):
    '''
        Load h5 data.
    '''
    with h5py.File(os.path.join(dir, name), 'r') as f:
        coords = f['/coords'][:].T
        # remove the 1 demension
        img_names = np.squeeze(f['/names'][:].T)
        cnts = np.squeeze(f['/cnts'][:].astype(np.int32).T)
        ids = np.squeeze(f['/ids'][:].astype(np.int32).T)

    start = 0
    data = []
    for i, cnt in enumerate(cnts):
        end = start + cnt
        data.append({
            'path': os.path.join(dir, str(img_names[i], 'UTF-8')),
            'coords': coords[start:end],
            'ids': ids[start:end]
        })
        start = end
    
    return data


def cal_attitude(cata: pd.DataFrame, coords: np.ndarray, ids: np.ndarray, h: int, w: int, f: int):
    '''
        Calculate the attitude matrix of star sensor.
    '''
    assert len(coords) == len(ids)
    n = len(ids)
    # get the view vectors
    vvs = np.full((n, 3), f)
    vvs[:, 0] = coords[:, 1]-w/2
    vvs[:, 1] = coords[:, 0]-h/2

    # get the refernce vectors
    stars = cata[cata['Star ID'].isin(ids)].copy()
    stars['X'] = np.cos(stars['Ra'])*np.cos(stars['De'])
    stars['Y'] = np.sin(stars['Ra'])*np.cos(stars['De'])
    stars['Z'] = np.sin(stars['De'])
    #! careful! use index to sort the stars, because the output of isin is not sorted by the given list
    stars = stars.set_index('Star ID').loc[ids]
    rvs = stars[['X', 'Y', 'Z']].to_numpy()

    return get_attitude_matrix(vvs.T, rvs.T)


# 多张实拍星图验证算法有效性
if True:
    # load test data
    # data = load_h5data('example/3P0/', '3P0_liebe5_pixel5_eps00005.h5')
    data = load_h5data('example/0P0/', '0P0_liebe5_pixel5_eps00005.h5')
    # data = load_h5data('example/xie/', 'xie_liebe3_pixel3.h5')

    # test image config
    h, w, f = 1040, 1288, 18500/4.8
    # h, w, f = 1024, 1288, 34000/6.7

    # set the number of test image
    num_img = len(data)
    data = data[:num_img]

    # get the path of test image
    img_paths = [item['path'] for item in data]

    # guide star catalogue
    gcata_path = 'catalogue/sao5.5_d0.03_9_10.csv'
    
    # generation parameters
    method = 'rac_nn'

    # parameters
    simu_params = {
        'h': h,
        'w': w,
        'fovy': 2*np.degrees(np.arctan(h/(2*f))),
        'fovx': 2*np.degrees(np.arctan(w/(2*f))),
        'limit_mag': 5.5,
        'rot': 1
    }
    meth_params = {
        'rac_nn': [
            0.5,            # Rb
            7.7,            # Rp
            [35, 75, 115],  # arr_ring
            18,             # num_sector
            3,              # num_neighbor
            0,              # use_prob
        ],
        # 'rac_nn': [
        #     0.1,            # Rb
        #     5.7,            # Rp
        #     [25, 55, 85],   # arr_ring
        #     18,             # num_sector
        #     3,              # num_neighbor
        #     0,              # use_prob
        # ],
    }
    extr_params = {
        'den': 'MEDIAN',    # denoise
        'thr': 'Liebe5',    # threshold
        'seg': 'CCL',       # segmentation
        'cen': 'MCoG',      # centroid
        'pixel': 5          # pixel number limit
    }
    # get the pattern radius for later fov restriction
    rp = np.radians(meth_params[method][1])

    # setup all the configs
    sim_cfg, _, gcata_name, gcata = setup(simu_params, gcata_path)
    gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
    ext_cfg = '_'.join(map(str, extr_params.values()))

    # full sao catalogue
    # because some of real ids may not be in the gcata
    cata = pd.read_csv('catalogue/sao.csv')

    # model parameters
    model_type = 'cnn2'
    model_path = os.path.join('model', sim_cfg, method, gen_cfg, model_type, 'best_model.pth')
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print info
    print(
        'Realshot Test',
        '\n-----------------------',
        '\nMETHOD INFO',
        '\nMethod:', method,
        '\nSimulation config:', sim_cfg,
        '\nGeneration config:', gen_cfg,
        '\nExtraction config:', ext_cfg,
        '\n-----------------------',
        '\nMODEL INFO',
        '\nModel type:', model_type,
        '\nDevice:', device,
    )

    # generate test samples for realshot
    df_dict = gen_real_sample(
        img_paths,
        meth_params,
        extr_params,
        f=f,
    )

    # test data
    df = df_dict[method]

    # dataset
    dataset = create_dataset(
        method,
        df,
        meth_params[method],
    )
    loader = DataLoader(dataset, batch_size=20, shuffle=False)

    # best model
    model = create_model(
        method,
        model_type,
        meth_params[method],
        len(gcata),
    )
    model.load_state_dict(torch.load(model_path))

    # predict the star id
    all_esti_idxs, all_probs = predict(model, loader, 0, device)
    all_esti_ids = np.full_like(all_esti_idxs, -1)
    mask = all_esti_idxs != -1    
    all_esti_ids[mask] = gcata.loc[all_esti_idxs[mask], 'Star ID'].to_numpy()

    # get the star coordinates
    all_coords = df[['row', 'col']].to_numpy()
    # need to substract 0.5 offset for later coords match
    all_coords -= 0.5

    assert len(all_esti_ids) == len(all_coords)

    # get the image id(same as image path)
    img_ids = df['img_id'].to_numpy()

    # the result dict
    res = {}

    # check the results by image
    for item in data:
        img_path, real_coords, real_ids = item['path'], item['coords'], item['ids']
        
        # get the attitude matrix
        real_atti = cal_attitude(cata, real_coords, real_ids, h=h, w=w, f=f)

        # subtract 1 offset, since row and column of matlab matrixs start with 1
        real_coords -= 1

        # get the image predications
        esti_ids = all_esti_ids[img_ids == img_path]
        esti_coords = all_coords[img_ids == img_path]
        esti_probs = all_probs[img_ids == img_path]

        # verify the results
        esti_ids, esti_atti = verify(
            gcata,
            coords=esti_coords, 
            ids=esti_ids, 
            probs=esti_probs,
            fov=2*rp, 
            h=h, 
            w=w, 
            f=f,
        )

        # postprocess
        esti_ids = postprocess(
            gcata,
            esti_coords, 
            esti_ids, 
            esti_atti,
            h=h, 
            w=w, 
            f=f,
        )

        # search for matched coordinates
        cnt = 0
        for real_coord, real_id in zip(real_coords, real_ids):
            mask = np.isclose(esti_coords, real_coord, atol=1e-1).all(axis=1)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]

            assert np.allclose(esti_coords[idx], real_coord, atol=1e-1)
            cnt += 1 if esti_ids[idx] == real_id else 0

        res[img_path] = (cnt >= 3) or np.allclose(real_atti, esti_atti, atol=1e-1)

        if DEBUG and False:
            print(
                'Image:', os.path.basename(img_path),
                '\nReal coords:\n', real_coords,
                '\nReal ids:\n', real_ids,
                '\nEsti coords:\n', esti_coords,
                '\nEsti ids:\n', esti_ids,
            )
        

    # number of successfully identified star image
    cnt = sum(res.values())
    # accuracy
    acc = cnt / len(res) * 100

    print(
        '----------------------'
        '\nTEST RESULTS'
        '\nTotal number of test star image:', len(data), 
        '\nNumber of successfully identified star image:', cnt, 
        '\nAccuracy of successfully identified star image:', acc, '%'  
    )
