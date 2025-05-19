import os, cv2
import numpy as np
import pandas as pd

from simulate import cata
from denoise import filter_image, denoise_image
from detect import cal_threshold, get_seed_coords
from extract import get_star_centroids
from test import draw_results
from realshot import identify_realshot_by_nn, cal_attitude, load_h5data
from utils import get_angdist, label_star_image


DEBUG = True


# 仿真实验
if False:
    res = {
        # grid 0.5_6_100 T=3.6
        # 'grid': {
        #     'pos': [(0.0, 96.84), (0.5, 95.44), (1.0, 92.18), (1.5, 89.63), (2.0, 84.49)],
        #     'mag': [(0.0, 98.58), (0.1, 95.17), (0.2, 92.84), (0.3, 90.91), (0.4, 83.06)],
        #     'fs': [(0.0, 93.26), (1.0, 92.08), (2.0, 93.19), (3.0, 91.19), (4.0, 88.99)],
        # },

        # grid 0.5_6_100 T=3.4
        'grid': {
            'pos': [(0.0, 98.17), (0.5, 97.16), (1.0, 95.47), (1.5, 92.58), (2.0, 86.84)],
            'mag': [(0.0, 99.09), (0.1, 96.4), (0.2, 94.58), (0.3, 92.85), (0.4, 85.8)],
            'fs': [(0.0, 95.17), (1.0, 93.99), (2.0, 93.49), (3.0, 92.69), (4.0, 91.19)],
        },

        # grid 0.5_6_100 T=3.2
        # 'grid': {
        #     'pos': [(0.0, 98.47), (0.5, 97.59), (1.0, 96.15), (1.5, 93.89), (2.0, 87.63)],
        #     'mag': [(0.0, 99.24), (0.1, 96.65), (0.2, 94.65), (0.3, 93.49), (0.4, 86.91)],
        #     'fs': [(0.0, 95.25), (1.0, 93.5), (2.0, 93.88), (3.0, 93.24), (4.0, 91.62)],
        # },

        # lpt 0.5_6_50_36 T=4
        'lpt': {
            'pos': [(0.0, 97.66), (0.5, 96.15), (1.0, 95.37), (1.5, 93.5), (2.0, 92.45)],
            'mag': [(0.0, 98.48), (0.1, 95.48), (0.2, 94.39), (0.3, 91.06), (0.4, 85.77)],
            'fs': [(0.0, 96.31), (1.0, 95.3), (2.0, 94.93), (3.0, 92.61), (4.0, 91.85)],
        },

        # lpt_nn 0.5_6_55_0 T=0.3
        'lpt_nn': {
            'pos': [(0.0, 98.88), (0.5, 98.89), (1.0, 98.35), (1.5, 97.97), (2.0, 97.35)],
            'mag': [(0.0, 100.0), (0.1, 98.56), (0.2, 98.47), (0.3, 97.14), (0.4, 94.02)],
            'fs': [(0.0, 99.24), (1.0, 97.97), (2.0, 96.96), (3.0, 96.24), (4.0, 95.49)],
        },

        # lpt_nn 0.5_6_55_0 T=0.5
        'lpt_nn': {
            'pos': [(0.0, 98.57), (0.5, 98.58), (1.0, 98.25), (1.5, 97.66), (2.0, 96.53)],
            'mag': [(0.0, 99.0), (0.1, 97.94), (0.2, 97.36), (0.3, 95.22), (0.4, 92.49)],
            'fs': [(0.0, 98.73), (1.0, 97.22), (2.0, 96.32), (3.0, 95.49), (4.0, 92.86)],
        },

        # lpt_nn 0.5_6_55_0 T=0.7
        # 'lpt_nn': {
        #     'pos': [(0.0, 98.06), (0.5, 98.07), (1.0, 97.53), (1.5, 96.14), (2.0, 94.9)],
        #     'mag': [(0.0, 99.8), (0.1, 97.23), (0.2, 97.03), (0.3, 94.69), (0.4, 89.55)],
        #     'fs': [(0.0, 98.22), (1.0, 96.46), (2.0, 95.18), (3.0, 92.98), (4.0, 90.98)]
        # }

        # rac_nn 0.5_6_25_55_85_18_3 T=0.5
        'rac_nn': {
            'pos': [(0.0, 99.39), (0.5, 99.49), (1.0, 99.49), (1.5, 99.09), (2.0, 98.06)],
            'mag': [(0.0, 100.0), (0.1, 99.38), (0.2, 99.08), (0.3, 98.88), (0.4, 95.44)],
            'fs': [(0.0, 99.62), (1.0, 98.86), (2.0, 97.97), (3.0, 96.62), (4.0, 95.74)]
        },

        # rac_nn 0.5_6_25_55_85_18_3 T=0.7
        # 'rac_nn': {
        #     'pos': [(0.0, 99.38), (0.5, 99.19), (1.0, 98.97), (1.5, 98.48), (2.0, 97.14)],
        #     'mag': [(0.0, 99.9), (0.1, 99.18), (0.2, 98.98), (0.3, 97.65), (0.4, 93.61)],
        #     'fs': [(0.0, 99.36), (1.0, 97.85), (2.0, 96.96), (3.0, 95.36), (4.0, 92.86)]
        # }
    }

    draw_results(res, save=True)


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
    
    # predict the star ids
    df_dict = identify_realshot_by_nn(
        [img_path],
        simu_params={
            'h': h,
            'w': w,
            'f': f,
            'fovy': 2*np.degrees(np.arctan(h/(2*f))),
            'fovx': 2*np.degrees(np.arctan(w/(2*f))),
            'limit_mag': 5.5,
            'rot': 1
        },
        meth_params={
            'rac_nn': [
                0.1,            # Rb
                4.5,            # Rp
                [25, 55, 85],   # arr_ring
                18,             # num_sector
                3,              # num_neighbor
                0,              # use_prob
            ],
        },
        extr_params={
            'den': 'NLM_BLF',   # denoise
            'thr': 'Liebe3',    # threshold
            'seg': 'RG',        # segmentation
            'cen': 'MCoG',      # centroid
            'pixel': 3,         # pixel number limit
            'T1': None,         # optional for RG
            'T2': 0,            # optional for RG
            'T3': None,         # optional for RG
        },
        model_types={
            'rac_nn': 'cnn3'
        },
        gcata_path = 'catalogue/sao5.5_d0.03_9_10.csv' # guide star catalogue
    )    

    # get esti coords and id
    df = df_dict['rac_nn']
    esti = df.loc[df['img_id']==img_path, ['star_id', 'row', 'col']].to_numpy()
    ids, coords = esti[:, 0].astype(int), esti[:, 1:3]

    # sort the coords by row for label
    ids = ids[np.argsort(coords[:, 0])]
    coords = coords[np.argsort(coords[:, 0])]
    mask = ids != -1
    
    # label star image
    label_star_image(img, coords, auto_label=True, circle=True, axis_on=False)
    label_star_image(img, coords[mask], ids[mask], circle=True, axis_on=False)


# 多张实拍星图验证算法有效性
if True:
    # load test data
    data = []
    for prefix in [
        '0P0', 
        '1P0', 
        '2P0',
        '3P0'
    ]:
        data.extend(load_h5data(f'example/{prefix}/', f'{prefix}_liebe5_pixel5_eps00005.h5')) 

    # get the path of test image
    # target_paths = ['00000064_000000000198AA97.bmp', '00000071_000000000198AF3A.bmp', '00000084_000000000198B7D6.bmp', '00000129_000000000198D4C4.bmp', '00000255_000000000199265E.bmp']
    # img_paths = [item['path'] for item in data if os.path.basename(item['path']) in target_paths]
    img_paths = [item['path'] for item in data]

    # test image config
    h, w, f = 1040, 1288, 18500/4.8

    # parameters
    simu_params = {
        'h': h,
        'w': w,
        'f': f,
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
    }
    extr_params = {
        'den': 'MEDIAN',    # denoise
        'thr': 'Liebe5',    # threshold
        'seg': 'CCL',       # segmentation
        'cen': 'MCoG',      # centroid
        'pixel': 5          # pixel number limit
    }

    # identify realshots
    df_dict = identify_realshot_by_nn(
        img_paths, 
        simu_params,
        meth_params,
        extr_params,
        model_types={
            'rac_nn': 'cnn3',
        },
        gcata_path='catalogue/sao5.5_d0.03_9_10.csv', # guide star catalogue，
        eps1=5e-5,
        eps2=1e-2,
        # output_dir='res/chapter4/realshot',
    )

    # only take rac results
    df = df_dict['rac_nn']

    # the result dict
    res = {}
    # failed star image paths
    failed_img_paths = []

    # check the results by image
    for item in data:
        img_path, real_coords, real_ids = item['path'], item['coords'], item['ids']

        if img_path not in img_paths:
            continue

        # get esti coords and id
        esti = df.loc[df['img_id']==img_path, ['star_id', 'row', 'col', 'gray', 'valid', 'verified']].to_numpy()
        esti_ids, esti_coords, grays, flags = esti[:, 0].astype(int), esti[:, 1:3].astype(float), esti[:, 3].astype(int), esti[:, 4:6].astype(bool)

        # add 0.5 offset, since row and column of matlab matrixs start with 1
        esti_coords += 0.5

        # get the attitude matrix
        real_atti = cal_attitude(cata, real_coords, real_ids, h=h, w=w, f=f)

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

        # add to result
        res[img_path] = cnt

        if DEBUG and cnt < 3:
            failed_img_paths.append(img_path)
            
            # print debug info
            print(
                'Image:', os.path.basename(img_path),
                # '\nReal coords:\n', real_coords,
                # '\nReal ids:\n', real_ids,
                # '\nReal attitude:\n', real_atti,
                # '\nEsti coords:\n', esti_coords,
                # '\nEsti ids:\n', esti_ids,
                # '\nEsti attitude:\n', esti_atti,
                '\nNumber of valid patterns:', np.sum(flags[:, 0]),
                '\nNumber of verified patterns:', np.sum(flags[:, 1]),
                '\nNumber of correct match:', cnt, 
                '\nNumber of stars:', len(flags),
                '\n',
            )
    
    # average number of successfully identified star in each image
    avg_sistar_cnt = sum(map(lambda x: res[x], res)) / len(data)
    # average number of valid reference star in each image
    avg_rstar_cnt = df.groupby('img_id')['valid'].sum().mean()
    # average number of identified star in each image
    avg_istar_cnt = df[df['star_id']!=-1].groupby('img_id').size().mean()
    # average number of star in each image
    avg_star_cnt = df.groupby('img_id').size().mean()
    
    # number of successfully identified star image
    img_cnt = sum(map(lambda x: res[x]>=3, res))
    # accuracy
    acc = img_cnt / len(data) * 100

    # describe the failed test star images
    if DEBUG:
        # df = gen_real_sample(failed_img_paths, meth_params, extr_params, simu_params['f'])['rac_nn']
        df = df[df['img_id'].isin(failed_img_paths)]
        print(df.groupby('img_id')[['valid', 'verified']].sum())        

    print(
        'TEST RESULTS'
        '\nAverage number of successfully identified stars in each image:', avg_sistar_cnt, #只能反映和matlab中代码识别相同的数量，受matlab代码识别限制
        '\nAverage number of identified stars in each image:', avg_istar_cnt,
        '\nAverage number of reference stars in each image:', avg_rstar_cnt, 
        '\nAverage number of stars in each image:', avg_star_cnt, 
        '\nNumber of successfully identified star image:', img_cnt, 
        '\nTotal number of test star image:', len(data), 
        '\nAccuracy of successfully identified star image:', acc, '%'  
    )
