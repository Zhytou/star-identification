import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from simulate import cata
from denoise import filter_image, denoise_image
from detect import cal_threshold, get_seed_coords, group_star
from extract import get_star_centroids
from generate import gen_real_sample
from dataset import create_dataset
from model import create_model
from test import predict
from utils import get_angdist


# 验证降噪\二值化\连通域等算法和matlab实现一致性
if False:
    img0 = cv2.imread('./example/xie/20161227225347/20161227225347.bmp', cv2.IMREAD_GRAYSCALE)
    img1 = filter_image(img0, 'MEDIAN')

    img2 = cv2.imread('./example/xie/20161227225347/20161227225347_f.bmp', cv2.IMREAD_GRAYSCALE)
    print(np.sum(img1!=img2))

    T = cal_threshold(img1, 'Liebe')
    print(T)

    bimg2 = np.zeros_like(img2)
    bimg2[img2 >= T] = 1

    img3 = cv2.imread('./example/xie/20161227225347/20161227225347_v.bmp', cv2.IMREAD_GRAYSCALE)
    print(np.sum(bimg2!=img3))

    num_label, limg3 = cv2.connectedComponents(img3, connectivity=4)
    print(num_label)

    img4 = cv2.imread('./example/xie/20161227225347/20161227225347_bw.bmp', cv2.IMREAD_GRAYSCALE)
    val = np.unique(img4[img4!=0])
    print(len(val))

    rows, cols = np.nonzero(limg3)
    labels = limg3[rows, cols]

    ulabels, ucnts = np.unique(labels, return_counts=True)
    print(len(ucnts[ucnts>=3]))

    coords = group_star(img1, 'CCL', T, connectivity=4, pixel_limit=3)
    print(len(coords))


# 展示降噪-提取各阶段效果
if False:
    img0 = cv2.imread('./example/xie/20161227225347/20161227225347.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('img0', img0)

    h, w = img0.shape
    y0, x0 = h/2, w/2
    f0 = 34000/6.7
    
    img1 = denoise_image(img0, 'BLF')
    cv2.imshow('img1', img1)

    T = cal_threshold(img1, 'Liebe')
    print('mean+3*std', T)

    coords = get_seed_coords(img1, 5, T, 10100, 1.2*T)
    img2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for coord in coords:
        cv2.circle(img2, (int(coord[1]), int(coord[0])), 5, (255, 0, 0), 1)
    cv2.imshow('img2', img2)

    cv2.waitKey(-1)


# 验证提取算法有效性——分别计算恒星在星敏感器坐标系以及天球坐标系下角距，比较对应值大小
if False:
    # name = 'cdata'
    name = '20161227225347'

    h, w = 1024, 1280
    if name != 'cdata':
        # 角距34mm/像元尺寸6.7um
        f = 34000/6.7
    else:
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
    #     'Liebe', 
    #     'CCL', 
    #     'CoG', 
    #     pixel_limit=3
    # )


# 验证识别算法有效性
if True:
    gcata = pd.read_csv('catalogue/sao5.5_d0.03_9_10.csv', usecols=["Star ID", "Ra", "De", "Magnitude"])

    img_paths = [
        './example/xie/cdata/cdata.bmp'
    ]

    meth_params = {
        'rac_1dcnn': [
            0.1,            # Rb
            4.5,            # Rp
            [50, 100],   # arr_ring
            16,             # num_sector
            3,              # num_neighbor
        ],
        # 'grid': [
        #     0.1,            # Rb
        #     4.5,            # Rp
        #     50,             # Grid Length
        # ]
    }

    data = np.load(img_paths[0].replace('.bmp', '.npz'), allow_pickle=True)


    df_dict = gen_real_sample(
        img_paths,
        meth_params,
        f=35269.52/5.5,
    )

    for method in df_dict:
        # test data
        df = df_dict[method]

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # dataset
        dataset = create_dataset(
            'rac_1dcnn',
            df,
            '[50, 100]_16_3',
        )
        loader = DataLoader(dataset, batch_size=20, shuffle=False)

        # best model
        model = create_model(
            method,
            meth_params[method],
            len(gcata),
        )
        model_path = 'model/1024_1280_11.398822251559647_9.129887427521604_5.5/rac_1dcnn/sao5.5_d0.03_9_10_0.1_4.5_[50, 100]_16_3/best_model.pth'
        model.load_state_dict(torch.load(model_path))

        # predict the star id
        all_esti_idxs = predict(method, model, loader, 0.1, torch.device('cpu'))
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

        # check the results by image
        for img_path in img_paths:
            cnt = 0

            # get the image predications
            esti_ids = all_esti_ids[img_ids == img_path]
            esti_coords = all_coords[img_ids == img_path]

            # get the real star id and corresponding coordinates
            data = np.load(img_path.replace('.bmp', '.npz'), allow_pickle=True)
            points = data['points'][data['idxs']-1]
            real_ids = data['ids']
            
            # swap row and col
            real_coords = np.vstack((points[:, 1], points[:, 0])).transpose()

            # search for matched coordinates
            for real_coord, real_id in zip(real_coords, real_ids):
                mask = np.isclose(esti_coords, real_coord, atol=1e-3).all(axis=1)
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                idx = idx[0]

                assert np.allclose(esti_coords[idx], real_coord, atol=1e-3)
                cnt += 1 if esti_ids[idx] == real_id else 0
            
                print(
                    '\nCoord', real_coord,
                    '\nEstimated star id', esti_ids[idx],
                    '\nReal star id', real_id, 
                )
                
            print(
                '------------------------------',
                '\nMethod', method,
                '\nImage', os.path.basename(img_path),
                '\nNumber of stars', len(esti_ids),
                '\nMatched stars', cnt,
            )

