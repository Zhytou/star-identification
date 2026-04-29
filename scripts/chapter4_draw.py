import os, cv2, torch, h5py
import numpy as np
import pandas as pd

from simulate import cata
from denoise import basic_filter
from detect import cal_threshold
from extract import get_star_centroids
from model import create_model
from realshot import identify_realshot_by_nn, load_h5data
from utils import gen_timestamp, cal_angdist, plot_line_chart


DEBUG = True


# 仿真实验结果作图
if True:
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
        # 'lpt_nn': {
        #     'pos': [(0.0, 98.88), (0.5, 98.89), (1.0, 98.35), (1.5, 97.97), (2.0, 97.35)],
        #     'mag': [(0.0, 100.0), (0.1, 98.56), (0.2, 98.47), (0.3, 97.14), (0.4, 94.02)],
        #     'fs': [(0.0, 99.24), (1.0, 97.97), (2.0, 96.96), (3.0, 96.24), (4.0, 95.49)],
        # },

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

    # 方法缩写
    abbr_2_name = {
        'rac_nn': '本文方法',
        'lpt_nn': '基于Polestar模式的神经网络算法',
        'grid': '栅格算法',
        'lpt': '改进的LPT算法'
    }
    # 测试缩写
    type_2_name = {
        'pos': '位置噪声(pixel)',
        'mag': '亮度噪声(Mv)',
        'fs': '伪星数目',
        # 'ms': '缺失星数目'
    }

    dir = os.path.join('res/chapter4/sim', gen_timestamp())
    
    # 英文缩写替换为中文名称
    for mabbr, mname in abbr_2_name.items():
        if mabbr not in res:
            continue
        res[mname] = res.pop(mabbr)
        for tabbr, tname in type_2_name.items():
                if tabbr not in res[mname]:
                    continue
                res[mname][tname] = res[mname].pop(tabbr)

    # 作直线图
    for tname in type_2_name.values():
        sub_res = {}
        for mname in abbr_2_name.values():
            if mname not in res or tname not in res[mname]:
                continue
            sub_res[mname] = res[mname][tname]
        plot_line_chart(sub_res, xlabel='噪声强度', ylabel='识别率(%)', yrange=(80, 100), img_name=tname+'.png', show=True, output_dir=dir)


# 模型内存和计算消耗统计
if False:
    gcata = pd.read_csv('./catalogue/sao6.0_d0.03_12_15.csv')
    num_class = len(gcata)
    
    from torchsummary import summary
    from thop import profile
    from torchstat import stat

    # proposed
    model = create_model(
        'rac_nn',
        'cnn3',
        [0.5, 6, [15, 35, 55], 18, 3, 0],
        num_class
    )
    input = torch.randn(1, 213)

    from test import eval_time
    print(eval_time(model, input))
    flops, params = profile(model, inputs=(input, ))
    print(
        flops, params,
        round(flops / (1024 * 1024), 2),
        round(params / (1024 * 1024) * 4, 2)
    )

    # model = create_model(
    #     'rac_nn',
    #     'cnn4',
    #     [0.5, 6, [15, 35, 55], 18, 3, 0],
    #     num_class
    # )
    # input = torch.randn(1, 213)
    # flops, params = profile(model, inputs=(input, ))
    # print(
    #     flops, params,
    #     round(flops / (1024 * 1024), 2),
    #     round(params / (1024 * 1024) * 4, 2)
    # )
    
    # Rijlaarsdam
    model = create_model(
        'lpt_nn',
        'fnn',
        [0.5, 6, 55, 0],
        num_class
    )
    input = torch.randn(1, 55)
    print(eval_time(model, input))
    flops, params = profile(model, inputs=(input, ))
    print(
        flops, params,
        round(flops / (1024 * 1024), 2),
        round(params / (1024 * 1024) * 4, 2)
    )

    from model import MIFNet, RPNet, GridVgg
    # model = MIFNet()
    # input = torch.randn(1, 11, 32)
    # flops, params = profile(model, inputs=(input, ))
    # print(
    #     flops, params,
    #     round(flops / (1024 * 1024), 2),
    #     round(params / (1024 * 1024) * 4, 2)
    # )

    model = RPNet()
    input = torch.randn(1, 400)
    print(eval_time(model, input))

    flops, params = profile(model, inputs=(input, ))
    print(
        flops, params,
        round(flops / (1024 * 1024), 2),
        round(params / (1024 * 1024) * 4, 2)
    )

    model = GridVgg()
    input = torch.randn(1, 1, 224, 224)
    print(eval_time(model, input))

    flops, params = profile(model, inputs=(input, ))
    print(
        flops, params,
        round(flops / (1024 * 1024), 2),
        round(params / (1024 * 1024) * 4, 2)
    )


# 验证降噪\二值化\连通域等算法和matlab实现一致性
if False:
    dir = 'realshot/xie/cdata'
    name = 'cdata'
    exten = '.bmp'

    img0 = cv2.imread(os.path.join(dir, name+exten), cv2.IMREAD_GRAYSCALE)

    # 验证中值滤波正确性
    img1 = basic_filter(img0, 'MMedian')
    img2 = cv2.imread(os.path.join(dir, name+'_median'+exten), cv2.IMREAD_GRAYSCALE)
    assert np.sum(img1!=img2) == 0, 'Wrong median filter!'

    # 验证阈值计算以及二值化正确性
    thr_meth = 'Liebe3'
    T = cal_threshold(img1, thr_meth)
    bimg2 = np.where(img2 >= T, 1, 0)
    img3 = cv2.imread(os.path.join(dir, thr_meth, name+'_binary'+exten), cv2.IMREAD_GRAYSCALE)
    assert np.sum(bimg2!=img3) == 0, 'Wrong segementation!'

    # 验证连通性标记正确性
    num_label, limg3 = cv2.connectedComponents(img3, connectivity=4)
    img4 = cv2.imread(os.path.join(dir, thr_meth, name+'_label'+exten), cv2.IMREAD_GRAYSCALE)
    assert num_label == len(np.unique(img4)), 'Wrong connected compononets labeling!'

    esti_coords = get_star_centroids(
        img0, 'MMedian', ['None', thr_meth, 'CCL', 'None'], 
        'MCoG', pixel_limit=5, gray=True
    )
    # matlab starpoints结果为行、列以及灰度和
    # 其中，由于matlab计算时从1开始，所以理论上应该比esti_coords大0.5
    with h5py.File(os.path.join(dir, thr_meth, 'starpoints.h5'), 'r') as f:
        real_coords = f['/points'][:].T
        print(real_coords, esti_coords)


# 验证提取算法有效性——分别计算恒星在星敏感器坐标系以及天球坐标系下角距，比较对应值大小
if False:
    name = 'cdata'

    h, w = 1024, 1280

    # 角距35mm/像元尺寸5.5um
    f = 35269.52/5.5

    data = np.load(f'./realshot/xie/{name}/{name}.npz', allow_pickle=True)

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

    print(np.allclose(cal_angdist(V1), cal_angdist(V2), atol=1e-4))


# 使用单张图片验证识别算法有效性，并在原图中标出恒星ID
if False:
    img_path = 'realshot/xie/cdata/cdata.bmp'
    save_dir = 'res/chapter4/single_realshot'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    h, w, f = 1024, 1280, 35269.52/5.5
    simu_params={
        'h': h,
        'w': w,
        'f': f,
        'fovy': 2*np.degrees(np.arctan(h/(2*f))),
        'fovx': 2*np.degrees(np.arctan(w/(2*f))),
        'limit_mag': 5.5,
        'rot': 1
    }
    meth_params={
        'rac_nn': [
            0.1,            # Rb
            4.5,            # Rp
            [25, 55, 85],   # arr_ring
            18,             # num_sector
            3,              # num_neighbor
            0,              # use_prob
        ],
    }
    extr_params={
        'den': 'None',      # denoise
        'seg': ['None', 'Liebe3', 'CCL', 'None'], # segmentation: [enhancement, threshold, label, operator]
        'cen': 'MCoG',      # centroid
        'pixel': 3,         # pixel number limit
    }
    df_dict = identify_realshot_by_nn(
        [img_path],
        simu_params=simu_params,
        meth_params=meth_params,
        extr_params=extr_params,
        model_types={
            'rac_nn': 'cnn3'
        },
        gcata_path='catalogue/sao5.5_d0.03_9_10.csv', # guide star catalogue
        output_dir=os.path.join(save_dir, gen_timestamp())
    )    


# 多张实拍星图验证算法有效性
if False:
    # 清华测试图像大小以及拍摄焦距
    h, w, f = 1040, 1288, 18500/4.8
    # 和matlab结果比较时，坐标误差阈值
    cen_err_threshold = 0.2
    
    # 参数
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
        'den': 'MMedian',   # denoise
        'seg': ['None', 'Liebe5', 'CCL', 'None'], # segmentation: [enhancement, threshold, label, operator]
        'cen': 'MCoG',      # centroid
        'pixel': 5          # pixel number limit
    }

    data, dfs, img_paths = [], [], []
    test_dir = 'realshot/tsinghua'
    test_num = 100
    test_prefixs = ['0P0', ]
    # target_img_paths = ['realshot/tsinghua/0P0/00000008_000000000198857B.bmp', 'realshot/tsinghua/1P0/00005088_0000000001B4D132.bmp', 'realshot/tsinghua/1P0/00005188_0000000001B51271.bmp', 'realshot/tsinghua/2P0/00003067_0000000001A9C63A.bmp', 'realshot/tsinghua/3P0/00001051_00000000019D162E.bmp']
    save_dir = 'res/chapter4/multiple_realshot'
    for prefix in test_prefixs:
        # 加载每个数据集中的测试数据
        subdata = load_h5data(os.path.join(test_dir, prefix), f'{prefix}_liebe5_pixel5_eps00005.h5')[:test_num]
        subimg_paths = [item['path'] for item in subdata]
        # subimg_paths = [item['path'] for item in subdata if item['path'] in target_img_paths]

        # 使用模型进行识别，并保存识别结果
        df_dict = identify_realshot_by_nn(
            subimg_paths, 
            simu_params,
            meth_params,
            extr_params,
            model_types={
                'rac_nn': 'lcnn',
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv', # guide star catalogue，
            eps0=3e-4, # threshold for verify
            eps1=3e-4, # threshold for postprocess triangle match
            eps2=1e-3, # threshold for postprocess unidentified star angular match
            # output_dir=os.path.join(save_dir, prefix),
        )
        dfs.append(df_dict['rac_nn'])
        data.extend(subdata)
        img_paths.extend(subimg_paths)

    df = pd.concat(dfs, ignore_index=True, copy=False)
    # 每张测试图片正确识别恒星数量
    res = {}
    # 错误识别图片列表
    failed_img_paths = []

    for item in data:
        img_path, real_coords, real_ids = item['path'], item['coords'], item['ids']
        if img_path not in img_paths:
            continue
        
        # 读取每张图片测试结果
        esti = df.loc[df['img_id']==img_path, ['star_id', 'row', 'col', 'gray', 'valid', 'verified']].to_numpy()
        esti_ids, esti_coords, grays, flags = esti[:, 0].astype(int), esti[:, 1:3].astype(float), esti[:, 3].astype(int), esti[:, 4:6].astype(bool)

        # 由于MATLAB矩阵的行和列都以1开头，因此需要加上0.5的偏移量。
        esti_coords += 0.5

        cnt = 0
        # 和matlab识别结果进行比较
        for real_coord, real_id in zip(real_coords, real_ids):
            mask = np.isclose(esti_coords, real_coord, atol=cen_err_threshold).all(axis=1)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            idx = idx[0]

            assert np.allclose(esti_coords[idx], real_coord, atol=cen_err_threshold)
            cnt += 1 if esti_ids[idx] == real_id else 0
        
        res[img_path] = cnt
        if cnt < 3:
            failed_img_paths.append(img_path)
        
        if DEBUG and cnt < 3: 
            print(
                'Image:', os.path.basename(img_path),
                '\nNumber of stars:', len(flags),
                '\nNumber of valid patterns:', np.sum(flags[:, 0]),
                '\nNumber of verified patterns:', np.sum(flags[:, 1]),
                '\nNumber of correct match:', cnt, 
                '\nReal coords:\n', real_coords,
                '\nReal ids:\n', real_ids,
                '\nEsti coords:\n', esti_coords,
                '\nEsti ids:\n', esti_ids,
                '\n',
            )

    # 删除错误识别图像
    for img_path in failed_img_paths:
        prefix = os.path.basename(os.path.dirname(img_path))
        img_id = os.path.splitext(os.path.basename(img_path))[0]+'.png'
        dir = os.path.join(save_dir, prefix, 'identify')
        if img_id in os.listdir(dir):
            os.remove(os.path.join(dir, img_id))

    # 平均正确识别恒星数量
    avg_sistar_cnt = sum(map(lambda x: res[x], res)) / len(data)
    # 平均有效（模型输出概率大于阈值）数量
    avg_rstar_cnt = df.groupby('img_id')['valid'].sum().mean()
    # 平均识别恒星数量
    avg_istar_cnt = df[df['star_id']!=-1].groupby('img_id').size().mean()
    # 平均测试恒星数量
    avg_star_cnt = df.groupby('img_id').size().mean()
    
    # 成功识别图像数量
    img_cnt = sum(map(lambda x: res[x]>=3, res))
    # 正确率
    acc = img_cnt / len(data) * 100

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
