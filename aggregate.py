import os, uuid, json
import pandas as pd
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from generate import setup, gen_sample, gen_dataset


def agg_dataset(meth_params: dict, simu_params: dict, gcata_path: str, offset: float=1, num_roll: int=360, num_thd: int=20):
    '''
        Aggregate the training datasets.
    Args:
        meth_params: the parameters of methods, possible methods include:
            'rac_nn': Rb Rp arr_Nr Ns Nn use_prob
            'lpt_nn': Rb Rp Nd use_prob
            'grid': Rb Rp Ng
            'lpt': Rb Rp Nd Nt
        test_params: the parameters for the test sample generation
            'pos': the standard deviation of the positional noise
            'mag': the standard deviation of the magnitude noise
            'fs': the number of false stars
            'ms': the number of missing stars
        simu_params: the parameters for the simulation
            'h': the height of the image
            'w': the width of the image
            'fovx': the field of view in x direction
            'fovy': the field of view in y direction
            'limit_mag': the limit magnitude
            'pixel': the pixel size
        gcata_path: the path to the guide star catalogue
        offset: the offset of degrees to be added to the RA and Dec of the stars
        num_roll: the number of rolls to be applied to the stars
        num_thd: the number of threads to generate the test samples
    '''

    if meth_params == {}:
        return 

    print('Dataset Generation')
    print('------------------')
    print('Method parameters:', meth_params)
    print('Simulation parameters:', simu_params)
    print('Guide star catalogue:', gcata_path)
    print('Offset:', offset)
    print('Number of rolls:', num_roll)
    print('----------------------')

    # setup gcata and several configs
    sim_cfg, noise_cfg, gcata_name, gcata = setup(simu_params, gcata_path)

    pool = ThreadPoolExecutor(max_workers=num_thd)
    tasks = []

    id_ra_des = gcata[['Star ID', 'Ra', 'De']].to_numpy()
    for cata_idx, (star_id, star_ra, star_de) in enumerate(id_ra_des):
        # generate the dataset
        tasks.append(pool.submit(
            gen_dataset,
            meth_params,
            simu_params,
            int(star_id),
            cata_idx,
            star_ra,
            star_de,
            offset=offset,
            num_roll=num_roll,
        ))

    mdf_dict = defaultdict(list)
    for task in tasks:
        df_dict = task.result()
        for method in df_dict:
            mdf_dict[method].append(df_dict[method])
    
    # csv file name
    name = f'{uuid.uuid1()}_{num_roll}'
    for method in mdf_dict:
        # dataset path for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        ds_path = os.path.join('dataset', sim_cfg, method, gen_cfg, str(offset), noise_cfg)
        
        # makedirs and store the dataset
        os.makedirs(ds_path, exist_ok=True)
        df = pd.concat(mdf_dict[method], ignore_index=True, copy=False)
        df.to_csv(
            os.path.join(ds_path, name),
            index=False
        )

    return


def agg_sample(num_img: int, meth_params: dict, simu_params: dict, test_params: dict, gcata_path: str, num_thd: int = 20):
    '''
        Aggregate the test samples. 
    Args:
        num_img: number of test images expected to be generated
        meth_params: the parameters of methods, possible methods include:
            'rac_nn': Rb Rp arr_Nr Ns Nn use_prob
            'lpt_nn': Rb Rp Nd use_prob
            'grid': Rb Rp Ng
            'lpt': Rb Rp Nd Nt
        test_params: the parameters for the test sample generation
            'pos': the standard deviation of the positional noise
            'mag': the standard deviation of the magnitude noise
            'fs': the number of false stars
            'ms': the number of missing stars
        simu_params: the parameters for the simulation
            'h': the height of the image
            'w': the width of the image
            'fovx': the field of view in x direction
            'fovy': the field of view in y direction
            'limit_mag': the limit magnitude
            'pixel': the pixel size
        num_thd: the number of threads to generate the test samples
    '''

    if num_img == 0:
        return

    print('Test Image Generation')
    print('----------------------')
    print('Method Parameters:', meth_params)
    print('Simulation Parameters:', simu_params)
    print('Number of test images expected to be generated:', num_img)
    print('----------------------')

    # setup gcata and several configs
    sim_cfg, _, gcata_name, gcata = setup(simu_params, gcata_path)

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thd)
    # tasks for later aggregation
    tasks = defaultdict(list)

    # add tasks to the thread pool
    for pos in test_params.get('pos', []):
        tasks[f'pos{pos}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, sigma_pos=pos))
    for mag in test_params.get('mag', []):
        tasks[f'mag{mag}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, sigma_mag=mag))
    for fs in test_params.get('fs', []):
        tasks[f'fs{fs}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, num_fs=fs))
    for ms in test_params.get('ms', []):
        tasks[f'ms{ms}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, num_ms=ms))
    
    # sub test name
    for st_name in tasks:
        for task in tasks[st_name]:
            # get the async task result and store the returned dataframe
            df_dict = task.result()
            for method in df_dict:
                gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
                st_path = os.path.join('test', sim_cfg, method, gen_cfg, st_name)
                if not os.path.exists(st_path):
                    os.makedirs(st_path)
                df = df_dict[method]
                df.to_csv(os.path.join(st_path, str(uuid.uuid1())), index=False)

    # aggregate all the test patterns
    for method in meth_params:
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        path = os.path.join('test', sim_cfg, method, gen_cfg)
        if not os.path.exists(path):
            continue
        # sub test dir names
        test_names = list(tasks.keys())
        for tn in test_names:
            p = os.path.join(path, tn)
            dfs = [pd.read_csv(os.path.join(p, f)) for f in os.listdir(p) if f != 'labels.csv']
            if len(dfs) > 0:        
                df = pd.concat(dfs, ignore_index=True, copy=False)
                df.to_csv(
                    os.path.join(p, 'labels.csv'),
                    index=False
                )
                # count the number of samples for each class
                print('Method and test name:', method, tn, '\nTotal number of images for this sub test', len(df['img_id'].unique()))

    pool.shutdown()

    return


def merge_dataset(dir: str, num_roll: int):
    '''
        Merge all the datasets into labels.csv.
    '''

    if os.path.exists(os.path.join(dir, 'merge.log')):
        with open(os.path.join(dir, 'merge.log'), 'r') as log_file:
            names = json.load(log_file)
    else:
        names = []

    dfs = []
    for offset in os.listdir(dir):
        if offset == 'labels.csv' or offset == 'merge.log':
            continue
        for noise_cfg in os.listdir(os.path.join(dir, offset)):
            name = os.path.join(offset, noise_cfg)
            if name in names:
                print(name)
                continue
            else:
                names.append(name)

            path = os.path.join(dir, name)
            dfs.extend(
                [pd.read_csv(os.path.join(path, file)) for file in os.listdir(path) if file.endswith(f'_{num_roll}')]
            )
    
    # get all the csv files
    if os.path.exists(os.path.join(dir, 'labels.csv')):
        dfs.append(pd.read_csv(os.path.join(dir, 'labels.csv')))

    # store dataset and merge.log
    df = pd.concat(dfs, ignore_index=True, copy=True)
    df.to_csv(os.path.join(dir, 'labels.csv'), index=False)
    with open(os.path.join(dir, 'merge.log'), 'w') as log_file:
        json.dump(names, log_file)


if __name__ == '__main__':
    if True:
        agg_dataset(
            meth_params={
                # 'lpt_nn': [0.5, 6, 55, 0],
                'rac_nn': [0.5, 6, [15, 55], 18, 3, 0],
            },
            simu_params={
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
            gcata_path='catalogue/sao6.0_d0.03_12_15.csv',
            offset=0,
            num_roll=1,
            num_thd=20
        )

    if False:
        agg_dataset(
            meth_params={
                'rac_nn': [0.1, 4.5, [10, 25, 40, 55], 18, 3, 0],
            },
            simu_params={
                'h': 1024,
                'w': 1280,
                'fovx': 11.398822251559647,
                'fovy': 9.129887427521604,
                'limit_mag': 5.5,
                'sigma_pos': 0,
                'sigma_mag': 0.5,
                'num_fs': 0,
                'num_ms': 0,
                'rot': 1
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            offset=0,
            num_roll=10,
            num_thd=20
        )
    
    if False:
        agg_sample(
            400, 
            {
                # 'grid': [0.5, 6, 100],
                # 'lpt': [0.5, 6, 50, 36],
                'lpt_nn': [0.5, 6, 55, 0],
                'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 0]
            }, 
            {
                'h': 1024,
                'w': 1282,
                'fovy': 12,
                'fovx': 14.9925,
                'limit_mag': 6,
                'rot': 1
            },
            {
                'pos': [0, 0.5, 1, 1.5, 2], 
                'mag': [0, 0.1, 0.2, 0.3, 0.4], 
                # 'fs': [0, 1, 2, 3, 4],
                # 'ms': [0, 1, 2, 3, 4]
            },
            './catalogue/sao6.0_d0.03_12_15.csv',
        )
    
    if False:
        # dir = 'dataset/1024_1280_11.398822251559647_9.129887427521604_5.5/rac_nn/sao5.5_d0.03_9_10_0.1_4.5_[50, 100]_16_3'
        # dir = 'dataset/1024_1282_12_14.9925_6_1/rac_nn/sao6.0_d0.03_12_15_0.5_6_[10, 25, 40, 55]_18_3'
        # dir = 'dataset/1024_1282_12_14.9925_6_1/lpt_nn/sao6.0_d0.03_12_15_0.5_6_55'
        dir = 'dataset/1024_1282_12_14.9925_6_1/rac_nn/sao6.0_d0.03_12_15_0.5_6_[15, 35, 55]_18_3'
        merge_dataset(dir, 10)