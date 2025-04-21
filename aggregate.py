import os
import uuid
import pandas as pd
import numpy as np
from math import radians, tan
from concurrent.futures import ThreadPoolExecutor

from generate import gen_sample, gen_dataset
from collections import defaultdict


def agg_dataset(meth_params: dict, simu_params: dict, gcata_path: str, offset: float=1, num_roll: int=360, num_thd: int=20):
    '''
        Aggregate the training datasets.
    Args:
        meth_params: the parameters of methods, possible methods include:
            'rac_1dcnn': Rb Rp arr_Nr Ns Nn
            'lpt_nn': Rb Rp Nd
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

    # simulate config
    sim_cfg = f"{simu_params['h']}_{simu_params['w']}_{simu_params['fovx']}_{simu_params['fovy']}_{simu_params['limit_mag']}"

    # guide star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=["Star ID", "Ra", "De", "Magnitude"])

    # noise config
    noise_cfg = f"{simu_params['sigma_pos']}_{simu_params['sigma_mag']}_{simu_params['num_fs']}_{simu_params['num_ms']}"

    # dataset path for each method
    ds_paths = {}
    for method in meth_params:
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        ds_paths[method] = os.path.join('dataset', sim_cfg, method, gen_cfg, str(offset), noise_cfg)

    pool = ThreadPoolExecutor(max_workers=num_thd)
    tasks = []

    star_id_ra_des = gcata[['Star ID', 'Ra', 'De']].to_numpy()
    for star_id, star_ra, star_de in star_id_ra_des:
        # generate the dataset
        tasks.append(pool.submit(
            gen_dataset,
            meth_params,
            simu_params,
            ds_paths,
            int(star_id),
            star_ra,
            star_de,
            offset=offset,
            num_roll=num_roll,
            gcata=gcata
        ))

    for task in tasks:
        task.result()

    # aggregate the dataset
    for method in meth_params:
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        dir = os.path.join('dataset', sim_cfg, method, gen_cfg)
        dfs = []

        for offset_dir in os.listdir(dir):
            if offset_dir == 'labels.csv':
                continue
            for noise_dir in os.listdir(os.path.join(dir, offset_dir)):
                for id_dir in os.listdir(os.path.join(dir, offset_dir, noise_dir)):
                    for file in os.listdir(os.path.join(dir, offset_dir, noise_dir, id_dir)):
                        if not file.endswith('_10.csv'):
                            continue
                        file_path = os.path.join(dir, offset_dir, noise_dir, id_dir, file)
                        df = pd.read_csv(file_path)
                        dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(dir, 'labels.csv'), index=False)

    return


def agg_sample(num_img: int, meth_params: dict, simu_params: dict, test_params: dict, gcata_path: str, num_thd: int = 20):
    '''
        Aggregate the test samples. 
    Args:
        num_img: number of test images expected to be generated
        meth_params: the parameters of methods, possible methods include:
            'rac_1dcnn': Rb Rp arr_Nr Ns Nn
            'lpt_nn': Rb Rp Nd
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

    # simulation config
    sim_cfg = f'{simu_params["h"]}_{simu_params["w"]}_{simu_params["fovx"]}_{simu_params["fovy"]}_{simu_params["limit_mag"]}'

    # read the guide star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])

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
                df = pd.concat(dfs, ignore_index=True)
                df.to_csv(os.path.join(p, 'labels.csv'), index=False)
                # count the number of samples for each class
                print('Method and test name:', method, tn, '\nTotal number of images for this sub test', len(df['img_id'].unique()))

    pool.shutdown()

    return


if __name__ == '__main__':
    if True:
        agg_dataset(
            meth_params={
                'rac_1dcnn': [0.1, 6, [25, 50], 16, 3],
                'lpt_nn': [0.1, 6, 25],
            },
            simu_params={
                'h': 512,
                'w': 512,
                'fovx': 12,
                'fovy': 12,
                'limit_mag': 6,
                'sigma_pos': 2,
                'sigma_mag': 0,
                'num_fs': 0,
                'num_ms': 0
            },
            gcata_path='catalogue/sao6.0_d0.03_12_15.csv',
            offset=1,
            num_roll=10,
            num_thd=20
        )

    if False:
        agg_dataset(
            meth_params={
                'rac_1dcnn': [0.1, 4.5, [50, 100], 16, 3],
            },
            simu_params={
                'h': 1024,
                'w': 1280,
                'fovx': 11.398822251559647,
                'fovy': 9.129887427521604,
                'limit_mag': 5.5,
                'sigma_pos': 2,
                'sigma_mag': 0,
                'num_fs': 0,
                'num_ms': 0
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            offset=3,
            num_roll=10,
            num_thd=20
        )
    
    if True:
        agg_sample(
            400, 
            {
                # 'grid': [0.3, 6, 50],
                # 'lpt': [0.3, 6, 25, 36],
                'lpt_nn': [0.1, 6, 25],
                # 'rac_1dcnn': [0.1, 6, [25, 50], 16, 3]
            }, 
            {
                'h': 512,
                'w': 512,
                'fovx': 12,
                'fovy': 12,
                'limit_mag': 6
            },
            {
                'pos': [0, 0.5, 1, 1.5, 2], 
                'mag': [0, 0.1, 0.2, 0.3, 0.4], 
                'fs': [0, 1, 2, 3, 4]
            },
            './catalogue/sao6.0_d0.03_12_15.csv',
        )
    