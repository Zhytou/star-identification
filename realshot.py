import os
import uuid
import pandas as pd
import numpy as np
from math import radians, tan
from concurrent.futures import ThreadPoolExecutor

from simulate import create_star_image
from generate import gen_pattern
from collections import defaultdict


def gen_dataset(meth_params: dict, simu_params: dict, ds_paths: dict, star_id: int, star_ra: float, star_de: float, offset :float, num_roll: int, gcata: pd.DataFrame):
    '''
        Generate dataset for NN model using the given star catalogue.
    '''
    name = uuid.uuid1()

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    ras = np.clip(np.full(num_roll, star_ra) + np.radians(np.random.normal(0, offset, num_roll)), 0, 2*np.pi)
    des = np.clip(np.full(num_roll, star_de) + np.radians(np.random.normal(0, offset, num_roll)), -np.pi/2, np.pi/2)
    rolls = np.arange(0, 2*np.pi, 2*np.pi/num_roll)
    pats_dict = defaultdict(list)

    # focus in pixels used to calculate the buffer and pattern radius
    f1 = simu_params['w'] / (2 * tan(radians(simu_params['fovx'] / 2)))
    f2 = simu_params['h'] / (2 * tan(radians(simu_params['fovy'] / 2)))
    assert np.isclose(f1, f2), f"The focal length in x direction is {f1}, while in y direction is {f2}."
    f = (f1+f2)/2

    # generate the star image
    for ra, de, roll in zip(ras, des, rolls):
        # stars is a np array [[id, row, col, mag]]
        _,  stars = create_star_image(ra, de, roll, 
            h=simu_params['h'], 
            w=simu_params['w'],
            fovx=simu_params['fovx'],
            fovy=simu_params['fovy'],
            # pixel=simu_params['pixel'],
            limit_mag=simu_params['limit_mag'],
            sigma_pos=simu_params['sigma_pos'],
            sigma_mag=simu_params['sigma_mag'],
            num_fs=simu_params['num_fs'],
            num_ms=simu_params['num_ms'], 
            coords_only=False
        )

        # get star ids and coordinates
        ids = stars[:, 0]
        coords = stars[:, 1:3]
        
        # set all the star ids to -1 except the given star id in order to make sure only generate the pattern for the given star id
        ids[ids != star_id] = -1
        
        if np.all(ids == -1):
            continue

        # generate a unique img id for later accuracy calculation
        img_id = str(uuid.uuid1())

        # generate the pattern(dict, method->pattern) for the given star id
        pat_dict = gen_pattern(
            meth_params, 
            coords, 
            ids, 
            img_id, 
            f=f,
            h=simu_params['h'], 
            w=simu_params['w'], 
            gcata=gcata
        )

        # store the patterns
        for method, pat in pat_dict.items():
            assert len(pat) == 1, f"Error: {len(pat)} patterns generated for method {method}."
            pat = pat[0]
            pat['roll'] = roll
            pats_dict[method].append(pat)

    for method in pats_dict:
        pats_dict[method] = pd.DataFrame(pats_dict[method])
        # make directory
        ds_path = os.path.join(ds_paths[method], f'{star_id}')
        os.makedirs(ds_path, exist_ok=True)
        # save the dataset
        pats_dict[method].to_csv(os.path.join(ds_path, f'{name}_{num_roll}.csv'), index=False)


def agg_dataset(meth_params: dict, simu_params: dict, gcata_path: str, offset: float=1, num_roll: int=360, num_thd: int=20):
    '''
        Aggregate the dataset into a single dataframe.
    '''

    if meth_params == {}:
        return 

    print('Dataset Generation')
    print('------------------')
    print('Method parameters:', meth_params)
    print('Simulation parameters:', simu_params)
    print('Guide star catalogue:', gcata_path)
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
    

if __name__ == '__main__':
    if False:
        dir = 'dataset/512_512_12_12_6/rac_1dcnn/sao6.0_d0.03_12_15_0.1_6_[25, 50]_16_3'
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

    if True:
        agg_dataset(
            meth_params={
                # 'rac_1dcnn': [0.1, 6, [25, 50], 16, 3],
                'lpt_nn': [0.1, 6, 25],
            },
            simu_params={
                'h': 512,
                'w': 512,
                'fovx': 12,
                'fovy': 12,
                'limit_mag': 6,
                'sigma_pos': 0,
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