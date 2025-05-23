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

    # setup gcata and several configs
    sim_cfg, noise_cfg, gcata_name, gcata = setup(simu_params, gcata_path)

    print(
        'Dataset Generation',
        '\n------------------',
        '\nMethod parameters:', meth_params,
        '\nSimulation config:', sim_cfg,
        '\nSimulation config:', noise_cfg,
        '\nGuide star catalogue:', gcata_name,
        '\nOffset:', offset,
        '\nNumber of rolls:', num_roll
    )

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

    # setup gcata and several configs
    sim_cfg, noise_cfg, gcata_name, gcata = setup(simu_params, gcata_path)

    print(
        'Test Image Generation',
        '\n----------------------',
        '\nMethod Parameters:', meth_params,
        '\nSimulation config:', sim_cfg,
        '\nNoise config:', noise_cfg,
        '\nGuide star catalogue:', gcata_name,
        '\nNumber of test images expected to be generated:', num_img,
    )

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thd)
    # tasks for later aggregation
    tasks = defaultdict(list)

    # parameter mapping
    param_mappings = {
        'pos': 'sigma_pos',
        'mag': 'sigma_mag',
        'fs': 'num_fs',
        'ms': 'num_ms',
    }

    # add tasks to the thread pool
    for param_type, kwarg_name in param_mappings.items():
        for value in test_params.get(param_type, []):
            # default noise
            base_kwargs = {
                'sigma_pos': simu_params['sigma_pos'],
                'sigma_mag': simu_params['sigma_mag']
            }

            # add test noise
            base_kwargs[kwarg_name] = value
            
            # sub test name
            test_name = f"{param_type}{value}"

            # raw data directory
            raw_dir = os.path.join('test', sim_cfg, 'raw', test_name, noise_cfg)

            # independent method parameters for each sub test, because some of sub test may already meet the requirements
            nmeth_params = {}

            # read labels.csv and get the unique ids of generated image for each method
            img_ids = {}
            for method in meth_params:
                gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
                file = os.path.join('test', sim_cfg, method, gen_cfg, test_name, noise_cfg, 'labels.csv')
                df = pd.read_csv(file) if os.path.exists(file) else pd.DataFrame()

                # add to nmeth_params only needed
                if df.empty or len(df['img_id'].unique()) < num_img:
                    nmeth_params[method] = meth_params[method]
                    img_ids[method] = [] if df.empty else df['img_id'].unique()

            # add task
            tasks[test_name].append(
                pool.submit(
                    gen_sample, 
                    num_img, 
                    nmeth_params, 
                    simu_params,
                    raw_dir,
                    img_ids,
                    gcata, 
                    **base_kwargs
                )
            )
    
    # sub test name
    for test_name in tasks:
        for task in tasks[test_name]:
            # get the async task result and store the returned dataframe
            df_dict = task.result()
            for method in df_dict:
                # make the sub test directory
                gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
                st_path = os.path.join('test', sim_cfg, method, gen_cfg, test_name, noise_cfg)
                os.makedirs(st_path, exist_ok=True)

                # store the sub test samples
                df = df_dict[method]
                df.to_csv(os.path.join(st_path, str(uuid.uuid1())), index=False)

    # aggregate all the test patterns
    for method in meth_params:
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        path = os.path.join('test', sim_cfg, method, gen_cfg)
        if not os.path.exists(path):
            continue
        
        test_names = list(tasks.keys())
        for test_name in test_names:
            # sub test dir names
            p = os.path.join(path, test_name, noise_cfg)
            dfs = [pd.read_csv(os.path.join(p, f)) for f in os.listdir(p) if f != 'labels.csv']
            if len(dfs) == 0:
                continue        

            # merge all the sub test samples
            df = pd.concat(dfs, ignore_index=True, copy=False)
            df.to_csv(
                os.path.join(p, 'labels.csv'),
                index=False
            )

            # count the number of samples for each class
            print(
                'Method and test name:', method, test_name, 
                '\nTotal number of images for this sub test', len(df['img_id'].unique())
            )

    pool.shutdown()

    return


def merge_dataset(dir: str, num_rolls: list[int]):
    '''
        Merge all the datasets into labels.csv.
    '''

    if os.path.exists(os.path.join(dir, 'merge.log')):
        with open(os.path.join(dir, 'merge.log'), 'r') as log_file:
            names = json.load(log_file)
    else:
        names = {}

    dfs = []
    for offset in os.listdir(dir):
        # offset dir
        subdir = os.path.join(dir, offset)
        if not os.path.isdir(subdir):
            continue

        for noise_cfg in os.listdir(subdir):
            name = os.path.join(offset, noise_cfg)
            path = os.path.join(dir, name)

            for num_roll in num_rolls:
                # json only use str as key
                num_roll = str(num_roll)

                if name in names.setdefault(num_roll, []):
                    print('Merged', name)
                    continue
                else:
                    print('Merging', name)
                    names[num_roll].append(name)

                dfs.extend([pd.read_csv(os.path.join(path, file)) for file in os.listdir(path) if file.endswith(num_roll)])

    # get all the csv files
    if os.path.exists(os.path.join(dir, 'labels.csv')):
        dfs.append(pd.read_csv(os.path.join(dir, 'labels.csv')))

    # store dataset and merge.log
    df = pd.concat(dfs, ignore_index=True, copy=False)
    df.to_csv(os.path.join(dir, 'labels.csv'), index=False)
    with open(os.path.join(dir, 'merge.log'), 'w') as log_file:
        json.dump(names, log_file)


if __name__ == '__main__':
    if False:
        agg_dataset(
            meth_params={
                'lpt_nn': [0.5, 6, 55, 0],
                # 'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 0],
            },
            simu_params={
                'h': 1024,
                'w': 1282,
                'fovy': 12,
                'fovx': 14.9925,
                'limit_mag': 6,
                'sigma_pos': 0,
                'sigma_mag': 0.5,
                'num_fs': 0,
                'num_ms': 0,
                'rot': 1
            },
            gcata_path='catalogue/sao6.0_d0.03_12_15.csv',
            offset=4,
            num_roll=10,
            num_thd=20
        )

    if False:
        # model for picdata
        agg_dataset(
            meth_params={
                'rac_nn': [0.1, 4.5, [25, 55, 85], 18, 3, 0],
            },
            simu_params={
                'h': 1024,
                'w': 1280,
                'fovx': 11.398822251559647,
                'fovy': 9.129887427521604,
                'limit_mag': 5.5,
                'sigma_pos': 3,
                'sigma_mag': 0,
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
        # model for xie
        agg_dataset(
            meth_params={
                'rac_nn': [0.1, 5.7, [25, 55, 85], 18, 3, 0],
            },
            simu_params={
                'h': 1024,
                'w': 1280,
                'fovx': 14.37611786938476,
                'fovy': 11.522621164995503,
                'limit_mag': 5.5,
                'sigma_pos': 0,
                'sigma_mag': 0,
                'num_fs': 5,
                'num_ms': 0,
                'rot': 1
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            offset=0,
            num_roll=10,
            num_thd=20
        )
    
    if False:
        agg_dataset(
            meth_params={
                'rac_nn': [0.5, 7.7, [35, 75, 115], 18, 3, 0],
            },
            simu_params={
                'h': 1040,
                'w': 1288,
                'fovy': 15.36777053565561,
                'fovx': 18.97205141393946,
                'limit_mag': 6,
                'sigma_pos': 0,
                'sigma_mag': 0.5,
                'num_fs': 0,
                'num_ms': 0,
                'rot': 1
            },
            gcata_path='catalogue/sao5.5_d0.03_9_10.csv',
            offset=1,
            num_roll=10,
            num_thd=20
        )

    if False:
        agg_sample(
            1000, 
            {
                # 'grid': [0.5, 6, 100],
                'lpt': [0.5, 6, 50, 50],
                # 'lpt_nn': [0.5, 6, 55, 0],
                # 'rac_nn': [0.5, 6, [15, 35, 55], 18, 3, 0],
                # 'rac_nn': [0.5, 6, [15, 35, 55], 18, 3, 1],
                # 'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 0],
                # 'rac_nn': [0.5, 6, [25, 55, 85], 18, 3, 1],
            }, 
            {
                'h': 1024,
                'w': 1282,
                'fovy': 12,
                'fovx': 14.9925,
                'limit_mag': 6,
                'sigma_pos': 0,
                'sigma_mag': 0.1,
                'num_fs': 0,
                'num_ms': 0,
                'rot': 1
            },
            {
                'pos': [0, 0.5, 1, 1.5, 2], 
                # 'mag': [0, 0.1, 0.2, 0.3, 0.4], 
                # 'fs': [0, 1, 2, 3, 4],
                # 'ms': [0, 1, 2, 3, 4]
            },
            './catalogue/sao6.0_d0.03_12_15.csv',
        )
    
    if True:
        # dir = 'dataset/1024_1280_9.129887427521604_11.398822251559647_5.5_1/rac_nn/sao5.5_d0.03_9_10_0.1_4.5_[25, 55, 85]_18_3_0'
        # dir = 'dataset/1024_1280_9.129887427521604_11.398822251559647_5.5_1/rac_nn/sao5.5_d0.03_9_10_0.1_4.5_[15, 35, 55]_18_3_0'
        # dir = 'dataset/1024_1280_11.522621164995503_14.37611786938476_5.5_1/rac_nn/sao5.5_d0.03_9_10_0.1_5.7_[25, 55, 85]_18_3_0'
        # dir = 'dataset/1024_1282_12_14.9925_6_1/lpt_nn/sao6.0_d0.03_12_15_0.5_6_55_0'
        # dir = 'dataset/1024_1282_12_14.9925_6_1/rac_nn/sao6.0_d0.03_12_15_0.5_6_[15, 35, 55]_18_3_0'
        # dir = 'dataset/1024_1282_12_14.9925_6_1/rac_nn/sao6.0_d0.03_12_15_0.5_6_[25, 55, 85]_18_3_0'
        dir = 'dataset/1040_1288_15.36777053565561_18.97205141393946_6_1/rac_nn/sao5.5_d0.03_9_10_0.5_7.7_[35, 75, 115]_18_3_0'
        merge_dataset(dir, [10, 20])