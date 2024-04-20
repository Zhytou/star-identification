import os
import uuid
import re
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.spatial.distance import cdist

from simulate import w, h, FOV, sim_cfg, create_star_image
from preprocess import get_star_centroids


# guide star catalogue for pattern match database and nn dataset generation
gcata_path = 'catalogue/SAO5.6_15_22.csv'
# use for generation config
gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
# guide star catalogue
gcatalogue = pd.read_csv(gcata_path, usecols= ["Star ID", "RA", "DE", "Magnitude"])

# number of reference star
num_class = len(gcatalogue)

# define the path to store the database and pattern as well as dataset
database_path = f'database/{sim_cfg}'
test_path = f'test/{sim_cfg}'
dataset_path = f'dataset/{sim_cfg}'


def generate_pm_database(gen_params: dict, use_preprocess: bool = False, num_thread: int = 10):
    '''
        Generate the pattern database for the given star catalogue.
    Args:
        gen_params: the parameters for the test sample generation
                'pm1': grid algorithm Rb Rp grid_len
                'pm2': radial and cyclic algorithm Rb Rr Rc N
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
    '''
    
    def generate_database(method: str, idxs: pd.Index):
        '''
            Generate the pattern database for the given star catalogue.
        Args:
            idxs: the indexes of star catalogue used to generate database
        Return:
            database: the pattern database
        '''

        database = []
        for ra, de in zip(gcatalogue.loc[idxs, 'RA'], gcatalogue.loc[idxs, 'DE']):
            # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
            img, star_info = create_star_image(ra, de, 0)
            # generate star_table: (row, col) -> star_id
            star_table = dict(map(lambda x: (x[1], x[0]), star_info))
            # get the centroids of the stars in the image
            if use_preprocess:
                stars = np.array(get_star_centroids(img))
            else:
                stars = np.array(list(star_table.keys()))

            star_id = star_table.get((h/2, w/2), -1)
            if star_id == -1:
                print('The star is not in the center of the image!')
                continue
            # calculate the relative coordinates
            stars = stars-(h/2, w/2)
            # calculate the distance between the star and the center of the image
            distances = np.linalg.norm(stars, axis=1)
            # sort the stars by distance with accending order
            stars = stars[distances.argsort()]
            distances = np.sort(distances)
            # exclude the reference star (h/2, w/2)
            assert stars[0][0] == 0 and stars[0][1] == 0 and distances[0] == 0
            stars, distances = stars[1:], distances[1:]
            if len(stars) < 2:
                continue
            if method == 'pm1':
                # exclude stars out of region
                stars = stars[(distances >= Rb) & (distances <= Rp)] 
                # find the nearest neighbor star
                if len(stars) == 0:
                    print(star_id)
                    continue
                nearest_star = stars[0]
                # calculate rotation angle & matrix
                angle = np.arctan2(nearest_star[1], nearest_star[0])
                M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                rotated_stars = np.dot(stars, M)
                assert round(rotated_stars[0][1])==0
                # calculate the pattern
                grid = np.zeros((grid_len, grid_len), dtype=int)
                for star in rotated_stars:
                    row = int((star[0]/Rp+1)/2*grid_len)
                    col = int((star[1]/Rp+1)/2*grid_len)
                    grid[row][col] = 1
                # store the 1's position of the grid
                database.append({
                    'pattern': ' '.join(map(str, np.flatnonzero(grid))),
                    'id': star_id
                })
            else:
                # count the number of stars in each ring
                r_cnts, _ = np.histogram(distances, bins=N, range=(Rb, Rr))
                # generate radial pattern 01 sequence
                r_pattern = np.zeros(N, dtype=int)
                r_pattern[np.nonzero(r_cnts)] = 1

                # exclude stars out of region
                stars = stars[(distances >= Rb) & (distances <= Rc)] 
                # calculate the angles between the star and the center of the image
                angles = np.arctan2(stars[:, 1], stars[:, 0])
                # rotate the stars until the nearest star lies on the horizontal axis
                angles = angles - angles[0]
                # make sure angles are in the range of [-pi, pi]
                angles %= 2*np.pi
                angles[angles >= np.pi] -= 2*np.pi
                angles[angles < -np.pi] += 2*np.pi
                # count the number of stars in each sector
                s_cnts, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
                # generate cyclic pattern 01 sequence
                c_pattern = np.zeros(8, dtype=int)
                c_pattern[np.nonzero(s_cnts)] = 1
                database.append({
                    'pattern': ' '.join(map(str, np.concatenate([r_pattern, c_pattern]))),
                    'id': star_id
                })

        # store the rest of the results
        return pd.DataFrame(database)

    # use thread pool to generate the database
    pool = ThreadPoolExecutor(max_workers=num_thread)
    tasks = defaultdict(list)

    # iterate the methods
    for method in gen_params.keys():
        if method == 'pm1':
            # parse parameters: buffer radius, pattern radius and grid length
            rb, rp, grid_len = gen_params[method]
            # radius in pixels
            Rb, Rp = rb/FOV*w, rp/FOV*w

            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{grid_len}'
        elif method == 'pm2':
            # parse parameters: buffer radius, radial radius, cyclic radius, number of rings
            rb, rr, rc, N = gen_params[method]
            # radius in pixels
            Rb, Rr, Rc = rb/FOV*w, rr/FOV*w, rc/FOV*w

            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rr}_{rc}_{N}'
        else:
            print('Invalid method!')
            return
        
        # number of round used for this method
        num_round = num_thread//len(gen_params)
        len_td = len(gcatalogue)//num_round
        # add task
        for i in range(num_round):
            beg, end = i*len_td, min((i+1)*len_td, len(gcatalogue))
            if beg >= end:
                continue
            task = pool.submit(generate_database, method, gcatalogue.index[beg:end])
            tasks[method].append(task)
    
    # wait all tasks to be done and merge all the results
    for method in tasks.keys():
        # make directory to store the database
        if method == 'pm1':
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{grid_len}'
        else:
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rr}_{rc}_{N}'
        if not os.path.exists(path):
            os.makedirs(path)

        # temporary results
        dfs = []
        for task in tasks[method]:
            df = task.result()
            if len(df) > 0:
                dfs.append(df)

        # merge the results
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(f'{path}/{method}.csv', index=False)


def generate_nn_dataset(mode: str, num_vec: int, idxs: list, use_preprocess: bool, R: int, num_ring: int, num_sector: int, num_neighbor: int, pos_noise_std: float, mv_noise_std: float, num_false_star: int):
    '''
        Generate radial and cyclic features dataset for NN model using the given star catalogue.
    Args:
        mode: generation mode
            1. 'random', use uniformed distributed vector on sphere
            2. 'supplementary', use catalogue index to generate samples for specific star
        num_vec: the number of vectors to be generated
        idxs: the indexes of star catalogue used to generate dataset
        R: the radius of the region in pixels
        num_ring: the number of rings
        num_sector: the number of sectors
        num_neighbor: the minimum number of neighbor stars in the region
        pos_noise_std: the standard deviation of the positional noise
        mv_noise_std: the standard deviation of the magnitude noise
        num_false_star: the number of false stars
    Returns:
        df: the dataset
    '''
    # store the label information
    labels = []

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    if mode == 'random':
        ras = np.random.uniform(-np.pi, np.pi, num_vec)
        des = np.arcsin(np.random.uniform(-1, 1, num_vec))
    elif mode == 'supplementary':
        ras = np.clip(gcatalogue.loc[idxs, 'RA']+np.radians(np.random.normal(0, 1, len(idxs))), -np.pi, np.pi)
        des = np.clip(gcatalogue.loc[idxs, 'DE']+np.radians(np.random.normal(0, 1, len(idxs))), -np.pi/2, np.pi/2)
    else:
        print('Invalid mode!')
        return

    # generate the star image
    for ra, de in zip(ras, des):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ra, de, 0, pos_noise_std=pos_noise_std, mv_noise_std=mv_noise_std, num_false_star=num_false_star)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))
        
        # get the centroids of the stars in the image
        if use_preprocess:
            stars = np.array(get_star_centroids(img))
        else:
            stars = np.array(list(star_table.keys()))
        if len(stars) < num_neighbor+1:
            continue

        # generate a unique img id for later accuracy calculation
        img_id = uuid.uuid1()

        # distances and angles between each star in FOV
        distances = cdist(stars, stars, 'euclidean')
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
            if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                continue
            # check if false star or star not in guide catalogue
            star_id = star_table.get(tuple(star), -1)
            if star_id == -1 or star_id not in gcatalogue['Star ID'].values:
                continue
            # get catalogue index of the guide star
            cata_idx = gcatalogue[gcatalogue['Star ID'] == star_id].index.to_list()[0]
            # generate label information for training and testing
            label = {
                'ra': ra,
                'de': de,
                'star_id': star_id,
                'cata_idx': cata_idx,
                'img_id': img_id
            }

            # angles is sorted by distance with accending order
            ags = ags[np.argsort(ds)]
            ds = np.sort(ds)
            ags = ags[ds <= R]
            # remove the first element of ags & ds, which is reference star
            ds, ags = ds[1:], ags[1:]
            # skip if only the reference star in the region
            if len(ags) == 0:
                continue

            ring_counts, _ = np.histogram(ds, bins=num_ring, range=(0, R))
            for i, rc in enumerate(ring_counts):
                label[f'ring{i}'] = rc

            # uses several neighbor stars as the starting angle to obtain the cyclic features
            for i, ag in enumerate(ags[:num_neighbor]):
                rotated_ags = ags - ag
                rotated_ags %= 2*np.pi
                rotated_ags[rotated_ags > np.pi] -= 2*np.pi
                rotated_ags[rotated_ags < -np.pi] += 2*np.pi
                sector_counts, _ = np.histogram(ags, bins=num_sector, range=(-np.pi, np.pi))
                for j, sc in enumerate(sector_counts):
                    label[f'n{i}_sector{j}'] = sc
            
            if len(ags) < num_neighbor:
                for i in range(len(ags), num_neighbor):
                    for j in range(num_sector):
                        label[f'n{i}_sector{j}'] = 0

            labels.append(label)

    # return the label information
    df = pd.DataFrame(labels)
    return df


def generate_test_samples(num_vec: int, gen_params: dict, use_preprocess: bool = False, pos_noise_std: float = 0, mv_noise_std: float = 0, num_false_star: int = 0):
    '''
        Generate pattern match test case.
    Args:
        num_vec: the number of vectors to be generated
        gen_params: the parameters for the test sample generation
                'pm1': grid algorithm Rb Rp grid_len
                'pm2': radial and cyclic algorithm Rb Rr Rc num_ring num_sector
                'nn': proposed 1dcnn algorithm R num_ring num_sector num_neighbor
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
        R: the radius of the region in pixels
        pos_noise_std: the standard deviation of the positional noise
        mv_noise_std: the standard deviation of the magnitude noise
        num_false_star: the number of false stars
    Returns:
        df: the test case
    '''

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    ras = np.random.uniform(-np.pi, np.pi, num_vec)
    des = np.arcsin(np.random.uniform(-1, 1, num_vec))

    # the dict to store the results
    patterns = defaultdict(list)

    # generate the star image
    for ra, de in zip(ras, des):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ra, de, 0, pos_noise_std=pos_noise_std, mv_noise_std=mv_noise_std, num_false_star=num_false_star)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))
        # get the centroids of the stars in the image
        if use_preprocess:
            stars = np.array(get_star_centroids(img))
        else:
            stars = np.array(list(star_table.keys()))

        # too few stars for quest algorithm to identify satellite attitude
        if len(stars) < 3:
            continue
        
        R = gen_params['nn'][0]/FOV*w
        # number of candidate primary stars in the region
        in_rect  = np.logical_and(
            np.logical_and(stars[:, 0] >= R/2, stars[:, 0] <= h-R/2),
            np.logical_and(stars[:, 1] >= R/2, stars[:, 1] <= w-R/2)
        )
        # too few stars to identify satellite attitude
        if np.sum(in_rect) < 3:
            continue

        # generate a unique img id for later accuracy calculation
        img_id = uuid.uuid1()

        # distances = cdist(stars, stars, 'euclidean')
        distances = np.linalg.norm(stars[:,None,:] - stars[None,:,:], axis=-1)
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
            # check if false star or not in guide star catalogue
            star_id = star_table.get(tuple(star), -1)
            if star_id == -1 or star_id not in gcatalogue['Star ID'].values:
                continue

            # angles is sorted by distance with accending order
            ss, ags = stars[np.argsort(ds)], ags[np.argsort(ds)]
            ds = np.sort(ds)
            # remove the first element, which is reference star
            assert star[0] == ss[0][0] and star[1] == ss[0][1] and ds[0] == 0 and ags[0] == 0
            ss, ds, ags = ss[1:], ds[1:], ags[1:]
            # calculate the relative coordinates
            ss = ss-star
            if len(ss) < 2:
                continue

            methods = list(gen_params.keys())
            for method in methods:
                if method == 'nn':
                    # parse the parameters: 
                    r, num_ring, num_sector, num_neighbor = gen_params[method]
                    # radius in pixels
                    R = r/FOV*w
                    if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                        continue
                    # get catalogue index of the guide star
                    cata_idx = gcatalogue[gcatalogue['Star ID'] == star_id].index.to_list()[0]
                    pattern = {
                        'star_id': star_id,
                        'cata_idx': cata_idx,
                        'img_id': img_id
                    }
                    ring_counts, _ = np.histogram(ds, bins=num_ring, range=(0, R))
                    for i, rc in enumerate(ring_counts):
                        pattern[f'ring{i}'] = rc

                    nn_ags = ags[ds <= R]
                    # uses several neighbor stars as the starting angle to obtain the cyclic features
                    for i, ag in enumerate(nn_ags[:num_neighbor]):
                        rotated_ags = nn_ags - ag
                        rotated_ags %= 2*np.pi
                        rotated_ags[rotated_ags > np.pi] -= 2*np.pi
                        rotated_ags[rotated_ags < -np.pi] += 2*np.pi
                        sector_counts, _ = np.histogram(ags, bins=num_sector, range=(-np.pi, np.pi))
                        for j, sc in enumerate(sector_counts):
                            pattern[f'n{i}_sector{j}'] = sc
                    
                    if len(ags) < num_neighbor:
                        for i in range(len(ags), num_neighbor):
                            for j in range(num_sector):
                                pattern[f'n{i}_sector{j}'] = 0

                    patterns[method].append(pattern)
                elif method == 'pm1':
                    # parse the parameters: buffer radius, pattern radius and grid length
                    rb, rp, grid_len = gen_params[method]
                    # radius in pixels
                    Rb, Rp = rb/FOV*w, rp/FOV*w
                    # exclude stars outside the region
                    pm1_ss, pm1_ags = ss[(ds >= Rb) & (ds <= Rp)], ags[(ds >= Rb) & (ds <= Rp)]
                    if len(pm1_ss) < 2:
                        continue
                    ag = pm1_ags[0]
                    M = np.array([[np.cos(ag), -np.sin(ag)], [np.sin(ag), np.cos(ag)]])
                    pm1_ss = np.dot(pm1_ss, M)
                    assert round(pm1_ss[0][1])==0
                    # calculate the pattern
                    grid = np.zeros((grid_len, grid_len), dtype=int)
                    for s in pm1_ss:
                        row = int((s[0]/Rp+1)/2*grid_len)
                        col = int((s[1]/Rp+1)/2*grid_len)
                        grid[row][col] = 1
                    # store the 1's position of grid
                    patterns[method].append({
                        'img_id': img_id,
                        'pattern': ' '.join(map(str, np.flatnonzero(grid))), 
                        'id': star_id
                    })
                elif method == 'pm2':
                    # parse the parameters: buffer radius, radial radius, cyclic radius, number of rings
                    rb, rr, rc, N = gen_params[method]
                    # radius in pixels
                    Rb, Rr, Rc = rb/FOV*w, rr/FOV*w, rc/FOV*w

                    # count the number of stars in each ring
                    r_cnts, _ = np.histogram(ds, bins=N, range=(Rb, Rr))
                    # generate radial pattern 01 sequence
                    r_pattern = np.zeros(N, dtype=int)
                    r_pattern[np.nonzero(r_cnts)] = 1

                    # exclude stars outside the region
                    pm2_ags = ags[(ds >= Rb) & (ds <= Rc)]
                    # rotate the stars until the nearest star lies on the horizontal axis
                    if len(pm2_ags) < 2:
                        continue
                    ags = ags-ags[0]
                    # make sure angles are in the range of [-pi, pi]
                    pm2_ags %= 2*np.pi
                    pm2_ags[pm2_ags > np.pi] -= 2*np.pi
                    pm2_ags[pm2_ags < -np.pi] += 2*np.pi
                    s_cnts, _ = np.histogram(pm2_ags, bins=8, range=(-np.pi, np.pi))
                    # generate cyclic pattern 01 sequence
                    c_pattern = np.zeros(8, dtype=int)
                    c_pattern[np.nonzero(s_cnts)] = 1

                    patterns[method].append({
                        'img_id': img_id,
                        'pattern': ' '.join(map(str, np.concatenate([r_pattern, c_pattern]))),
                        'id': star_id
                    })                    
                else:
                    print('Invalid method')
    
    # convert the results into dataframe
    for key in patterns:
        patterns[key] = pd.DataFrame(patterns[key])
    return patterns


def aggregate_nn_dataset(types: dict, use_preprocess: bool, region_r: float, num_ring: int, num_sector: int, num_neighbor: int, pos_noise_stds: list = [], mv_noise_stds: list = [], num_false_stars: list = [], num_thread: int = 20):
    '''
        Aggregate the dataset. Firstly, the number of samples for each class is counted. Then, roughly generate classes with too few samples using generate_nn_dataset function's 'random' mode. Lastly, the rest are finely generated to ensure that the number of samples in each class in the entire dataset reaches the standard.
    Args:
        types: key->the types of the dataset, values->the minumin number of samples for each class
        region_r: the radius of the region in degrees
        num_ring: the number of rings
        num_sector: the number of sectors
        num_neighbor: the number of neighbor stars
        pos_noise_stds: list of positional noise
        mv_noise_stds: list of magnitude noise
        num_false_stars: list of false star number
        num_thread: the number of threads to generate the dataset
    '''

    def wait_tasks(tasks: dict, root_path: str, file_name: str, col_name: str = None, num_sample_per_class: dict = None):
        '''
            Wait for all tasks to be done. Then, merge the result of async tasks and store it in labels.csv. 
        Args:
            tasks: the tasks to be done
            root_path: the root path to store the dataset
            col_name: the column name of the label
            num_sample_per_class: the minumum number for each class
                                if num_sample_per_class is -1, the function will not truncate the dataset
        Returns:
            tasks: the tasks done(reserved keys and empty list for values)
        '''
        for key in tasks.keys():
            if len(tasks[key]) == 0:
                continue

            # make directory for every type of dataset
            path = os.path.join(root_path, key)
            if not os.path.exists(path):
                os.makedirs(path)

            # store the samples        
            dfs = []
            for task in tasks[key]:
                df = task.result()
                if len(df) > 0:
                    df.to_csv(f"{path}/{uuid.uuid1()}", index=False)
                    dfs.append(df)
            
            # remain the old samples
            if os.path.exists(f'{path}/{file_name}'):
                dfs.append(pd.read_csv(f'{path}/{file_name}'))

            # aggregate the dataset
            df = pd.concat(dfs, ignore_index=True)

            # print dataset distribution per class
            if col_name:
                df_info = df[col_name].value_counts()
                print(len(df_info), df_info.tail(3))
            
            # truncate and store the dataset
            if num_sample_per_class !=  None:
                df = df.groupby(col_name).apply(lambda x: x.sample(n=min(len(x), num_sample_per_class[key])))
                df_info = df[col_name].value_counts()
                print(key, col_name, len(df_info), len(df_info[df_info < num_sample_per_class[key]]), len(df_info[df_info < num_sample_per_class[key]//2]))
            
            df.to_csv(f'{path}/{file_name}', index=False)

            # clear tasks[key] after aggregation
            tasks[key] = []

        return tasks

    def parse_params(s: str):
        '''
            Parse special test parameters.
        Args:
            s: the string to be parsed
        Returns:
            pns: the positional noise standard deviation
            mns: the magnitude noise standard deviation
            nfs: the number of false stars
        '''
        pns, mns, nfs = 0, 0, 0
        match = re.match('.+\/(pos|mv|fs)([0-9]+\.?[0-9]*)', s)
        if match:
            test_type, number = match.groups()
            if test_type == 'pos':
                pns = float(number)
            elif test_type == 'mv':
                mns = float(number)
            else:  # test_type == 'fs'
                nfs = int(number)
        return pns, mns, nfs

    # generate config
    gen_cfg = f'{gcata_name}_{int(use_preprocess)}_{region_r}_{num_ring}_{num_sector}_{num_neighbor}'
    R = int(region_r/FOV*w)
    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thread)
    
    # generate config for special test
    if 'test' in types.keys():
        for pns in pos_noise_stds:
            types[f'test/pos{pns}'] = types['test']
        for mns in mv_noise_stds:
            types[f'test/mv{mns}'] = types['test']
        for nfs in num_false_stars:
            types[f'test/fs{nfs}'] = types['test']
        types['test/default'] = types.pop('test')

        print(types)

    # rough generation
    tasks = defaultdict(list)
    for key in types.keys():
        # reuse old samples
        files = []
        if os.path.exists(os.path.join(dataset_path, gen_cfg, key)):
            files = os.listdir(os.path.join(dataset_path, gen_cfg, key))
        if len(files) > 0:
            df = pd.concat([pd.read_csv(os.path.join(dataset_path, gen_cfg, key, file)) for file in files if file != 'labels.csv'])
            df.to_csv(os.path.join(dataset_path, gen_cfg, key, 'labels.csv'), index=False)
            # count the number of samples for each class
            df = df['cata_idx'].value_counts()
            pct = len(df[df > types[key]])/len(gcatalogue)
            if len(df[df < types[key]]) == 0:
                avg_num = 0
            else:
                avg_num = np.sum(types[key]-df[df < types[key]])/len(df[df < types[key]])
            print('pct: ', pct, ' len(df[df < types[key]]): ', len(df[df < types[key]]), ' avg_num:', avg_num)
            if pct > 0.5 or avg_num < types[key]/3:
                print(f'{key} skip rough generation!')
                continue
        
        # parse parameters
        pos_noise_std, mv_noise_std, num_false_star = parse_params(key)
        print(key, pos_noise_std, mv_noise_std, num_false_star)

        # roughly generate the samples for each class
        num_vec = types[key]*300
        if num_vec > 900:
            num_round = num_vec//900+1
            num_vec = 900
            if num_round > num_thread//4:
                num_round = num_thread//4
                num_vec = types[key]*300//num_round
        else:
            num_round = 1
        for _ in range(num_round):
            task = pool.submit(generate_nn_dataset, 'random', num_vec, [], use_preprocess, R, num_ring, num_sector, num_neighbor, pos_noise_std, mv_noise_std, num_false_star)
            tasks[key].append(task)

    # wait for all tasks to be done and merge the results
    tasks = wait_tasks(tasks, os.path.join(dataset_path, gen_cfg), 'labels.csv', 'cata_idx')

    # fine generation
    for key in types.keys():
        df = pd.read_csv(os.path.join(dataset_path, gen_cfg, key, 'labels.csv'))
        # count the number of samples for each class
        df = df['cata_idx'].value_counts()
        # add those even unexisting class(catalogue idx)
        df = df.reindex(gcatalogue.index, fill_value=0)
        # count class whose samples less than limit
        df = df[df < types[key]]
        # repeat each the index of df (which is catalogue idx needed)
        idxs = df.index.repeat(types[key]-df).to_list()
        print(len(df), len(idxs), idxs[:3], idxs[-3:])
        # parse parameters
        pos_noise_std, mv_noise_std, num_false_star = parse_params(key)

        # for class whose sample less than num_sample_per_class, generate the dataset til the number of samples reach the standard
        if len(idxs) > 1000:
            len_td = 1000
            num_round = len(idxs)//len_td
            if num_round > num_thread//2:
                num_round = num_thread//2
                len_td = len(idxs)//num_round
        else:
            len_td = len(idxs)
            num_round = 1
        for i in range(num_round+1):
            beg, end = i*len_td, min((i+1)*len_td, len(idxs))
            if beg >= end:
                continue
            task = pool.submit(generate_nn_dataset, 'supplementary', 0, idxs[beg: end], use_preprocess, R, num_ring, num_sector, num_neighbor, pos_noise_std, mv_noise_std, num_false_star)
            tasks[key].append(task)
    
    wait_tasks(tasks, os.path.join(dataset_path, gen_cfg), 'labels.csv', 'cata_idx', types)


def aggregate_test_samples(num_vec: int, gen_params: dict, use_preprocess: bool = False, pos_noise_stds: list = [], mv_noise_stds: list = [], num_false_stars: list = [], num_thread: int = 20):
    '''
    Aggregate the test samples. 
    Args:
        num_vec: number of vectors used to generate test samples
        region_r: the radius of the region in degrees
        gen_params: the parameters for the test sample generation
                'pm1': grid algorithm Rb Rp grid_len
                'pm2': radial and cyclic algorithm Rb Rr Rc num_ring num_sector
                'nn': proposed 1dcnn algorithm R num_ring num_sector num_neighbor
        pos_noise_stds: list of positional noise
        pos_noise_stds: list of positional noise
        mv_noise_stds: list of magnitude noise
        num_false_stars: list of false star number
        num_thread: the number of threads to generate the test samples
    '''

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thread)
    # tasks for later aggregation
    tasks = defaultdict(list)

    if num_vec < 500:
        num_thread = 1
    elif num_vec//num_thread > 500:
        num_vec = num_vec//num_thread
    else:
        num_thread = num_vec//500
        num_vec = 500

    for _ in range(num_thread):
        task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess)
        tasks['default'].append(task)

        for pns in pos_noise_stds:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, pos_noise_std=pns)
            tasks[f'pos{pns}'].append(task)

        for mns in mv_noise_stds:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, mv_noise_std=mns)
            tasks[f'mv{mns}'].append(task)

        for nfs in num_false_stars:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, num_false_star=nfs)
            tasks[f'fs{nfs}'].append(task)

    # get the async task result and store the returned dataframe
    for key in tasks.keys():
        for task in tasks[key]:
            df_dict = task.result()
            for method in df_dict.keys():
                gen_cfg = f'{gcata_name}_{int(use_preprocess)}_'+'_'.join(map(str, gen_params[method]))
                path = os.path.join(test_path, method, gen_cfg, key)
                if not os.path.exists(path):
                    os.makedirs(path)
                df = df_dict[method]
                df.to_csv(os.path.join(path, str(uuid.uuid1())), index=False)

    # aggregate all the test patterns
    for method in gen_params:
        gen_cfg = f'{gcata_name}_{int(use_preprocess)}_'+'_'.join(map(str, gen_params[method]))
        path = os.path.join(test_path, method, gen_cfg)
        # sub test dir names
        test_names = os.listdir(path)
        for tn in test_names:
            p = os.path.join(path, tn)
            dfs = [pd.read_csv(os.path.join(p, f)) for f in os.listdir(p) if f != 'labels.csv']
            print(p, len(dfs))
            if len(dfs) > 0:        
                df = pd.concat(dfs, ignore_index=True)
                df.to_csv(os.path.join(p, 'labels.csv'))


if __name__ == '__main__':
    # generate_pm_database({'pm1': [0, 6, 60], 'pm2': [0, 6, 10, 100]})
    aggregate_nn_dataset({'train': 10, 'validate': 1, 'test': 2}, use_preprocess=False, region_r=6, num_ring=100, num_sector=18, num_neighbor=3)
    # aggregate_test_samples(400, {'nn': [6, 100, 18, 3], 'pm1': [0, 6, 60]})