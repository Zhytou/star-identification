import os
import uuid
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.spatial.distance import cdist

from simulate import w, h, FOV, catalogue_path, create_star_image
from preprocess import get_star_centroids


# region radius in pixels
region_r = int(w/FOV*6)

# star catalogue
catalogue = pd.read_csv(catalogue_path, usecols= ["Star ID", "RA", "DE", "Magnitude"])

# number of reference star
num_class = len(catalogue)

# define the path to store the database and pattern as well as dataset
sim_cfg = f"{os.path.basename(catalogue_path).rsplit('.', 1)[0]}_{w}x{h}_{FOV}x{FOV}"
database_path = f'database/{sim_cfg}'
pattern_path = f'pattern/{sim_cfg}'
dataset_path = f'data/{sim_cfg}'


def generate_pm_database(method: int, use_preprocess: bool = False, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30):
    '''
        Generate the pattern database for the given star catalogue.
    Args:
        method: the method to generate the pattern database
                1: grid algorithm
                2: radial and cyclic algorithm
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
        grid_len: the length of the grid
        num_ring: the number of rings
        num_sector: the number of sectors
    '''
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    
    database = []
    for ra, de in zip(catalogue['RA'], catalogue['DE']):
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
        # exclude stars out of region
        stars = stars[distances < region_r]
        # exclude the reference star (h/2, w/2)
        assert stars[0][0] == 0 and stars[0][1] == 0 and distances[0] == 0
        stars, distances = stars[1:], distances[1:]
        if len(stars) < 2:
            continue
        if method == 1:        
            # find the nearest neighbor star
            nearest_star = stars[0]
            # calculate rotation angle & matrix
            angle = np.arctan2(nearest_star[1], nearest_star[0])
            M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_stars = np.dot(stars, M)
            assert round(rotated_stars[0][1])==0
            # calculate the pattern
            grid_pattern = np.zeros((grid_len, grid_len), dtype=int)
            for star in rotated_stars:
                row = int(star[0]/region_r*grid_len)
                col = int(star[1]/region_r*grid_len)
                grid_pattern[row][col] = 1
            database.append({
                'pattern': ''.join(map(str, grid_pattern.flatten())),
                'id': star_id
            })
        elif method == 2:
            # calculate the angles between the star and the center of the image
            angles = np.arctan2(stars[:, 1], stars[:, 0])
            # rotate the stars until the nearest star lies on the horizontal axis
            angles = angles - angles[0]
            # make sure angles are in the range of [-pi, pi]
            angles %= 2*np.pi
            angles[angles > np.pi] -= 2*np.pi
            angles[angles < -np.pi] += 2*np.pi
            # get the radial and cyclic features
            ring_counts, _ = np.histogram(distances, bins=num_ring, range=(0, region_r))
            sector_counts, _ = np.histogram(angles, bins=num_sector, range=(-np.pi, np.pi))
            database.append({
                'pattern': ''.join(map(str, np.concatenate([ring_counts, sector_counts]))),
                'id': star_id
            })
        else:
            print('Invalid method!')
            return

    df = pd.DataFrame(database)
    if method == 1:
        df.to_csv(f"{database_path}/db{method}_{grid_len}x{grid_len}.csv", index=False)
    else:
        df.to_csv(f"{database_path}/db{method}_{num_ring}_{num_sector}.csv", index=False)


def generate_pm_samples(method: int, num_sample: int, use_preprocess: bool = False, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30, pos_noise_std: float = 0, mv_noise_std: float = 0, num_false_star: int = 0):
    '''
        Generate pattern match test case.
    Args:
        method: the method to generate the pattern
                1: grid algorithm
                2: radial and cyclic algorithm
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
        grid_len: the length of the grid
        num_ring: the number of rings
        num_sector: the number of sectors
        pos_noise_std: the standard deviation of the positional noise
        mv_noise_std: the standard deviation of the magnitude noise
        num_false_star: the number of false stars
    '''
    if method == 1:
        pattern_test_path = os.path.join(pattern_path, f'db{method}_{grid_len}x{grid_len}_pos{pos_noise_std}_mv{mv_noise_std}_fs{num_false_star}_test')
    elif method == 2:
        pattern_test_path = os.path.join(pattern_path, f'db{method}_{num_ring}_{num_sector}_pos{pos_noise_std}_mv{mv_noise_std}_fs{num_false_star}_test')
    else:
        print('Invalid method!')
        return
    if not os.path.exists(pattern_test_path):
        os.makedirs(pattern_test_path)

    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.uniform(-np.pi, np.pi, num_sample)
    des = np.arcsin(np.random.uniform(-1, 1, num_sample))

    patterns = []
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

        if len(stars) < 1:
            continue

        # distances = cdist(stars, stars, 'euclidean')
        distances = np.linalg.norm(stars[:,None,:] - stars[None,:,:], axis=-1)
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
            # check if false star
            star_id = star_table.get(tuple(star), -1)
            if star_id == -1:
                continue
            if star[0] < region_r or star[0] > h-region_r or star[1] < region_r or star[1] > w-region_r:
                continue

            # angles is sorted by distance with accending order
            ss, ags = stars[np.argsort(ds)], ags[np.argsort(ds)]
            ds = np.sort(ds)
            ss, ags = ss[ds < region_r], ags[ds < region_r]
            # remove the first element, which is reference star
            assert star[0] == ss[0][0] and star[1] == ss[0][1] and ds[0] == 0 and ags[0] == 0
            ss, ds, ags = ss[1:], ds[1:], ags[1:]
            # calculate the relative coordinates
            ss = ss-star
            if len(ss) < 2:
                continue

            if method == 1:
                ag = ags[0]
                M = np.array([[np.cos(ag), -np.sin(ag)], [np.sin(ag), np.cos(ag)]])
                ss = np.dot(ss, M)
                # calculate the pattern
                grid_pattern = np.zeros((grid_len, grid_len), dtype=int)
                for s in ss:
                    row = int(s[0]/region_r*grid_len)
                    col = int(s[1]/region_r*grid_len)
                    grid_pattern[row][col] = 1
                patterns.append({
                    'pattern': ''.join(map(str, grid_pattern.flatten())), 
                    'id': star_id
                })
            else:
                # rotate the stars until the nearest star lies on the horizontal axis
                ags = ags-ags[0]
                # make sure angles are in the range of [-pi, pi]
                ags %= 2*np.pi
                ags[ags > np.pi] -= 2*np.pi
                ags[ags < -np.pi] += 2*np.pi
                # get the radial and cyclic features
                ring_counts, _ = np.histogram(ds, bins=num_ring, range=(0, region_r))
                sector_counts, _ = np.histogram(ags, bins=num_sector, range=(-np.pi, np.pi))
                patterns.append({
                    'pattern': ''.join(map(str, np.concatenate([ring_counts, sector_counts]))),
                    'id': star_id
                })
            
    df = pd.DataFrame(patterns)
    df.to_csv(os.path.join(pattern_test_path, str(uuid.uuid1())),index=False)


def generate_nn_samples(mode: str, num_vec: int, idxs: list, num_ring: int = 200, num_sector: int = 30, num_neighbor: int = 4, pos_noise_std: float = 0, mv_noise_std: float = 0, num_false_star: int = 0):
    '''
        Generate radial and cyclic features dataset for NN model using the given star catalogue.
    Args:
        mode: generation mode
            1. 'random', use uniformed distributed vector on sphere
            2. 'supplementary', use catalogue index to generate samples for specific star
        num_vec: the number of vectors to be generated
        idxs: the indexes of star catalogue used to generate dataset
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
        ras = np.clip(catalogue.loc[idxs, 'RA']+np.radians(np.random.normal(0, 1)), -np.pi, np.pi)
        des = np.clip(catalogue.loc[idxs, 'DE']+np.radians(np.random.normal(0, 1)), -np.pi/2, np.pi/2)
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
        stars = np.array(get_star_centroids(img))
        if len(stars) < num_neighbor+1:
            continue

        distances = cdist(stars, stars, 'euclidean')
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
            if star[0] < h/2-300 or star[0] > h/2+300 or star[1] < w/2-300 or star[1] > w/2+300:
                continue
            # check if false star
            star_id = star_table.get(tuple(star), -1)
            if star_id == -1:
                continue
            # get catalogue index of the guide star
            catalogue_idx = catalogue[catalogue['Star ID'] == star_id].index.to_list()[0]
            # generate label information for training and testing
            label = {
                'ra': ra,
                'de': de,
                'star_id': star_id,
                'catalogue_idx': catalogue_idx
            }

            # angles is sorted by distance with accending order
            ags = ags[np.argsort(ds)]
            ds = np.sort(ds)
            ags = ags[ds < region_r]
            # remove the first element of ags & ds, which is reference star
            ds, ags = ds[1:], ags[1:]
            # make sure several neighbor stars are located in the region
            if len(ags) < num_neighbor+num_false_star:
                continue
        
            ring_counts, _ = np.histogram(ds, bins=num_ring, range=(0, region_r))
            for i, rc in enumerate(ring_counts):
                label[f'ring_{i}'] = rc

            # uses several neighbor stars as the starting angle to obtain the cyclic features
            for i, ag in enumerate(ags[:num_neighbor]):
                rotated_ags = ags - ag
                rotated_ags %= 2*np.pi
                rotated_ags[rotated_ags > np.pi] -= 2*np.pi
                rotated_ags[rotated_ags < -np.pi] += 2*np.pi
                sector_counts, _ = np.histogram(ags, bins=num_sector, range=(-np.pi, np.pi))
                for j, sc in enumerate(sector_counts):
                    label[f'neighbor_{i}_sector_{j}'] = sc
            labels.append(label)

    # return the label information
    df = pd.DataFrame(labels)
    return df


def aggregate_pm_test(num_round: int, methods: list = [1, 2], num_sample: int = 1000, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30, pos_noise_stds: list = [0, 0.5, 1, 1.5, 2], mv_noise_stds: list = [0, 0.1, 0.2], num_false_stars: list = [0, 1, 2, 3, 4, 5]):
    '''
        Aggregate the pattern test.
    Args:
        num_round: the number of rounds to generate the test
                    if num_round == 0: only aggregate the test cases
        methods: the methods to generate the pattern
        num_sample: the number of samples to be generated
        grid_len: the length of the grid
        num_ring: the number of rings
        num_sector: the number of sectors
        pos_noise_stds: the standard deviations of the positional noise
        mv_noise_stds: the standard deviations of the magnitude noise
        num_false_stars: the number of false stars
    '''

    def merge_test(test_path: str):
        '''
            Merge the test cases into a single file.
        Args:
            path: the path to the test cases
        '''
        files = os.listdir(test_path)
        df = pd.concat([pd.read_csv(os.path.join(test_path, file)) for file in files if file != 'test.csv'], ignore_index=True)
        df.to_csv(f"{test_path}/test.csv", index=False)

    # use thread pool
    num_thread = len(pos_noise_stds)+len(mv_noise_stds)+len(num_false_stars)
    pool = ThreadPoolExecutor(max_workers=num_thread)
    # iterate to generate the test cases
    all_task = []
    for _ in range(num_round):
        for method in methods:
            if method > 2:
                continue
            for pos_noise_std in pos_noise_stds:
                task = pool.submit(generate_pm_samples, method, num_sample, grid_len=grid_len, um_ring=num_ring, num_sector=num_sector, pos_noise_std=pos_noise_std)
                all_task.append(task)
            for mv_noise_std in mv_noise_stds:
                task = pool.submit(generate_pm_samples, method, num_sample, grid_len=grid_len, num_ring=num_ring, num_sector=num_sector, mv_noise_std=mv_noise_std)
                all_task.append(task)
            for num_false_star in num_false_stars:    
                task = pool.submit(generate_pm_samples, method, num_sample, grid_len=grid_len, num_ring=num_ring, num_sector=num_sector, num_false_star=num_false_star)
                all_task.append(task)
    
    # wait for all task done
    for task in all_task:
        task.result()
    
    # aggregate the test
    for method in methods:
        if method == 1:
            for pos_noise_std in pos_noise_stds:
                merge_test(os.path.join(pattern_path, f'db{method}_{grid_len}x{grid_len}_pos{pos_noise_std}_mv{0}_fs{0}_test'))
            for mv_noise_std in mv_noise_stds:
                merge_test(os.path.join(pattern_path, f'db{method}_{grid_len}x{grid_len}_pos{0}_mv{mv_noise_std}_fs{0}_test'))
            for num_false_star in num_false_stars:
                merge_test(os.path.join(pattern_path, f'db{method}_{grid_len}x{grid_len}_pos{0}_mv{0}_fs{num_false_star}_test'))
        elif method == 2:
            for pos_noise_std in pos_noise_stds:
                merge_test(os.path.join(pattern_path, f'db{method}_{num_ring}_{num_sector}_pos{pos_noise_std}_mv{0}_fs{0}_test'))
            for mv_noise_std in mv_noise_stds:
                merge_test(os.path.join(pattern_path, f'db{method}_{num_ring}_{num_sector}_pos{0}_mv{mv_noise_std}_fs{0}_test'))
            for num_false_star in num_false_stars:
                merge_test(os.path.join(pattern_path, f'db{method}_{num_ring}_{num_sector}_pos{0}_mv{0}_fs{num_false_star}_test'))
        else:
            continue


def aggregate_nn_dataset(types: dict = {'train': 20, 'validate': 10, 'test': 5}, num_ring: int = 200, num_sector: int = 30, num_neighbor: int = 4, pos_noise_stds: list = [0.5, 1, 1.5, 2], mv_noise_stds: list = [0.1, 0.2], num_false_stars: list = [1, 2, 3, 4, 5], num_thread: int = 20, num_round: int = 3):
    '''
        Aggregate the dataset. Firstly, the number of samples for each class is counted. Then, roughly generate classes with too few samples using generate_nn_samples function's 'random' mode. Lastly, the rest are finely generated to ensure that the number of samples in each class in the entire dataset reaches the standard.
    Args:
        types: key->the types of the dataset, values->the minumin number of samples for each class
        num_ring: the number of rings
        num_sector: the number of sectors
        num_neighbor: the number of neighbor stars
        pos_noise_stds: list of positional noise
        num_thread: the number of threads to generate the dataset
        num_round: the number of rounds for rough generation
    '''

    def get_cata_idxs():
        '''
            Get the indexes of the catalogue.
        '''
        df_info = df['star_id'].value_counts()
        df_info = df_info[df_info < 20]
        idxs = catalogue[catalogue['Star ID'].isin(df_info.index)].index.to_list()
        idxs = []

    # generate config
    gen_cfg = f'{num_ring}_{num_sector}_{num_neighbor}'
    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thread)
    # iterate to roughly generate the dataset
    all_task = defaultdict(list)
    for key in types.keys():
        # count the number of samples for each class
        if os.path.exists(os.path.join(dataset_path, gen_cfg, key, 'labels.csv')):
            df = pd.read_csv(os.path.join(dataset_path, gen_cfg, key, 'labels.csv'))
            df_info = df['star_id'].value_counts()
            pct = len(df_info[df_info < types[key]])/len(catalogue)
            if pct < 0.5:
                continue
        # roughly generate using generate_nn_samples 'random' mode
        num_vec = types[key]*10
        for _ in range(num_round):
            # train, validate and default test
            task = pool.submit(generate_nn_samples, 'random', num_vec, [], num_ring, num_sector, num_neighbor)
            all_task[key].append(task)
            
            # # positional noise test
            # for pns in pos_noise_stds:
            #     task = pool.submit(generate_nn_samples, 'random', num_vec, [], num_ring, num_sector, num_neighbor, pos_noise_std=pns)
            #     all_task[f'{key}/pos{pns}'].append(task)

            # # magnitude noise test
            # for mns in mv_noise_stds:
            #     task = pool.submit(generate_nn_samples, 'random', num_vec, [], num_ring, num_sector, num_neighbor, mv_noise_std=mns)
            #     all_task[f'{key}/mv{mns}'].append(task)
                
            # # false star test
            # for nfs in num_false_stars:
            #     task = pool.submit(generate_nn_samples, 'random', num_vec, [], num_ring, num_sector, num_neighbor, num_false_star=nfs)
            #     all_task[f'{key}/fs{nfs}'].append(task)

    # wait for all tasks to be done
    for key in all_task.keys():
        # make directory for every type of dataset
        path = os.path.join(dataset_path, gen_cfg, key)
        if not os.path.exists(path):
            os.makedirs(path)

        # store the samples        
        dfs = []
        for task in all_task[key]:
            df = task.result()
            df.to_csv(f"{path}/{uuid.uuid1()}", index=False)
            dfs.append(df)

        # remain the old samples in labels.csv
        if os.path.exists(f'{path}/labels.csv'):
            dfs.append(pd.read_csv(f'{path}/labels.csv'))

        # aggregate the dataset
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(f'{path}/labels.csv', index=False)
    
    for 

    # calculate samples less than limit
    df_info = df['star_id'].value_counts()
    df_info = df_info[df_info < 20]
    idxs = catalogue[catalogue['Star ID'].isin(df_info.index)].index.to_list()
    print(len(df_info), df_info.tail(50))

    # for class whose sample less than num_sample_per_class, generate the dataset again
    
    
    print(idxs)


if __name__ == '__main__':
    # generate_pm_database(1, grid_len=60)
    # generate_pm_database(2)
    # aggregate_pm_test(0, [2], grid_len = 60, num_sample=100)
    aggregate_nn_dataset({'train': 1}, num_neighbor=3)
    