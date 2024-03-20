import os
import uuid
from math import degrees
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

from simulate import w, h, FOV, catalogue_path, create_star_image
from preprocess import get_star_centroids


# region radius in pixels
region_r = int(w/FOV*6)

# star catalogue
catalogue = pd.read_csv(catalogue_path, usecols= ["Star ID", "RA", "DE", "Magnitude"])

# define the path to store the database and pattern as well as dataset
simulte_name = f"{os.path.basename(catalogue_path).rsplit('.', 1)[0]}_{w}x{h}_{FOV}x{FOV}"
database_path = f'database/{simulte_name}'
pattern_path = f'pattern/{simulte_name}'
point_dataset_path = f'data/star_points/{simulte_name}'


def generate_pattern_database(method: int, use_preprocess: bool = False, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30):
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
        img, star_info = create_star_image(degrees(ra), degrees(de), 0)
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


def generate_pattern_test_case(method: int, num_sample: int, use_preprocess: bool = False, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30, pos_noise_std: int = 0, mv_noise_std: float = 0, num_false_star: int = 0):
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
    ras = np.random.uniform(-180, 180, num_sample)
    des = np.degrees(np.arcsin(np.random.uniform(-1, 1, num_sample)))

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


def generate_point_dataset(type: str, num_sample: int, num_ring: int = 200, num_sector: int = 30, pos_noise_std: int = 0, mv_noise_std: float = 0, num_false_star: int = 0, num_neighbor_limit: int = 4):
    '''
        Generate the dataset from the given star catalogue.
    Args:
        type: 'train', 'validate', 'test'
        num_sample: the number of samples to be generated
        num_ring: the number of rings
        num_sector: the number of sectors
        pos_noise_std: the standard deviation of the positional noise
        mv_noise_std: the standard deviation of the magnitude noise
        num_false_star: the number of false stars
    '''
    dataset_path = os.path.join(point_dataset_path, f'{num_ring}_{num_sector}_{num_neighbor_limit}', type)
    if type == 'test':
        database_path = os.path.join(database_path, f'pos{pos_noise_std}_mv{mv_noise_std}_fs{num_false_star}_test')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # store the label information
    labels = []

    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.uniform(-180, 180, num_sample)
    des = np.degrees(np.arcsin(np.random.uniform(-1, 1, num_sample)))

    # generate the star image
    for ra, de in zip(ras, des):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ra, de, 0, pos_noise_std=pos_noise_std, mv_noise_std=mv_noise_std, num_false_star=num_false_star)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))
        # get the centroids of the stars in the image
        stars = np.array(get_star_centroids(img))
        if len(stars) < num_neighbor_limit+1:
            continue

        distances = cdist(stars, stars, 'euclidean')
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
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
            if len(ags) < num_neighbor_limit+num_false_star:
                continue
        
            ring_counts, _ = np.histogram(ds, bins=num_ring, range=(0, region_r))
            for i, rc in enumerate(ring_counts):
                label[f'ring_{i}'] = rc

            # uses several neighbor stars as the starting angle to obtain the cyclic features
            for i, ag in enumerate(ags[:num_neighbor_limit]):
                rotated_ags = ags - ag
                rotated_ags %= 2*np.pi
                rotated_ags[rotated_ags > np.pi] -= 2*np.pi
                rotated_ags[rotated_ags < -np.pi] += 2*np.pi
                sector_counts, _ = np.histogram(ags, bins=num_sector, range=(-np.pi, np.pi))
                for j, sc in enumerate(sector_counts):
                    label[f'neighbor_{i}_sector_{j}'] = sc
            labels.append(label)

    # save the label information
    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(dataset_path, str(uuid.uuid1())), index=False)


def aggregate_point_dataset(num_round: int, types: list = ['train', 'validate', 'test'], num_sample: int = 1000, num_ring: int = 200, num_sector: int = 30, pos_noise_stds: list = [0, 1, 2], mv_noise_stds: list = [0, 0.1, 0.2], num_false_stars: list = [0, 1, 2, 3, 4, 5], num_neighbor_limit: int = 4):
    '''
        Aggregate the dataset.
    Args:
        num_round: the number of rounds to generate the dataset
        types: the types of the dataset
        num_ring: the number of rings
        num_sector: the number of sectors
        num_neighbor_limit: the number of neighbor stars
    '''
    # use thread pool
    num_thread = num_round*len(types)
    pool = ThreadPoolExecutor(max_workers=num_thread)
    # generate dataset
    all_task=[]
    for i in range(num_thread):
        task = pool.submit(generate_point_dataset, 'train', num_sample, num_ring, num_sector)
        all_task.append(task)
    # wait for all tasks to be done
    for task in all_task:
        task.result()

    # after all tasks are done, aggregate the dataset
    for type in ['train', 'validate', 'test', 'positional_noise_test', 'magnitude_noise_test', 'false_star_test']:
        dataset_path = os.path.join(point_dataset_path, type)
        
        files = os.listdir(dataset_path)
        df = pd.concat([pd.read_csv(os.path.join(dataset_path, file)) for file in files if file != 'labels.csv'], ignore_index=True)
        df_info = df['star_id'].value_counts()
        print(type, len(df_info), df_info.head(5), df_info.tail(5))

        df.to_csv(f"{dataset_path}/labels.csv", index=False)


def aggregate_pattern_test(num_round: int, methods: list = [1, 2], num_sample: int = 1000, grid_len: int = 8, num_ring: int = 200, num_sector: int = 30, pos_noise_stds: list = [0, 1, 2], mv_noise_stds: list = [0, 0.1, 0.2], num_false_stars: list = [0, 1, 2, 3, 4, 5]):
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
                task = pool.submit(generate_pattern_test_case, method, num_sample, grid_len=grid_len, um_ring=num_ring, num_sector=num_sector, pos_noise_std=pos_noise_std)
                all_task.append(task)
            for mv_noise_std in mv_noise_stds:
                task = pool.submit(generate_pattern_test_case, method, num_sample, grid_len=grid_len, num_ring=num_ring, num_sector=num_sector, mv_noise_std=mv_noise_std)
                all_task.append(task)
            for num_false_star in num_false_stars:    
                task = pool.submit(generate_pattern_test_case, method, num_sample, grid_len=grid_len, num_ring=num_ring, num_sector=num_sector, num_false_star=num_false_star)
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
    

if __name__ == '__main__':
    # generate_pattern_database(1, grid_len=60)
    # generate_pattern_database(2)
    aggregate_pattern_test(0, [2], grid_len = 60, num_sample=100)
    