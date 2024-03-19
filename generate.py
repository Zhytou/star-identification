import os
import uuid
from math import degrees
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

import config
from config import catalogue_path, config_name, h, w, num_ring, num_sector, num_neighbor_limit, region_r, num_false_star
from simulate import create_star_image
from preprocess import get_star_centroids


catalogue = pd.read_csv(catalogue_path, usecols= ["Star ID", "RA", "DE", "Magnitude"])
num_input = num_ring + num_sector * num_neighbor_limit
num_class = len(catalogue)

database_path = f'database/{config_name}'
point_dataset_path = f'data/star_points/{config_name}'
pattern_path = f'pattern/{config_name}'


def generate_pattern_database(method: int, use_preprocess: bool = False, grid_len: int = 8):
    '''
        Generate the pattern database for the given star catalogue.
    Args:
        method: the method to generate the pattern database
                1: grid algorithm
                2: radial and cyclic algorithm
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
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
            pattern = ''.join(map(str, ring_counts))#np.concatenate([ring_counts, sector_counts])))
            record = {
                'pattern': pattern,
                'id': star_id
            }
            for i, d in enumerate(distances):
                record[f'd_{i}'] = d
            database.append(record)
        else:
            print('Invalid method!')

    df = pd.DataFrame(database)
    df.to_csv(f"{database_path}/db_{method}.csv", index=False)


def generate_pattern_test_case(method: int, num_sample: int, use_preprocess: bool = False, grid_len: int = 8):
    '''
        Generate pattern match test case.
    '''
    if not os.path.exists(pattern_path):
        os.makedirs(pattern_path)

    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.uniform(-180, 180, num_sample)
    des = np.degrees(np.arcsin(np.random.uniform(-1, 1, num_sample)))

    patterns = []
    # generate the star image
    for ra, de in zip(ras, des):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ra, de, 0)
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
            elif method == 2:
                # rotate the stars until the nearest star lies on the horizontal axis
                ags = ags-ags[0]
                # make sure angles are in the range of [-pi, pi]
                ags %= 2*np.pi
                ags[ags > np.pi] -= 2*np.pi
                ags[ags < -np.pi] += 2*np.pi
                # get the radial and cyclic features
                ring_counts, _ = np.histogram(ds, bins=num_ring, range=(0, region_r))
                sector_counts, _ = np.histogram(ags, bins=num_sector, range=(-np.pi, np.pi))
                pattern = ''.join(map(str, ring_counts))#np.concatenate([ring_counts, sector_counts])))
                record = {
                    'pattern': pattern,
                    'id': star_id
                }
                for i, d in enumerate(ds):
                    record[f'd_{i}'] = d
                patterns.append(record)
            else:
                pass
    
    df = pd.DataFrame(patterns)
    df.to_csv(f"{pattern_path}/test_{method}.csv", index=False)


def generate_point_dataset(type: str, num_sample: int):
    '''
        Generate the dataset from the given star catalogue.
    Args:
        type: 'train', 'validate', 'test', 'positional_noise_test', 'magnitude_noise_test', 'false_star_test'
        num_sample: the number of samples to be generated
    '''
    
    dataset_path = os.path.join(point_dataset_path, type)
    if type == 'positional_noise_test':
        config.type_noise = 'pos'
    elif type == 'magnitude_noise_test':
        config.type_noise = 'mv'
    elif type == 'false_star_test':
        config.type_noise = 'false_star'
        dataset_path = os.path.join(dataset_path, f'{num_false_star}')
    else:
        config.type_noise = 'none'
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
        img, star_info = create_star_image(ra, de, 0)
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
            if (config.type_noise == 'false_star' and len(ags) < num_neighbor_limit+num_false_star) or len(ags) < num_neighbor_limit:
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


if __name__ == '__main__':
    # num_thread = 6
    # # use thread pool
    # pool = ThreadPoolExecutor(max_workers=num_thread)
    # # generate dataset
    # all_task=[]
    # for i in range(num_thread):
    #     task = pool.submit(generate_point_dataset, 'train', 2000)
    #     # if i%3 == 0:
    #     #     task = pool.submit(generate_point_dataset, 'train', 2000)
    #     # elif i%3 == 1:
    #     #     task = pool.submit(generate_point_dataset, 'validate', 2000)
    #     # elif i%3 == 2:
    #     #     task = pool.submit(generate_point_dataset, 'test', 1000)
    #     all_task.append(task)
    # # wait for all tasks to be done
    # for task in all_task:
    #     task.result()

    # generate_point_dataset('false_star_test', 1000)
    # generate_point_dataset('positional_noise_test', 1000)
    # generate_point_dataset('magnitude_noise_test', 1000)

    # # after all tasks are done, aggregate the dataset
    # for type in ['train', 'validate', 'test', 'positional_noise_test', 'magnitude_noise_test', 'false_star_test']:
    #     dataset_path = os.path.join(point_dataset_path, type)
    #     if type == 'false_star_test':
    #         dataset_path = os.path.join(dataset_path, f'{num_false_star}')
    #     files = os.listdir(dataset_path)
    #     labels = pd.concat([pd.read_csv(os.path.join(dataset_path, file)) for file in files if file != 'labels.csv'], ignore_index=True)
    #     labels_info = labels['star_id'].value_counts()
    #     print(type, len(labels_info), labels_info.head(5), labels_info.tail(5))
        
    #     if type == 'train':
    #         labels = labels.groupby('star_id').head(70)
    #     elif type == 'validate':
    #         labels = labels.groupby('star_id').head(20)
    #     else:
    #         labels = labels.groupby('star_id').head(20)

    #     labels.to_csv(f"{dataset_path}/labels.csv", index=False)
    
    # for type in ['train', 'validate', 'test']:
    #     dataset_path = os.path.join(point_dataset_path, type)
    #     labels = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))
    #     labels_info = labels['star_id'].value_counts()
    #     print(type, len(labels_info), labels_info.head(5), labels_info.tail(5))

    generate_pattern_database(1, grid_len=60)
    generate_pattern_test_case(1, 500, grid_len=60)