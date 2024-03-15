import os
import uuid
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from math import degrees, atan
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

import config
from config import catalogue_path, config_name, num_ring, num_sector, num_neighbor_limit, region_r, num_false_star
from simulate import create_star_image
from preprocess import get_star_centroids


dataset_root_path = 'data'
catalogue = pd.read_csv(catalogue_path, usecols= ["Star ID", "RA", "DE", "Magnitude"])
num_input = num_ring + num_sector * num_neighbor_limit
num_class = len(catalogue)

# image dataset setting
img_dataset_sub_path = os.path.join(dataset_root_path, f'star_images/{config_name}')
# point dataset setting
point_dataset_path = os.path.join(dataset_root_path, f'star_points/{config_name}')


def get_rotation_angle(neighbor_star: np.ndarray, reference_star: np.ndarray) -> float:
    '''
        Calculate the angle to rotate the neighbor star around reference star until it aligns to x axis.
    Args:
        neighbor_star: the neighbor star
        reference_star: the reference star
    Returns:
        angle: the angle to rotate the nearest neighbor star around reference star
    '''
    if neighbor_star[1] == reference_star[1]:
        angle = 90
        if neighbor_star[0] > reference_star[0]:
            angle = 270
    elif neighbor_star[0] == reference_star[0]:
        angle = 0
        if neighbor_star[1] < reference_star[1]:
            angle = 180
    else:
        angle = abs(degrees(atan((neighbor_star[0] - reference_star[0]) / (neighbor_star[1] - reference_star[1]))))
        # (-90, 90) -> (0, 360)
        if neighbor_star[1] > reference_star[1]:
            if neighbor_star[0] > reference_star[0]:
                angle = 360 - angle
        else:
            if neighbor_star[0] < reference_star[0]:
                angle = 180 - angle
            else:
                angle = 180 + angle
    return angle
    

def generate_image_dataset(num_sample: int, h: int=224, w: int=224):
    '''
        Generate the dataset from the given star catalogue.

        Here's the brief introduction of the generation process. Select a guide star as the reference star. Then, find the closest neighbor star and rotate the image until the adjacent star lies on the horizontal axis. Finally, crop the image and save it.
    Args:
        num_sample: the number of samples to be generated
        h: the width of the image
        w: the length of the image
    '''

    dataset_path = os.path.join(dataset_root_path, img_dataset_sub_path)
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
        if len(star_info) < 3 + 1:
            continue
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))

        # get the centroids of the stars in the image
        stars = get_star_centroids(img)
        # data structure for fast nearest neighbour search
        tree = KDTree(stars)

        stars = np.array(stars)
        # choose a guide star as the reference star
        for guide_star in stars:
            # check if false star
            star_id = star_table.get(tuple(guide_star), -1)
            if star_id == -1:
                continue
            # check if the guide star is too close to the edge
            row1, row2 = int(guide_star[0] - h/2), int(guide_star[0] + h/2)
            col1, col2 = int(guide_star[1] - w/2), int(guide_star[1] + w/2)
            if row1 < 0 or row2 > img.shape[0] or col1 < 0 or col2 > img.shape[1]:
                continue
            
            # find the top star_num_per_img nearest neighbor star to the reference one
            distances, idxs = tree.query(guide_star, 3 + 1)
            # make sure at least star_num_per_img neighbor stars are located in the region
            if distances[-1] >= min(h, w) / 2:
                continue

            # find the nearest neighbor star
            nearest_star = stars[idxs[1]]
            # calculate rotation angle & matrix
            angle = get_rotation_angle(nearest_star, guide_star)
            # be careful that getRotationMatrix2D takes (width, height) as input, in other words (col, row)
            M = cv2.getRotationMatrix2D((float(guide_star[1]), float(guide_star[0])), -angle, 1)
            rotated_img = cv2.warpAffine(img, M, img.shape)

            # crop and save the image
            img_name = f'{datetime.now()}_{star_id}.png'
            cv2.imwrite(os.path.join(dataset_path, img_name), rotated_img[row1:row2, col1:col2])
            # generate label information for training and testing
            labels.append({
                'img_name': img_name,
                'ra': ra,
                'de': de,
                'star_id': star_id
            })

    # save the label information
    df = pd.DataFrame(labels)

    # read the old label information
    if os.path.exists(os.path.join(dataset_path, 'labels.csv')):
        odf = pd.read_csv(os.path.join(dataset_path, 'labels.csv'), usecols=['img_name', 'ra', 'de', 'star_id'])
        df = pd.concat([df, odf], ignore_index=True)
    df.to_csv(os.path.join(dataset_path, 'labels.csv'))


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
            if len(ags) < num_neighbor_limit:
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
    #     # if i%6 == 0:
    #     #     task = pool.submit(generate_point_dataset, 'train', 2000)
    #     # elif i%6 == 1:
    #     #     task = pool.submit(generate_point_dataset, 'validate', 2000)
    #     # elif i%6 == 2:
    #     #     task = pool.submit(generate_point_dataset, 'test', 1000)
    #     # elif i%6 == 3:
    #     #     task = pool.submit(generate_point_dataset, 'positional_noise_test', 1000)
    #     # elif i%6 == 4:
    #     #     task = pool.submit(generate_point_dataset, 'magnitude_noise_test', 1000)
    #     # else:
    #     #     task = pool.submit(generate_point_dataset, 'false_star_test', 1000)
    #     all_task.append(task)
    # # wait for all tasks to be done
    # for task in all_task:
    #     task.result()

    generate_point_dataset('false_star_test', 1000)
    # generate_point_dataset('positional_noise_test', 1000)
    # generate_point_dataset('magnitude_noise_test', 1000)

    # after all tasks are done, aggregate the dataset
    for type in ['train', 'validate', 'test', 'positional_noise_test', 'magnitude_noise_test', 'false_star_test']:
        dataset_path = os.path.join(point_dataset_path, type)
        if type == 'false_star_test':
            dataset_path = os.path.join(dataset_path, f'{num_false_star}')
        files = os.listdir(dataset_path)
        labels = pd.concat([pd.read_csv(os.path.join(dataset_path, file)) for file in files if file != 'labels.csv'], ignore_index=True)
        labels_info = labels['star_id'].value_counts()
        print(type, len(labels_info), labels_info.head(5), labels_info.tail(5))
        
    # #     if type == 'train':
    # #         labels = labels.groupby('star_id').head(70)
    # #     elif type == 'validate':
    # #         labels = labels.groupby('star_id').head(20)
    # #     else:
    # #         labels = labels.groupby('star_id').head(20)

        labels.to_csv(f"{dataset_path}/labels.csv", index=False)
    
    # # for type in ['train', 'validate', 'test']:
    # #     dataset_path = os.path.join(point_dataset_path, type)
    # #     labels = pd.read_csv(os.path.join(dataset_path, 'labels.csv'))
    # #     labels_info = labels['star_id'].value_counts()
    # #     print(type, len(labels_info), labels_info.head(5), labels_info.tail(5))    