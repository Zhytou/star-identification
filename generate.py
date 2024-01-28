import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from math import degrees, radians, atan, cos, sin
from scipy.spatial import KDTree

from simulate import create_star_image
from preprocess import get_star_centroids


dataset_root_path = 'data'
# star number limit per sample
star_num_per_sample = 7
label_file = 'labels.csv'

# image dataset setting
img_dataset_sub_path = 'star_images'

# point dataset setting
point_dataset_sub_path = 'star_points'


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

    if not os.path.exists(dataset_root_path):
        os.mkdir(dataset_root_path)
    if not os.path.exists(os.path.join(dataset_root_path, img_dataset_sub_path)):
        os.mkdir(os.path.join(dataset_root_path, img_dataset_sub_path))

    dataset_path = os.path.join(dataset_root_path, img_dataset_sub_path)

    # store the label information
    labels = []

    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.randint(-180, 180, num_sample)
    des = np.random.uniform(-90, 90, num_sample)

    # generate the star image
    for i in range(num_sample):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ras[i], des[i], 0)
        if len(star_info) < star_num_per_sample + 1:
            continue
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))

        # get the centroids of the stars in the image
        stars = get_star_centroids(img)
        # assert len(stars) == len(star_info), f'{ras[i], des[i]}'
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
            
            # find the top star_num_per_sample nearest neighbor star to the reference one
            distances, idxs = tree.query(guide_star, star_num_per_sample + 1)
            # make sure at least star_num_per_sample neighbor stars are located in the region
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
                'ra': ras[i],
                'de': des[i],
                'star_id': star_id
            })

    # save the label information
    df = pd.DataFrame(labels)

    # read the old label information
    if os.path.exists(os.path.join(dataset_path, label_file)):
        odf = pd.read_csv(os.path.join(dataset_path, label_file), usecols=['img_name', 'ra', 'de', 'star_id'])
        df = pd.concat([df, odf], ignore_index=True)
    df.to_csv(os.path.join(dataset_path, label_file))


def generate_point_dataset(num_sample: int):
    '''
        Generate the dataset from the given star catalogue.
    Args:
        num_sample: the number of samples to be generated
    '''

    if not os.path.exists(dataset_root_path):
        os.mkdir(dataset_root_path)
    if not os.path.exists(os.path.join(dataset_root_path, point_dataset_sub_path)):
        os.mkdir(os.path.join(dataset_root_path, point_dataset_sub_path))

    dataset_path = os.path.join(dataset_root_path, point_dataset_sub_path)

    # store the label information
    labels = []

    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.randint(-180, 180, num_sample)
    des = np.random.uniform(-90, 90, num_sample)

    # generate the star image
    for i in range(num_sample):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ras[i], des[i], 0)
        if len(star_info) < star_num_per_sample + 1:
            continue
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))

        # get the centroids of the stars in the image
        stars = get_star_centroids(img)
        # assert len(stars) == len(star_info), f'{ras[i], des[i]}'
        # data structure for fast nearest neighbour search
        tree = KDTree(stars)

        stars = np.array(stars)
        # choose a guide star as the reference star
        for guide_star in stars:
            # check if false star
            star_id = star_table.get(tuple(guide_star), -1)
            if star_id == -1:
                continue
            
            # find the top star_num_per_sample nearest neighbor star to the reference one
            _, idxs = tree.query(guide_star, star_num_per_sample + 1)
            # find the nearest neighbor star
            nearest_star = stars[idxs[1]]
            # calculate rotation angle & matrix
            angle = radians(-get_rotation_angle(nearest_star, guide_star))
            M = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
            
            neighbor_stars = stars[idxs[1:]] - guide_star
            # calculate the angle of each neighbor star
            angles = np.array(list(map(lambda x: get_rotation_angle(x, guide_star), neighbor_stars)))
            # sort the neighbor stars by angle
            neighbor_stars = neighbor_stars[np.argsort(angles)]
            # calculate the position of the neighbor stars after rotation
            rotated_stars = np.round(np.dot(neighbor_stars, M.T), 5)
            # generate label information for training and testing
            label = {
                'ra': ras[i],
                'de': des[i],
                'star_id': star_id
            }
            for i, rotated_star in enumerate(rotated_stars):
                label[f'point{i}_x'], label[f'point{i}_y'] = rotated_star[0], rotated_star[1]
            labels.append(label)

    # save the label information
    df = pd.DataFrame(labels)

    # read the old label information
    if os.path.exists(os.path.join(dataset_path, label_file)):
        odf = pd.read_csv(os.path.join(dataset_path, label_file), usecols=['points', 'ra', 'de', 'star_id'])
        df = pd.concat([df, odf], ignore_index=True)
    df.to_csv(os.path.join(dataset_path, label_file))


if __name__ == '__main__':
    generate_point_dataset(10)
    # generate_image_dataset(100)
