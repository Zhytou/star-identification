import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from math import degrees, atan
from scipy.spatial import KDTree

from simulate import create_star_image
from preprocess import get_star_centroids


dataset_root_path = 'data'
# image dataset setting
img_dataset_sub_path = 'star_images'
label_file = 'labels.csv'
# star number limit per image
star_num_limit = 5


def generate_image_dataset(num_img: int, h: int=256, w: int=256):
    '''
        Generate the dataset from the given star catalogue.

        Here's the brief introduction of the generation process. Select a guide star as the reference star. Then, find the closest neighbor star and rotate the image until the adjacent star lies on the horizontal axis. Finally, crop the image and save it.
    Args:
        num_img: the number of images to be generated
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
    ras = np.random.randint(-180, 180, num_img)
    des = np.random.uniform(-90, 90, num_img)

    # generate the star image
    for i in range(num_img):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ras[i], des[i], 0)
        if len(star_info) < star_num_limit:
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
            
            # find the top star_num_limit nearest neighbor star to the reference one
            distances, idxs = tree.query(guide_star, star_num_limit)
            # make sure at least star_num_limit neighbor stars are located in the region
            if distances[-1] >= min(h, w) / 2:
                continue

            # find the nearest neighbor star
            nearest_star = stars[idxs[1]]
            # calculate rotation angle & matrix
            if nearest_star[1] == guide_star[1]:
                angle = 90
            else:
                angle = degrees(atan((nearest_star[0] - guide_star[0]) / (nearest_star[1] - guide_star[1])))
            # be careful that getRotationMatrix2D takes (width, height) as input, in other words (col, row)
            M = cv2.getRotationMatrix2D((float(guide_star[1]), float(guide_star[0])), angle, 1)
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
        df = pd.merge(df, odf, on='img_name')
    df.to_csv(os.path.join(dataset_path, label_file))


if __name__ == '__main__':
    generate_image_dataset(100)
