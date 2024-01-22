import cv2
import numpy as np
from math import degrees, atan
from scipy.spatial import KDTree

from simulate import create_star_image
from preprocess import get_star_centroids

def generate_dataset(num_img: int, w: int=256, l: int=256):
    '''
        Generate the dataset from the given star catalogue.

        Here's the brief introduction of the generation process. Select a guide star as the reference star. Then, find the closest neighbor star and rotate the image until the adjacen star lies on the horizontal axis. Finally, crop the image and save it.
    Args:
        num_img: the number of images to be generated
        w: the width of the image
        l: the length of the image
    '''
    # generate random right ascension[0, 360] and declination[-90, 90]
    # ras = np.random.uniform(0, 360, num_img)
    # des = np.random.uniform(-90, 90, num_img)

    # generate the star image
    for i in range(num_img):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(69, -12, -13)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))

        # get the centroids of the stars in the image
        stars = get_star_centroids(img)
        assert(len(stars) == len(star_info))
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
            if guide_star[0] - w/2 < 0 or guide_star[0] + w/2 > img.shape[0] or guide_star[1] - l/2 < 0 or guide_star[1] + l/2 > img.shape[1]:
                continue
            
            # find the nearest neighbor star to the reference one
            _, idxs = tree.query(guide_star, 2)
            nearest_star = stars[idxs[1]]

            # calculate rotation angle & matrix
            angle = degrees(atan((nearest_star[0] - guide_star[0]) / (nearest_star[1] - guide_star[1])))
            # be careful getRotationMatrix2D takes (width, height) as input, in other words (col, row)
            M = cv2.getRotationMatrix2D((float(guide_star[1]), float(guide_star[0])), angle, 1)
            rotated_img = cv2.warpAffine(img, M, img.shape)

            # crop and save the image
            row1, row2 = int(guide_star[0] - w/2), int(guide_star[0] + w/2)
            col1, col2 = int(guide_star[1] - l/2), int(guide_star[1] + l/2)

            # generate label information for training and testing
            cv2.imwrite(f'{star_id}.png', rotated_img[row1:row2, col1:col2])



if __name__ == '__main__':
    generate_dataset(1)
