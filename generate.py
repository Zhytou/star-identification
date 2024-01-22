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
        img, _ = create_star_image(69, -12, -13)

        # get the centroids of the stars in the image
        stars = get_star_centroids(img)
        # data structure for fast nearest neighbour search
        tree = KDTree(stars)

        stars = np.array(stars)
        # choose a guide star as the reference star, and find the nearest star to it
        for guide_star in stars:
            if guide_star[0] - w/2 < 0 or guide_star[0] + w/2 > img.shape[0] or guide_star[1] - l/2 < 0 or guide_star[1] + l/2 > img.shape[1]:
                continue
            _, idxs = tree.query(guide_star, 2)
            nearest_star = stars[idxs[1]]
            
            # calculate rotation angle & matrix
            angle = degrees(atan((nearest_star[1] - guide_star[1]) / (nearest_star[0] - guide_star[0])))
            M = cv2.getRotationMatrix2D(guide_star, angle, 1)
            rotated_img = cv2.warpAffine(img, M, img.shape)

            # crop and save the image
            row1, row2 = int(guide_star[0] - w/2), int(guide_star[0] + w/2)
            col1, col2 = int(guide_star[1] - l/2), int(guide_star[1] + l/2)

            cv2.imwrite(f'{i}.png', rotated_img)
            # cv2.imwrite(f'{i}.png', rotated_img[row1:row2, col1:col2])
        # generate label information for training




if __name__ == '__main__':
    generate_dataset(1)
