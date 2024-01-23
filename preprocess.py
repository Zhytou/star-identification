import json
import cv2
import numpy as np
from skimage import measure

from simulate import noise_std

np.set_printoptions(threshold=np.inf)


def get_star_centroids(img: np.ndarray) -> list[tuple[int, int]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
    Returns:
        centroids: the centroids of the stars in the image
    '''

    def cal_multiwind_threshold(img: np.ndarray, wind_len: int=200, num_wind: int=5) -> int:
        """
            Calculate the threshold of the image using the method "multi-window threshold division" from https://ieeexplore.ieee.org/abstract/document/1008988.
        Args:
            wind_len: the length of the window
            num_wind: the number of the windows
        Returns:
            threshold: the threshold of the image
        """        
        # initialize random windows
        winds = []
        for i in range(num_wind):
            x = np.random.randint(0, w - wind_len)
            y = np.random.randint(0, h - wind_len)
    
            wind = img[y:y+wind_len, x:x+wind_len]    
            mean = np.mean(wind)  
            winds.append(mean)

        # calculate the mean of the window means
        tot_mean = np.mean(winds)

        # threshold = background_mean + std * 5
        threshold = tot_mean + noise_std * 5

        return threshold

    def group_star(img: np.ndarray, method: int) -> list[list[tuple[int, int]]]:
        """
            Group the facula(potential star) in the image.
        Args:
            img: the image to be processed
            method: method of connectivity
        Returns:
        """
        # if img[u, v] > 0: 1, else: 0
        _, binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

        # label connected regions of the same value in the binary image
        labeled_img, label_num = measure.label(binary_img, return_num=True, connectivity=method)

        group_coords = []
        for label in range(1, label_num + 1):
            # get the coords for each label
            rows, cols = np.nonzero(labeled_img == label)
            group_coords.append(list(zip(rows, cols)))

        return group_coords

    # get the image size
    h, w = img.shape

    # calaculate the threshold
    threshold = cal_multiwind_threshold(img)

    # if img[u, v] < threshold: 0, else: img[u, v]
    _, nimg = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    # rough group star using connectivity
    group_coords = group_star(nimg, 2)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for coords in group_coords:
        row_sum = 0
        col_sum = 0
        gray_sum = 0
        for row, col in coords:
            row_sum += row * (img[row][col] - threshold)
            col_sum += col * (img[row][col] - threshold)
            gray_sum += img[row][col] - threshold
        centroids.append((round(row_sum/gray_sum), round(col_sum/gray_sum)))

    return centroids


if __name__ == '__main__':
    # read the image
    img = cv2.imread('test2.png', 0)

    # read star info
    with open('test2.json', 'r') as f:
        real_stars = json.load(f)

    coords = []
    for i in range(len(real_stars)):
        row, col = real_stars[i][1]
        coords.append((row, col))

    # get the centroids
    stars = get_star_centroids(img)

    # test the accuracy
    h, w = img.shape
    for star in stars:
        row, col = star
        for coord in coords:
            if abs(row - coord[0]) < 5 and abs(col - coord[1]) < 5:
                print('True')
