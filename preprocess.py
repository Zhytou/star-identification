import json
import cv2
import numpy as np
from skimage import measure

from simulate import noise_std, create_star_image

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

    # if img[u, v] < threshold + 20: 0, else: img[u, v]
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
    error_num = 0
    num_test = 1000
    
    # generate random right ascension[0, 360] and declination[-90, 90]
    ras = np.random.randint(-180, 180, num_test)
    des = np.random.randint(-90, 90, num_test)

    # generate the star image
    for i in range(num_test):
        img, star_info = create_star_image(ras[i], des[i], 0)
        
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))

        # get the centroids of the stars in the image
        stars = get_star_centroids(img)

        if len(stars) != len(star_info):
            error_num += 1
            # print(f'{ras[i], des[i]}')
            # print(len(stars), len(star_info))
            # print(stars)
            # print(star_info)

        # for star in stars:
        #     # check if false star
        #     star_id = star_table.get(tuple(star), -1)
        #     if star_id == -1:
        #         error_num += 1
        #         # print(f'{ras[i], des[i]}')
        #         # print(len(stars), len(star_info))
        #         # print(stars)
        #         # print(star_info)
        #         break
    
    print(f'error_num: {error_num}, accuracy: {(num_test - error_num)*100.0/num_test}%')
