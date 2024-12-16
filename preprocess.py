import cv2
import numpy as np
from skimage import measure
from math import radians

from simulate import create_star_image, ROI


def filter_image(img):
    '''
        Use gaussian filter to reduce the noise in the image.
    Args:
        img: the image to be processed
    Returns:
        filter_img: the image after filtering
        mse: the mean square error between the original image and the filtered image
    '''
    # get the image size
    h, w = img.shape

    # low pass filter for noise
    filter_img = cv2.GaussianBlur(img, (2*ROI+1, 2*ROI+1), 1)

    # caculate the MSE
    mse = np.sum((img - filter_img)**2) / (h * w)

    return filter_img, mse


def cal_multiwind_threshold(img: np.ndarray, wind_len: int=40, num_wind: int=20) -> int:
    """
        Calculate the threshold of the image using the method "multi-window threshold division" from https://ieeexplore.ieee.org/abstract/document/1008988.
    Args:
        wind_len: the length of the window
        num_wind: the number of the windows
    Returns:
        threshold: the threshold of the image
    """        
    # initialize random windows
    threshold = 0

    # get the image size
    h, w = img.shape

    for i in range(num_wind):
        x = np.random.randint(0, w - wind_len)
        y = np.random.randint(0, h - wind_len)

        wind = img[y:y+wind_len, x:x+wind_len]    
        mean = np.mean(wind)  
        std = np.std(wind)
        threshold += mean + 5 * std

    return round(threshold/num_wind)


def get_star_centroids(img: np.ndarray, method: str) -> list[tuple[float, float]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
        method: centroid algorithm
    Returns:
        centroids: the centroids of the stars in the image
    '''

    def group_star(img: np.ndarray, threshold: int, connectivity: int) -> list[list[tuple[int, int]]]:
        """
            Group the facula(potential star) in the image.
        Args:
            img: the image to be processed
            connectivity: method of connectivity
        Returns:
        """
        # if img[u, v] < threshold: 0, else: img[u, v]
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

        # if img[u, v] > 0: 1, else: 0
        _, binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

        # label connected regions of the same value in the binary image
        labeled_img, label_num = measure.label(binary_img, return_num=True, connectivity=connectivity)

        group_coords = []
        for label in range(1, label_num + 1):
            # get the coords for each label
            rows, cols = np.nonzero(labeled_img == label)
            coords = list(zip(rows, cols))
            # two small or too big to be a star
            if len(coords) < 9 or len(coords) > 100:
                continue
            group_coords.append(coords)

        return group_coords

    # get the image size
    h, w = img.shape

    # low pass filter for noise
    filter_img, _ = filter_image(img)

    # calaculate the threshold
    threshold = cal_multiwind_threshold(filter_img, wind_len=int(max(h*0.7, w*0.7)), num_wind=10)

    # rough group star using connectivity
    group_coords = group_star(filter_img, threshold, 2)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for coords in group_coords:
        row_sum = 0
        col_sum = 0
        gray_sum = 0
        for row, col in coords:
            if method == 'default centroid':
                row_sum += row * filter_img[row][col]
                col_sum += col * filter_img[row][col]
                gray_sum += filter_img[row][col]
            elif method == 'square centroid':
                row_sum += row * pow(filter_img[row][col], 2)
                col_sum += col * pow(filter_img[row][col], 2)
                gray_sum += pow(filter_img[row][col], 2)
            elif method == 'centroid with threshold':
                row_sum += row * (filter_img[row][col] - threshold)
                col_sum += col * (filter_img[row][col] - threshold)
                gray_sum += filter_img[row][col] - threshold
            else:
                print('wrong centroid method!')
                return []
        centroids.append((round(row_sum/gray_sum, 3), round(col_sum/gray_sum, 3)))

    return centroids


if __name__ == '__main__':
    filter_test = False
    centroid_test = True

    if filter_test:
        ra, de, roll = radians(29.2104), radians(-12.0386), radians(0)
        img, _ = create_star_image(ra, de, roll, white_noise_std=20)
        h, w = img.shape
        filter_img, mse = filter_image(img)
        threshold = cal_multiwind_threshold(filter_img, wind_len=int(max(h*0.7, w*0.7)), num_wind=10)
        cv2.imwrite('before_filter.png', img)
        cv2.imwrite('after_filter.png', filter_img)
        print(mse, threshold)

    if centroid_test:
        arr_pos_err = {
            'default centroid': [],
            'square centroid': [],
            'centroid with threshold': []
        }
        for white_noise_std in range(0, 10, 2):
            # times of estimation using centroid algorithm
            num_esti_times = 5
            # random ra & de test
            num_test = 100
            # generate random right ascension[0, 360] and declination[-90, 90]
            ras = np.random.uniform(0, 2*np.pi, num_test)
            des = np.arcsin(np.random.uniform(-1, 1, num_test))
            # centroid position error
            pos_err = {
                'default centroid': 0,
                'square centroid': 0,
                'centroid with threshold': 0
            }
            # generate the star image
            for i in range(num_test):
                img, star_info = create_star_image(ras[i], des[i], 0, white_noise_std=0)
                real_coords = np.array([x[1] for x in star_info])
                # estimate the centroids of the stars in the image
                for method in pos_err.keys():
                    arr_esti_coords = []
                    for _ in range(num_esti_times):
                        esti_coords = np.array(get_star_centroids(img, method='centroid with threshold'))
                        if len(real_coords) != len(esti_coords):
                            print('Wrong star number extracted: ', len(real_coords), len(esti_coords))
                            continue
                        arr_esti_coords.append(esti_coords)
                    avg_esti_coords = np.mean(arr_esti_coords, axis=0)
                    # calculate the position error
                    diff = real_coords[:, None] - avg_esti_coords
                    dist = np.linalg.norm(diff, axis=-1)
                    pos_err[method] += np.sum(np.min(dist, axis=0))

            for method in pos_err.keys():
                arr_pos_err[method].append(pos_err[method]/num_test)
        
        print(arr_pos_err)
