import numpy as np
import cv2
from skimage import measure

def get_centroids(img: np.ndarray):
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
    Returns:
        centroids: the centroids of the stars in the image
    '''

    def cal_multiwind_threshold(img: np.ndarray, wind_len: int, num_wind: int):
        """
            Calculate the threshold of the image using the method "multi-window threshold division" from https://ieeexplore.ieee.org/abstract/document/1008988.
        Args:
            wind_len: the length of the window
            num_wind: the number of the windows
        Returns:
            threshold: the threshold of the image
        """
        # get the image size
        l, w = img.shape
        
        # initialize random windows
        winds = []
        for i in range(num_wind):
            x = np.random.randint(0, l - wind_len)
            y = np.random.randint(0, w - wind_len)
    
            wind = img[y:y+wind_len, x:x+wind_len]    
            mean = np.mean(wind)  
            winds.append(mean)

        # calculate the mean of the window means
        tot_mean = np.mean(winds)

        # calculate the standard deviation of the image   
        std = np.std(img)

        # threshold = background_mean + std * 5
        threshold = tot_mean + std * 5

        return threshold

    def group_star(img: np.ndarray, method: int):
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
            xs, ys = np.nonzero(labeled_img == label)
            group_coords.append(list(zip(xs, ys)))

        return group_coords
        
    # calaculate the threshold
    threshold = cal_multiwind_threshold(img, 20, 5)

    # if img[u, v] < threshold: 0, else: img[u, v]
    _, nimg = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    # rough group star using connectivity
    group_coords = group_star(nimg, 2)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for coords in group_coords:
        x_sum = 0
        y_sum = 0
        gray_sum = 0
        for x, y in coords:
            x_sum += x * (img[x][y] - threshold)
            y_sum += y * (img[x][y] - threshold)
            gray_sum += img[x][y] - threshold
        centroids.append((x_sum / gray_sum, y_sum / gray_sum))

    return centroids

def remove_white_noise(img: np.ndarray):
    """
        Remove the white noise from the image.
    Args:
        img: the image to be processed
    Returns:
        img: the processed image
    """
    # get the image size
    l, w = img.shape

    # get the image mean
    img_mean = np.mean(img)

    # get the image standard deviation
    img_std = np.std(img)

    # get the image threshold
    img_threshold = img_mean + 3 * img_std

    # remove the white noise
    for i in range(l):
        for j in range(w):
            if img[i][j] > img_threshold:
                img[i][j] = 0

    return img


if __name__ == '__main__':
    # read the image
    img = cv2.imread('test2.png', 0)

    # get the centroids
    get_centroids(img)