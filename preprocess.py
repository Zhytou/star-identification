import numpy as np
import cv2

def get_centroids(img: np.ndarray):
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
    '''
    # get the image size
    l, w = img.shape

    # multiple window threshold division https://ieeexplore.ieee.org/abstract/document/1008988
    wind_len = 20
    num_wind = 5 

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

    # threadhold = background_mean + std * 5
    threshold = tot_mean + std * 5

    # get the centroids


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
    img = cv2.imread('test.png', 0)

    # get the centroids
    get_centroids(img)