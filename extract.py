import cv2
import numpy as np

from denoise import denoise_with_nlm, denoise_with_star_distri
from detect import cal_threshold, group_star


def cal_wind_boundary(center: tuple[int, int], wind_size: int, h: int, w: int) -> tuple[int, int, int, int]:
    '''
        Calculate the boundary of the window.
    Args:
        center: the center of the window
        wind_size: the size of the window
    Returns:
        t: the top boundary of the window
        b: the bottom boundary of the window
        l: the left boundary of the window
        r: the right boundary of the window
    '''
    # construct window
    t = max(0, center[0] - wind_size//2)
    b = min(h-1, center[0] + wind_size//2)
    l = max(0, center[1] - wind_size//2)
    r = min(w-1, center[1] + wind_size//2)

    return t, b, l, r


def cal_center_of_guassian_curve(img: np.ndarray, rows, cols) -> tuple[float, float]:
    '''
        Calculate the centroid of the star using the gaussian fitting.
    Args:
        img: the image to be processed
    Returns:
        centroid: the centroid of the star
    '''
    x, y = rows+0.5, cols+0.5

    # construct the matrix A
    A = np.column_stack([
        x**2 + y**2,
        x,
        y,
        -np.ones_like(x)
    ]).astype(np.float64)

    if np.linalg.cond(A) > 1e12:
        print('A is ill-conditioned')
        return 0.0, 0.0 

    # ln(I(x, y))
    Y = np.log(img[rows, cols].flatten(order='F')+1e-11).astype(np.float64)

    # solve the linear equation X = ||Y - AX||min
    X, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return round(-X[1]/(2*X[0]), 3), round(-X[2]/(2*X[0]), 3)


def cal_center_of_gravity(img: np.ndarray, rows: np.ndarray, cols: np.ndarray, method: str, T: int=-1, center: tuple[int, int]=None, A: float=200, sigma: float=1.0) -> tuple[float, float]:
    '''
        Calculate the centroid of the star in the window.
    Args:
        img: the image to be processed
        rows/cols: the coordinates of the pixels in the window
        method: centroid algorithm
            'CoG': center of gravity
            'MCoG': modified center of gravity
            'WCoG': weighed center of gravity
            # 'IWCoG': iterative weighed center of gravity
        T: the threshold of the star
        center: initial centroid used in WCoG
        sigma: the sigma used in WCoG and IWCoG
    Returns:
        centroid: the centroid of the star
    '''
    
    # gray
    g = img[rows, cols]
    # row and column add 0.5 to get the center of the pixel
    x, y = rows+0.5, cols+0.5

    if method == 'CoG':
        # row multiply gray and sum, column multiply gray and sum
        xgs, ygs = np.sum(x * g), np.sum(y * g)
        gs = np.sum(g)
    elif method == 'MCoG':
        xgs, xgs = np.sum(x * (g - T)), np.sum(y * (g - T))
        gs = np.sum(g - T)
    elif method == 'WCoG':
        # weight for each pixel used in WCoG and IWCoG
        d = (x-center[0])**2 + (y-center[1])**2
        w = A*np.exp(-d/(2*sigma**2))
        xgs, ygs = np.sum(x * g * w), np.sum(y * g * w)
        gs = np.sum(g * w)
        if gs == 0.0:
            return 0.0, 0.0
    else:
        print('wrong gravity method!')
        return 0.0, 0.0
    
    center = round(xgs/gs, 3), round(ygs/gs, 3)
    return center


def get_star_centroids(img: np.ndarray, thr_method: str, cen_method: str, wind_size: int=-1) -> list[tuple[float, float]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
        thr_method: threshold calculation method
        cen_method: centroid algorithm
        wind_size: the size of the window used to calculate the centroid
    Returns:
        centroids: the centroids of the stars in the image
    '''

    # get the image size
    h, w = img.shape

    # denoise
    filtered_img = cv2.addWeighted(denoise_with_nlm(img, 10, 7, 49), 0.5, denoise_with_nlm(img, 10, 5, 25), 0.5, 0)
    filtered_img = denoise_with_star_distri(filtered_img, half_size=3)
    
    # calaculate the threshold
    T =  cal_threshold(filtered_img, thr_method)

    # rough group star using connectivity
    group_coords = group_star(filtered_img, T, connectivity=4, pixel_limit=7)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for (rows, cols) in group_coords:
        vals = filtered_img[rows, cols]
        # get the brightest pixel and use it as the center
        idx = np.argmax(vals)
        brightest = rows[idx], cols[idx]

        if wind_size != -1:    
            # construct window
            t, b, l, r = cal_wind_boundary(brightest, wind_size, h, w)
            nrows, ncols = np.meshgrid(np.arange(t, b+1), np.arange(l, r+1))
            nrows, ncols = nrows.flatten(), ncols.flatten()
            centroid = cal_center_of_gravity(filtered_img, nrows, ncols, cen_method, T, center=brightest, n=10)
        else:
            centroid = cal_center_of_gravity(filtered_img, rows, cols, cen_method, T, center=brightest, n=10)

        centroids.append(centroid)

    return centroids
