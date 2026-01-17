import cv2
import numpy as np

from denoise import denoise_image
from detect import group_star, cal_threshold


def cal_center_of_guassian_curve(img: np.ndarray, rows, cols) -> tuple[float, float]:
    '''
        Calculate the centroid of the star using the gaussian fitting.
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


def cal_center_of_gravity(img: np.ndarray, rows: np.ndarray, cols: np.ndarray, method: str, T: float=0, A: float=200, sigma: float=1.0) -> tuple[float, float]:
    '''
        Calculate the centroid of the star in the window.
    '''
    
    y, x = rows, cols
    g = img[y, x]
    y, x = y + 0.5, x + 0.5

    if method == 'CoG' or method == 'CCoG':
        xgs, ygs = np.sum(x * g), np.sum(y * g)
        gs = np.sum(g)
    elif method == 'MCoG':
        xgs, ygs = np.sum(x * (g - T)), np.sum(y * (g - T))
        gs = np.sum(g - T)
    elif method == 'WCoG':
        i = np.argmax(g)
        x0, y0 = x[i], y[i]
        d2 = (x - x0)**2 + (y - y0)**2
        w = A*np.exp(-d2 / (2 * sigma**2))
        xgs, ygs = np.sum(x * g * w), np.sum(y * g * w)
        gs = np.sum(g * w)
    else:
        print('Invalid gravity method!')
        return 0.0, 0.0
    
    gs = np.maximum(gs, 1e-10)
    return ygs/gs, xgs/gs, gs


def get_star_centroids(img: np.ndarray, den_meth: str, seg_meth: list['str'], cen_meth: str, pixel_limit: int=5, connectivity=4, need_gray: bool=False) -> np.ndarray:
    '''
        Get the centroids of the stars in the image.
    '''

    # preprocess with denoising method
    filtered_img = denoise_image(img, den_meth)
    
    # rough group star using connectivity
    group_coords = group_star(filtered_img, seg_meth, connectivity=connectivity, pixel_limit=pixel_limit)

    # calculate the centroid coordinate with threshold and weight
    T = cal_threshold(filtered_img, seg_meth[1])
    centroids = np.array([cal_center_of_gravity(filtered_img, rows, cols, cen_meth, T) for rows, cols in group_coords])

    # sort by star luminosity
    centroids = centroids[np.argsort(centroids[:, 2])][::-1]
    if not need_gray:
        centroids = centroids[:, :2]

    return centroids
