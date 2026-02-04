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


def cal_center_of_gravity(img: np.ndarray, rows: np.ndarray, cols: np.ndarray, cen_meth: str, thr_meth: str='Liebe3', T: float=None, A: float=200, size: int=15, sigma: float=1.0) -> tuple[float, float]:
    '''
        Calculate the centroid of the star in the window.
    '''

    h, w = img.shape
    r = size // 2

    g = img[rows, cols]
    y, x = rows, cols
    x0, y0 = x[np.argmax(g)], y[np.argmax(g)]
    if T is None:
        y1, y2 = max(0, y0 - r), min(h - 1, y0 + r + 1)
        x1, x2 = max(0, x0 - r), min(w - 1, x0 + r + 1)
        T = cal_threshold(img[y1:y2, x1:x2], thr_meth)

    if cen_meth == 'CoG' or cen_meth == 'CCoG':
        xgs, ygs = np.sum(x * g), np.sum(y * g)
        gs = np.sum(g)
    elif cen_meth == 'MCoG':
        xgs, ygs = np.sum(x * (g - T)), np.sum(y * (g - T))
        gs = np.sum(g - T)
        print(xgs, ygs, gs)
    elif cen_meth == 'WCoG':
        d2 = (x - x0)**2 + (y - y0)**2
        wt = A*np.exp(-d2 / (2 * sigma**2))
        xgs, ygs = np.sum(x * g * wt), np.sum(y * g * wt)
        gs = np.sum(g * wt)
    else:
        print('Invalid gravity cen_meth!')
        return 0.0, 0.0
    
    gs = np.maximum(gs, 1e-10)
    return ygs / gs + 0.5, xgs / gs + 0.5, gs


def get_star_centroids(img: np.ndarray, den_meth: str, seg_meth: list['str'], cen_meth: str, size: int | list[int], pixel_limit: int=5, connectivity=4, gray: bool=False, output_dir: str=None) -> np.ndarray:
    '''
        Get the centroids of the stars in the image.
    '''

    # preprocess with denoising method
    filtered_img = denoise_image(img, den_meth)
    
    # rough group star using connectivity
    group_coords = group_star(filtered_img, seg_meth, size, connectivity=connectivity, pixel_limit=pixel_limit, output_dir=output_dir)

    # calculate the centroid coordinate with threshold and weight
    centroids = np.array([cal_center_of_gravity(filtered_img, rows, cols, cen_meth) for rows, cols in group_coords]) if len(group_coords) > 0 else np.zeros((0, 3))

    # sort by star luminosity
    centroids = centroids[np.argsort(centroids[:, 2])][::-1]
    if not gray:
        centroids = centroids[:, :2]

    return centroids
