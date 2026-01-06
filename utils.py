import os
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import rank_filter, maximum_filter, gaussian_filter, correlate
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def gen_combos(n: int, k: int):
    '''
        Generate C(n, k).
    '''
    combos = np.array(list(combinations(range(n), k)))
    return combos


def gen_gaussian_kernel(sigma: float, size: int, x: np.ndarray=None, order: int=2, normalize: bool=True):
    '''
        Generate a gaussian kernel.
    '''
    assert size % 2 == 1 and (order in (1, 2) or (x is not None and x.ndim in (1, 2)))

    r = size // 2
    if x is None:
        x = np.arange(-r, r + 1)
        if order == 1:
            xx = x ** 2
        else:  # order == 2
            x1, x2 = np.meshgrid(x, x)
            xx = x1 ** 2 + x2 ** 2
    else:
        xx = x ** 2
        
    kernel = np.exp(- xx / (2 * sigma **2))
    if normalize:
        kernel /= kernel.sum()

    return kernel


def cal_derivative(img: np.ndarray, order: tuple[int, int], sigma: float):
    '''
        Calculate derivative of an image with gaussian filter.
    '''
    assert img.ndim == 2 or img.ndim == 3

    # change image data type to avoid overflow
    fimg = img.astype(np.float64)

    return gaussian_filter(fimg, sigma=sigma, order=order, axes=(-2, -1))


def cal_difference(img: np.ndarray, dir: int):
    '''
        Calculate directional finite difference of neighboring pixels.
    '''
    # change image data type to avoid overflow
    fimg = img.astype(np.float64)

    # calculate neighboring differences
    dir %= 8
    if dir == 0:      # ↓
        kernel = np.array([[-1],
                           [ 1]])
    elif dir == 1:    # ↘
        kernel = np.array([[-1,  0],
                           [ 0,  1]])
    elif dir == 2:    # →
        kernel = np.array([[-1, 1]])
    elif dir == 3:    # ↗
        kernel = np.array([[ 0,  1],
                           [-1,  0]])
    elif dir == 4:    # ↑
        kernel = np.array([[ 1],
                           [-1]])
    elif dir == 5:    # ↖
        kernel = np.array([[ 1,  0],
                           [ 0, -1]])
    elif dir == 6:    # ←
        kernel = np.array([[1, -1]])
    elif dir == 7:    # ↙
        kernel = np.array([[ 0, -1],
                           [ 1,  0]])

    diff = correlate(fimg, kernel, mode='constant')
    return diff


def cal_doh(img: np.ndarray, sigma: float):
    '''
        Calculate determination of hessian.
    '''
    assert img.ndim == 2 or img.ndim == 3

    dxx = cal_derivative(img, order=(0, 2), sigma=sigma)
    dyy = cal_derivative(img, order=(2, 0), sigma=sigma)
    dxy = cal_derivative(img, order=(1, 1), sigma=sigma)

    return dxx * dyy - dxy**2


def cal_log(img: np.ndarray, sigma: float):
    '''
        Calculate laplacian of gaussian.
    '''
    assert img.ndim == 2 or img.ndim == 3

    dxx = cal_derivative(img, order=(0, 2), sigma=sigma)
    dyy = cal_derivative(img, order=(2, 0), sigma=sigma)

    return dxx + dyy


def cal_dog(img: np.ndarray, sigma1: float, sigma2: float):
    '''
        Calculate difference of gaussian.
    '''
    assert img.ndim == 2 or img.ndim == 3

    if sigma1 > sigma2:
        sigma1, sigma2 = sigma2, sigma1

    img1 = gaussian_filter(img, sigma1)
    img2 = gaussian_filter(img, sigma2)

    return img2 - img1


def cal_ly(img: np.ndarray, sigma: float):
    '''
        Calculate the ly operator result.
    '''
    assert img.ndim == 2 or img.ndim == 3

    # construct gradient covariance matrix with first derivatives
    dx = cal_derivative(img, order=(0, 1), sigma=sigma)
    dy = cal_derivative(img, order=(1, 0), sigma=sigma)

    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy
    
    # sum up dx2, dy2, dxy of area by gaussian filter
    # gradient covariance matrix = np.array([[adx2, adxy], [adxy, ady2]])
    adx2 = gaussian_filter(dx2, sigma=sigma, axes=(-2, -1))
    ady2 = gaussian_filter(dy2, sigma=sigma, axes=(-2, -1))
    adxy = gaussian_filter(dxy, sigma=sigma, axes=(-2, -1))

    # compute trace and determinant of the 2x2 structure tensor at every pixel
    tr = adx2 + ady2
    det = adx2 * ady2 - adxy * adxy

    # avoid division by zero
    eps = np.finfo(np.float64).eps  # ~2.2e-16

    # compute LY features
    q = 4.0 * det / (tr * tr + eps) # anisotropy measure
    w = det / (tr + eps)            # strength of the local structure

    return q, w


def cal_sobel(img: np.ndarray, sigma: float):
    '''
        Calculate the sobel operator result.
    '''
    assert img.ndim == 2 or img.ndim == 3
    
    dx = cal_derivative(img, order=(0, 1), sigma=sigma)
    dy = cal_derivative(img, order=(1, 0), sigma=sigma)

    return np.hypot(dx, dy)


def cal_angdist(points1: np.ndarray, points2: np.ndarray=None):
    '''
        Calculate the angular distance of the points.
    '''
    if points2 is None:
        points2 = points1

    assert points1.shape[1] == 3 and points2.shape[1] == 3
    
    norm1 = np.linalg.norm(points1, axis=1)
    norm2 = np.linalg.norm(points2, axis=1)
    angd = np.dot(points1, points2.T) / np.outer(norm1, norm2)

    return angd


def cal_mse_psnr_ssim(img1: np.ndarray, img2: np.ndarray, ndigits: int=-1):
    '''
        Calculate peak signal-to-noise ratio and the structural similarity between the original image and the filtered image.
    Args:
        img1: the original image
        img2: the image after filtering
        ndigits: Number of decimal places to round to. If -1 (default), keep full precision.
    Returns:
        mse, psnr, mssim
    '''
    assert img1.shape == img2.shape and img1.dtype == img2.dtype and (img1.dtype == np.uint8 or img1.dtype == np.float32)

    # max value
    mv = 255 if img1.dtype == np.uint8 else 1.0

    # caculate the mse
    mse = mean_squared_error(img1, img2)
    # print(mse, np.mean((img1 - img2)**2))
    
    # caculate the psnr
    psnr = peak_signal_noise_ratio(img1, img2, data_range=mv) if mse > 0 else np.inf
    # print(psnr, 10 * np.log10(mv**2 / mse) if mse > 0 else np.inf)
    
    # caculate the mean ssim
    mssim = structural_similarity(img1, img2, win_size=3, data_range=mv)

    # round to ndigits precision if needed
    if ndigits != -1:
        mse, psnr, mssim = np.round(mse, ndigits), np.round(psnr, ndigits), np.round(mssim, ndigits)

    return mse, psnr, mssim


def cal_rc_p_f1(tp: int | float | np.ndarray, fp : int | float | np.ndarray, fn: int | float | np.ndarray, percent: bool=True, ndigits: int=-1):
    '''
        Calculate recall, precision, and f1-score.
    Args:
        tp: true positives - the number of correct detections
        fp: false positives - the number of false alarms
        fn: false negatives - the number of missed targets
        percent: If True, return metrics in percent (e.g., 85.0 instead of 0.85).
        ndigits: Number of decimal places to round to. If -1 (default), keep full precision.
    Returns:
        rc, p, f1
    '''
    assert np.ndim(tp) == np.ndim(fp) == np.ndim(fn)

    # denominator constant for safe division
    eps = 1e-10

    # compute recall
    rc = tp / np.maximum(tp + fn, eps)

    # compute precision
    p = tp / np.maximum(tp + fp, eps)

    # compute F1-score
    f1 = 2 * (p * rc) / np.maximum(p + rc, eps)

    # convert into percentage if needed
    if percent:
        rc, p, f1 = rc * 100.0, p * 100.0, f1 * 100.0

    # round to ndigits precision if needed
    if ndigits != -1:
        rc, p, f1 = np.round(rc, ndigits), np.round(p, ndigits), np.round(f1, ndigits)

    return rc, p, f1


def find_overlap_and_unique(a: np.ndarray, b: np.ndarray, threshold: float=2, return_count_only: bool=False):
    '''
        Find the overlap and unique parts of two point sets.
    '''
    if a.size == 0 or b.size == 0:
        res = np.array([]), np.array([]), a, b
        if return_count_only:
            return tuple(map(len, res))
        else:
            return res

    assert a.shape[1] == 2 and b.shape[1] == 2

    # calculate the L2 distance between each points in both a and b
    dist = np.sqrt(np.sum((a[:, None] - b[None, :])**2, axis=2)) # (m, n)

    # find the closest point in b for each point in a
    min_idx = np.argmin(dist, axis=1)                           # min_idx: shape (m,), with values in [0, n) 
    
    # only if distance is smaller than eps, the match is valid
    mask = np.min(dist, axis=1) < threshold
    overlap_a, overlap_b = a[mask], b[min_idx][mask]            # through b[min_idx][mask], each point in 'a' is matched to at most one nearest point in 'b' (via argmin), avoiding duplicate matches.
    unique_a = a[~mask]
    
    # only if distance is smaller than threshold, the match is valid
    mask = np.min(dist, axis=0) < threshold
    unique_b = b[~mask]
    
    res = overlap_a, overlap_b, unique_a, unique_b
    if return_count_only:
        return tuple(map(len, res))
    else:
        return res


def find_close_pair(coords: np.ndarray, threshold: int=7, method: str='L1'):
    '''
        Find the close pair of coordinates whose distance is less than the threshold.
    '''
    assert(coords.shape[1] == 2)

    n, _ = coords.shape

    if method == 'L1':
        diff = np.abs(coords[:, np.newaxis, :] - coords[np.newaxis, :, :])
        dist = np.sum(diff, axis=2)            
    else: #method == 'L2':
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

    # triangle upper mask
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    # indexs of close pair
    i, j = np.where((dist < threshold) & mask)

    if len(i) > 0:
        return coords[i[0]], coords[j[0]]
    
    return None


def are_collinear(a: np.ndarray, b: np.ndarray, eps: float=1e-5):
    '''
        Determine whether vectors are collinear.
    '''
    assert a.shape == (3,) and b.shape == (3,)

    if np.allclose(a, 0) or np.allclose(b, 0):
        return True
    
    return np.linalg.norm(np.cross(a, b)) < eps


def is_local_topk(img: np.ndarray, k: int, connectivity: int=4, footprint: np.ndarray=None):
    '''
        Determine whether each pixel in the image is among the top-k largest values within its local neighborhood.
    '''
    if footprint is None:
        if connectivity == 4:
            footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        else:  # connectivity == 8
            footprint = np.ones((3, 3), dtype=bool)

    local_topk = rank_filter(img, rank=k, footprint=footprint, mode='constant', cval=-np.inf)

    return img >= local_topk


def is_near_local_max(img: np.ndarray, connectivity: int=4, footprint: np.ndarray=None):
    '''
        Determine whether each pixel in the image is close to the maximum values in its local neighborhood.
    '''

    if footprint is None:
        if connectivity == 4:
            footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        else:  # connectivity == 8
            footprint = np.ones((3, 3), dtype=bool)

    max_map = maximum_filter(img, footprint=footprint, mode='constant', cval=-np.inf)
    
    mean = np.mean(img)
    std = np.std(img)
    gap = min(5, 0.5 * mean, 0.5 * std)

    return max_map - img < gap


def con_orthogonal_basis(a: np.ndarray, b: np.ndarray):
    '''
        Construct orthogonal basis for vector a and b.
    '''
    assert a.shape == (3,) and b.shape == (3,)
    assert np.any(a != 0) and np.any(b != 0)
    
    x = a/np.linalg.norm(a)
    y = np.cross(a, b)/np.linalg.norm(np.cross(a, b))
    z = np.cross(x, y)/np.linalg.norm(np.cross(x, y))

    m = np.vstack([x, y, z]).T
    assert np.allclose(m @ m.T, np.identity(3), atol=1e-2)

    return m
    

def traid(v: np.ndarray, w: np.ndarray, i: int=0, j: int=1):
    '''
        Get the three-dimensional attitude matrix of the star sensor using Triad algorithm. Each column in v is a unit vector representing the direction of a star as measured by the star sensor, while column vector in w is expressed in celestial coordinate system. 
    Args:
        v: view vectors(3, n)
        w: reference vectors(3, n)
    Returns:
        r: the rotation matrix(v = r @ w)
    '''
    assert v.shape[0] == 3 and w.shape[0] == 3 and v.shape[1] == w.shape[1]
    assert not np.any(np.isnan(v)) and not np.any(np.isnan(w))
    assert not are_collinear(v[:, i], v[:, j]) and not are_collinear(w[:, i], w[:, j]) 

    vm = con_orthogonal_basis(v[:, i], v[:, j])
    wm = con_orthogonal_basis(w[:, i], w[:, j])
    r = vm @ np.linalg.inv(wm) 

    # ? why assertion always fail
    # for i, j in result:
    #     vm = con_orthogonal_basis(v[:, i], v[:, j])
    #     wm = con_orthogonal_basis(w[:, i], w[:, j])
    #     assert np.allclose(vm, r @ wm, atol=1e-1), f'{i}, {j}, {vm}, {r @ wm}'

    return r


def quest(v: np.ndarray, w: np.ndarray, weights: np.ndarray=None):
    '''
        Get the three-dimensional attitude matrix of the star sensor using Quest algorithm. Each column in v is a unit vector representing the direction of a star as measured by the star sensor, while column vector in w is expressed in celestial coordinate system. 
    Args:
        v: view vectors(3, n)
        w: reference vectors(3, n)
        weights: vector weights
    Returns:
        r: the rotation matrix(v = r @ w)
    '''
    assert v.shape[0] == 3 and w.shape[0] == 3 and v.shape[1] == w.shape[1]
    assert not np.any(np.isnan(v)) and not np.any(np.isnan(w))
    assert np.allclose(np.linalg.norm(v, axis=0), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(w, axis=0), 1, atol=1e-6)

    #TODO: fix
    n = v.shape[1]
    if weights is None:
        weights = np.ones(n)/n
    else:
        weights = weights/np.sum(weights)

    S = w @ (weights * v).T
    
    sigma = np.trace(S)
    z = np.array([S[1,2]-S[2,1], S[2,0]-S[0,2], S[0,1]-S[1,0]])
    
    K = np.zeros((4,4))
    K[:3,:3] = S + S.T - sigma*np.eye(3)
    K[:3,3] = z
    K[3,:3] = z
    K[3,3] = sigma
    
    eigenvals, eigenvecs = np.linalg.eig(K)
    max_idx = np.argmax(eigenvals.real)
    q = eigenvecs[:, max_idx].real
    
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    
    r = np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])
    
    u, _, vh = np.linalg.svd(r)
    return u @ vh


def plot_gray_3d(img: np.ndarray, method: str='plot_surface', color_map: str='gray', label_text: bool=False):
    '''
        Plot the gray image in 3 dimension.
    '''
    # get the image size
    h, w = img.shape

    # generate the coordinates
    x, y = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
    
    x, y = np.meshgrid(x, y)
    z = img
    
    # create 3d axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot surface/bar3d
    if method == 'plot_surface':
        ax.plot_surface(
            x, y, z, 
            cmap=color_map, 
        )
    else: # method == 'bar3d'
        x, y = x.flatten(), y.flatten()
        dx = dy = 0.9       # bar width
        dz = z.flatten()    # bar height
        ax.bar3d(
            x, y, 0, 
            dx, dy, dz, 
            color='gray',
            linewidth=0.3,
            alpha=0.9
        )

    # label text
    if label_text:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        ax.set_xlabel('X/像素', labelpad=5)
        ax.set_ylabel('Y/像素', labelpad=5)
        # ax.set_zlabel('Z')
        ax.set_zlabel('灰度', rotation=90)
    
    # image title
    # ax.set_title(title)
    plt.show()


def plot_grad_field(img: np.ndarray, sigma: float, scale: float=1):
    '''
        Plot the gradient vector field.
    '''
    eps = 1e-10
    h, w = img.shape

    # compute gradients
    grad_y, grad_x = cal_derivative(img, order=(1, 0), sigma=sigma), cal_derivative(img, order=(0, 1), sigma=sigma)
    grad = np.hypot(grad_y, grad_x)
    max_grad = np.max(grad, initial=eps)
    grad_y, grad_x = grad_y / max_grad, grad_x / max_grad
    
    # create grid
    y, x = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing='ij')

    # plot
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.quiver(x, y, grad_x, grad_y, color='r', angles='xy', scale_units='xy', scale=scale)
    # plt.title("Gradient Vector Field")
    plt.axis('off')
    plt.show()


def plot_freq_spectrum(img: np.ndarray):
    '''
        Plot the frequency spectrum of the image.
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)    
    fdb_img = 20 * np.log(np.abs(fshift))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fdb_img, cmap='gray')
    plt.title('Frequency Spectrum')
    plt.axis('off')
    plt.show()


def label_star_image(img: np.ndarray, coords: np.ndarray, ids: np.ndarray=None, circle: bool=False, auto_label: bool=False, axis_on: bool=True, show: bool=True, output_path: str=None):
    '''
        Label the stars in the image with id or circle.
    '''
    h, w = img.shape[:2]

    _, ax = plt.subplots(figsize=(10, 10))
    if axis_on:
        #!NOTE: when axes are on, invert y-axis so that image origin aligns with array indexing (row 0 at bottom); origin='lower' + invert together place data correctly while keeping axis labels in row-coordinates
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, origin='lower')  
        ax.axis('on')
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()
    else:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, origin='upper')
        ax.axis('off')

    if np.all(ids==None):
        ids = np.arange(len(coords))+1 if auto_label else np.full(len(coords), -1)

    for id, (row, col) in zip(ids, coords):
        row, col = int(row), int(col)
        if circle:
            ax.plot(col, row, 'o', mec='blue', mfc='none', ms=10, mew=2)
        if id != -1:
            row, col = min(row+10, h-20), min(col-20, w-20)
            ax.text(col, row, str(id), fontsize=10, color='white', ha='left', va='top')

    if output_path: #!NOTE: must save before show, otherwise the figure will be cleared by plt.show() and saved as blank/white image
        dir = os.path.abspath(os.path.dirname(output_path))
        os.makedirs(dir, exist_ok=True)
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    plt.close()


def label_detect_result(img: np.ndarray, real_coords: np.ndarray, esti_coords: np.ndarray, dist_threshold: float, axis_on: bool=False, show: bool=True, output_path: str=None):
    '''
        Label the detect result with different shapes and colors.
    '''
    
    h, w = img.shape
    _, matched_coords, missed_coords, false_coords = find_overlap_and_unique(real_coords, esti_coords, dist_threshold)
    detect_res = np.vstack([
        np.hstack([coords, np.full((coords.shape[0], 1), i, dtype=int)])
        for i, coords in enumerate([real_coords, matched_coords, false_coords])
    ])

    _, ax = plt.subplots(figsize=(10, 10))
    if axis_on:
        #!NOTE: when axes are on, invert y-axis so that image origin aligns with array indexing (row 0 at bottom); origin='lower' + invert together place data correctly while keeping axis labels in row-coordinates
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, origin='lower')  
        ax.axis('on')
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()
    else:
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, origin='upper')
        ax.axis('off')

    legend_elements = [
        Line2D([0], [0], marker='+', linestyle='None', mec='red', mfc='red', ms=10, mew=2, label='真实星点'),
        Line2D([0], [0], marker='^', linestyle='None', mec='yellow', mfc='none', ms=10, mew=2, label='正确检测'),
        Line2D([0], [0], marker='o', linestyle='None', mec='blue', mfc='none', ms=10, mew=2, label='错误检测')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.75, 1), loc='upper left', prop={'family': 'SimHei', 'size': 12})

    for row, col, label in detect_res:
        if label == 0:
            ax.plot(col, row, 'r+', ms=10, mew=2)                           # red cross
        elif label == 1:
            ax.plot(col, row, '^', mec='yellow', mfc='none', ms=10, mew=2)  # yellow triangle
        else: # label == 2
            ax.plot(col, row, 'o', mec='blue', mfc='none', ms=10, mew=2)    # blue circle

    if output_path:
        dir = os.path.abspath(os.path.dirname(output_path))
        os.makedirs(dir, exist_ok=True)
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    plt.close()

    print(
        'Total Number of Stars:', len(real_coords),
        '\nNumber of Matched Stars:', len(matched_coords),
        '\nNumber of Miss Stars:', len(missed_coords),
        '\nNumber of False Stars:', len(false_coords),
        '\nMiss:\n', 
        np.round(missed_coords, 2),
        '\nFalse:\n', 
        np.round(false_coords, 2)
    )


def describe_database(db: pd.DataFrame):
    '''
        Describe the database.
    '''

    db.columns = db.columns.astype(int)
    db_info = np.sum(db.notna().to_numpy(), axis=1)
    max_cnt, min_cnt, avg_cnt = np.max(db_info), np.min(db_info), np.sum(db_info)/len(db)

    print(
        'Max count of 1 in pattern matrix', max_cnt, 
        '\nMin count of 1 in pattern matrix', min_cnt, 
        '\nAvg count of 1 in pattern matrix', avg_cnt
    )

    plt.hist(db_info, bins=max_cnt, edgecolor='black')
    plt.show()
