import os
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import rank_filter, gaussian_filter, correlate
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def gen_combos(n: int, k: int):
    '''
        Generate C(n, k).
    '''
    combos = np.array(list(combinations(range(n), k)))
    return combos


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
        Calculate the ly operator result.
    '''
    assert img.ndim == 2 or img.ndim == 3
    
    dx = cal_derivative(img, order=(0, 1), sigma=1)
    dy = cal_derivative(img, order=(1, 0), sigma=1)

    return np.hypot(dx, dy)


def find_overlap_and_unique(A: np.ndarray, B: np.ndarray, eps: float=2):
    '''
        Find the overlap parts of two point sets.
    '''
    if A.size == 0:
        return np.array([]), np.array([]), np.array([]), B
    if B.size == 0:
        return np.array([]), np.array([]), A, np.array([])

    assert A.shape[1] == 2 and B.shape[1] == 2

    # calculate the L2 distance between each points in both A and B
    dist = np.sqrt(np.sum((A[:, None] - B[None, :])**2, axis=2)) # (m, n)

    # find the closest point in B for each point in A
    min_idx = np.argmin(dist, axis=1)
    
    # only if distance is smaller than eps, the match is valid
    mask = np.min(dist, axis=1) < eps
    overlap_A, overlap_B = A[mask], B[min_idx][mask]    
    unique_A = A[~mask]
    
    # only if distance is smaller than eps, the match is valid
    mask = np.min(dist, axis=0) < eps
    unique_B = B[~mask]

    # # struct numpy array
    # BS = B.view([('x', B.dtype), ('y', B.dtype)]).reshape(-1)
    # OBS = overlap_B.view([('x', overlap_B.dtype), ('y', overlap_B.dtype)]).reshape(-1)
    # unique_B = np.setdiff1d(BS, OBS).view(A.dtype).reshape(-1, 2)
    
    return overlap_A, overlap_B, unique_A, unique_B


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


def is_local_topk(img: np.ndarray, mask: np.ndarray, k: int, connectivity: int=4):
    '''
        Determine whether each masked location is among the top-k largest values in its neighborhood.
    '''
    if connectivity == 4:
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    else:  # connectivity == 8
        footprint = np.ones((3, 3), dtype=bool)

    local_topk = rank_filter(img, rank=k, footprint=footprint, mode='constant', cval=-np.inf)

    return mask & (img >= local_topk)


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

    #! Something wrong, need to fix
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


def get_angdist(points1: np.ndarray, points2: np.ndarray=None):
    '''
        Get the angular distance of the points.
    '''
    if points2 is None:
        points2 = points1

    assert points1.shape[1] == 3 and points2.shape[1] == 3
    
    norm1 = np.linalg.norm(points1, axis=1)
    norm2 = np.linalg.norm(points2, axis=1)
    angd = np.dot(points1, points2.T) / np.outer(norm1, norm2)

    return angd


def convert_rade2deg(ra: float, dec: float):
    '''
        Convert the RA and DE from degree to timezone.
    '''
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    return coord.to_string('hmsdms')


def draw_gray_3d(img: np.ndarray, method: str='plot_surface', color_map: str='gray'):
    '''
        Draw the gray image in 3 dimension.
    Args:
        img: the image to be processed
    '''
    # get the image size
    h, w = img.shape

    # generate the coordinates
    x, y = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
    
    x, y = np.meshgrid(x, y)
    z = img
    
    # create 3D image
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if method == 'plot_surface':
        ax.plot_surface(
            x, y, z, 
            cmap=color_map, 
        )
    elif method == 'bar3d':
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
    else:
        for i in range(h):
            z_line = z[i, :]
            ax.plot(x[i,:], [i] * len(x[i,:]), z_line, linewidth=1.5, alpha=0.7,
                    color=plt.cm.viridis(z_line.max() / np.max(np.abs(img))))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (gray value)')
    # ax.set_title(title)
    plt.show()


def draw_freq_spectrum(img: np.ndarray):
    '''
        Draw the frequency spectrum of the image.
    Args:
        img: the image to be processed
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


def label_star_image(img: np.ndarray, coords: np.ndarray, ids: np.ndarray=None, circle: bool=False, auto_label: bool=False, axis_on: bool=True, grid_on: bool=False, grid_step: int=10, show: bool=True, output_path: str=None):
    '''
        Label the stars in the image with id or circle.
    '''
    h, w = img.shape[:2]

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray', origin='lower')  

    if axis_on:
        ax.axis('on')
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
    else:
        ax.axis('off')
    ax.invert_yaxis()

    if grid_on:
        ax.set_xticks(np.arange(0, w, grid_step))
        ax.set_yticks(np.arange(0, h, grid_step))
        ax.grid(grid_on, color='r', linewidth=2)

    if np.all(ids==None):
        ids = np.arange(len(coords))+1 if auto_label else np.full(len(coords), -1)

    for id, (row, col) in zip(ids, coords):
        row, col = int(row), int(col)
        if circle:
            circle = Circle((col, row), 10, edgecolor='b', facecolor='none')
            ax.add_patch(circle)
        if id != -1:
            row, col = min(row+10, h-20), min(col-20, w-20)
            ax.text(col, row, str(id), fontsize=10, color='white', ha='left', va='top')

    if show:
        plt.show()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=1, dpi=300)
    
    plt.close()


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


def cal_snr(img: np.ndarray, noised_img: np.ndarray):
    '''
        Calculate the signal-to-noise ratio between the original image and the noised image.
    Args:
        img: the original image
        noised_img: the noised image
    Returns:
        snr: the signal-to-noise ratio
    '''
    snr = 10 * np.log10(np.sum(img**2) / np.sum((img - noised_img)**2))

    return snr


def cal_mse_psnr_ssim(img1: np.ndarray, img2: np.ndarray):
    '''
        Calculate peak signal-to-noise ratio and the structural similarity between the original image and the filtered image.
    Args:
        img1: the original image
        img2: the image after filtering
    Returns:
        mse: the mean sqaure error
        psnr: the peak signal-to-noise ratio
        mssim: the mean structural similarity
    '''
    assert(img1.dtype == img2.dtype and (img1.dtype == np.uint8 or img1.dtype == np.float32))

    # max value
    mv = 255 if img1.dtype == np.uint8 else 1.0

    # caculate the MSE
    mse = mean_squared_error(img1, img2)
    # print(mse, np.mean((img1 - img2)**2))
    
    # caculate the PSNR
    psnr = peak_signal_noise_ratio(img1, img2, data_range=mv) if mse > 0 else np.inf
    # print(psnr, 10 * np.log10(mv**2 / mse) if mse > 0 else np.inf)
    
    # caculate the SSIM
    mssim = structural_similarity(img1, img2, win_size=3, data_range=mv)

    mse, psnr, mssim = round(mse, 2), round(psnr, 2), round(mssim, 2)

    return mse, psnr, mssim