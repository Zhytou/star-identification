import cv2
import numpy as np
import pywt
import scipy.ndimage as nd

from collections import defaultdict
from utils import get_neighbors


def basic_filter(img: np.ndarray, method: str='GAUSSIAN', size: int=3) -> np.ndarray:
    '''
        Conventional noise reducing filters.
    Args:
        img: the image to be processed
        method: the method of filtering
    Returns:
        filtered_img: the image after filtering
    '''
    if method == 'GAUSSIAN':
        filtered_img = cv2.GaussianBlur(img, (size, size), 0.7)
    elif method == 'MEAN':
        filtered_img = cv2.blur(img, (size, size))
    elif method == 'MEDIAN':
        # d = size//2
        # padded_img = np.pad(img, ((d, d), (d, d)), mode='constant')
        # filtered_img = cv2.medianBlur(padded_img, size)
        # filtered_img = filtered_img[d:-d, d:-d]
        filtered_img = cv2.medianBlur(img, size)
    elif method == 'BLF':
        filtered_img = cv2.bilateralFilter(img, 7, 20, 1.5)
    elif method == 'GLF':
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        kernel = cv2.getGaussianKernel(size, 0.7)
        kernel = np.outer(kernel, kernel.transpose())
        kernel_padded = np.pad(kernel, ((0, img.shape[0] - kernel.shape[0]), (0, img.shape[1] - kernel.shape[1])), mode='constant')
        kernel_f = np.fft.fft2(kernel_padded)
        kernel_fshift = np.fft.fftshift(kernel_f)

        filtered_fshift = fshift * kernel_fshift
        filtered_f = np.fft.ifftshift(filtered_fshift)
        filtered_img = np.fft.ifft2(filtered_f)
        filtered_img = np.abs(filtered_img)
    else:
        print('Invalid filter method!')
        return None
    
    return filtered_img


def morph_filter(img: np.ndarray, method: str='max', se=cv2.MORPH_RECT, size: int=3) -> np.ndarray:
    '''
        Morphology filter.
    '''
    if method == 'max':
        kernel = cv2.getStructuringElement(se, (size, size))
        filtered_img = cv2.dilate(img, kernel)
    elif method == 'min':
        kernel = cv2.getStructuringElement(se, (size, size))
        filtered_img = cv2.erode(img, kernel)
    elif method == 'open':
        kernel = cv2.getStructuringElement(se, (size, size))
        filtered_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif method == 'close':
        kernel = cv2.getStructuringElement(se, (size, size))
        filtered_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        print('Invalid morph method')
        return None
    
    return filtered_img


def denoise_with_nlm(img: np.ndarray, patch_size: int=7, wind_size: int=21, sigma: int=10):
    '''
        Non-local means denoising.
    Args:
        img: the image to be processed
        patch_size: the size of the patch
        wind_size: the size of the search window
        sigma: the parameter regulating filter strength
    Returns:
        denoised_img: the image after filtering
    '''
    denoised_img = cv2.fastNlMeansDenoising(img, None, sigma, patch_size, wind_size)

    return denoised_img


def denoise_with_wavelet(img, wavelet: str, level: int=3, threshold: float=2):
    '''
        Denoise with wavelet transform.
    Args:
        img: the image to be processed
        wavelet: the name of wavelet
        level: the decomposition level
    Returns:
        denoised_img: the image after filtering
    '''
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    cA = coeffs[0]              # low freq coeff
    cD = coeffs[1:]             # high freq coeff
    denoised_coeffs = [cA]      # keep low freq coeff

    for cd in cD:
        denoised_cd = [pywt.threshold(band, threshold, mode='soft') for band in cd]
        denoised_coeffs.append(denoised_cd)
    
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    return denoised_image


def denoise_with_shearlet(img, wavelet: str, level: int=3, threshold: float=2):
    '''
        Denoise with shearlet transform.
    '''
    pass


def denoise_with_blf(img: np.ndarray, size: int, sigma_color: float, sigma_space: float, threshold: int=100):
    '''
        Improved bilateral filter denoising.
    Args:
        img: the image to be processed
        size: the size of template
        sigma_color: the standard deviation of the color space
        sigma_space: the standard deviation of the coordinate space
        threshold: the threshold for gray difference
    Returns:
        filtered_img: the image after filtering
    '''

    def custom_activation(x, threshold):
        '''
            Custom activation function.
        '''
        return np.where(x > threshold, np.inf, x)

    h, w = img.shape
    d = size
    if d % 2 == 0:
        d = d + 1
    r = d // 2

    # pad the image for color weight calculation and change the type in case negative overflow
    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant').astype(np.int16)

    # use a strided view to get the patches(h, w, d, d)
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))
    grays = img[..., None, None] # the central gray value of each image patch

    # calculate the color and space weights
    color_diffs = custom_activation(np.abs(grays - patches), threshold)
    color_weights = np.exp(-(color_diffs ** 2) / (2 * sigma_color ** 2))

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    space_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    # apply bilateral filter    
    bilateral_weights = color_weights * space_weights
    bilateral_weights = bilateral_weights / bilateral_weights.sum(axis=(-2, -1), keepdims=True)
    filtered_img1 = (bilateral_weights * patches).sum(axis=(-2, -1))

    # calculate weight sum of 8-connectivity neighbor
    neighbors = get_neighbors(8)
    weightsum = bilateral_weights[..., r+neighbors[:, 0], r+neighbors[:, 1]].sum(axis=-1)

    # apply mean filter
    mean_weights = np.zeros((d, d))
    mean_weights[r+neighbors[:, 0], r+neighbors[:, 1]] = 1
    mean_weights = mean_weights / mean_weights.sum()
    filtered_img2 = (mean_weights[None, None, ...] * patches).sum(axis=(-2, -1))

    denoised_img = np.where((np.abs(weightsum) > 0.1) | (img < threshold), filtered_img1, filtered_img2).astype(np.uint8)

    return denoised_img


def denoise_with_amf(img: np.ndarray, size1: int, size2: int):
    '''
        Denoise with adaptive median filter.
    '''

    # generate all the windows sizes
    sizes = np.arange(size1, size2, 2)  #(n, )

    # get median, min and max of img under different the window sizes
    mids = np.stack([basic_filter(img, 'MEDIAN', size=s) for s in sizes])   # (n, h, w)
    mins = np.stack([morph_filter(img, 'min') for s in sizes])              # (n, h, w)
    maxs = np.stack([morph_filter(img, 'max') for s in sizes])              # (n, h, w)
    
    # get valid median values
    mask1 = (mins < mids) & (mids < maxs)                                   # (n, h, w)
    
    # get possible noisy pixels
    mask2 = (mins >= img[None, ...]) | (maxs <= img[None, ...])             # (n, h, w)

    # set possible noisy pixels to valid median values
    mask = mask1 & mask2
    idxs = np.argmax(mask, axis=0)  # the first window size index whose median value is valid
    
    denoised_img = np.where(
        np.any(mask, axis=0),  # (h, w)
        mids[idxs],
        img        
    )
    
    return denoised_img


def denoise_with_cmg(img: np.ndarray, size: int=5, sigma: float=1):
    '''
        Denoise with combined morphology operation and modified gaussian filter.
        https://doi.org/10.27241/d.cnki.gnjgu.2024.001923
    '''

    Id = np.percentile(img, 99)
    T1 = 1.4
    T2 = 1.2
    T3 = 3
    Atten = 2

    d = size
    r = d // 2
    h, w = img.shape

    # 1. apply erosion operator
    denoised_img = morph_filter(img, 'min', se=cv2.MORPH_CROSS, size=size) 

    # 2. select non-star pixels
    offsets = get_neighbors(4)                                                  # 4-connectivity offsets (4, 2)
    coords = np.stack(np.meshgrid(np.arange(r, h+r), np.arange(r, w+r)), axis=-1)   # coordinates of the entire image pixels (h, w)
    coords = coords[..., None, :] + offsets[None, None, ...]                    # 4-connectivity neighborhoods (h, w, 4, 2)
    padded_img = np.pad(img, ((r, r), (r, r)), mode='edge')                     # padded zero as the border of image

    img = np.maximum(img, 1e-10)
    min_map = np.maximum(np.min(padded_img[coords[..., 0], coords[..., 1]], axis=-1), 1e-10)       # (h, w)
    max_map = np.maximum(np.max(padded_img[coords[..., 0], coords[..., 1]], axis=-1), 1e-10)       # (h, w)
    
    S1 = (img < Id) & (img / min_map > T1)                           # non star pixels
    S2 = (img > Id) & ((img / max_map < T2) | (max_map / img < T2)) & (max_map / min_map < T3) # star pixels

    # 3. apply sliding window guassian filter for S1
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))  # patches for guassian filter

    x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))  # offsets
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))                # default gaussian kernel
    weights = kernel * np.stack([
        (x <= 0) & (y <= 0),                                    # north west sub window
        (x >= 0) & (y <= 0),                                    # north east sub window
        (x <= 0) & (y >= 0),                                    # south west sub window
        (x >= 0) & (y >= 0),                                    # south east sub window
    ], axis=0)                                                  # all the sliding window weights (4, d, d)
    weights /= weights.sum(axis=(-2, -1), keepdims=True)        # normalized weights

    n = np.sum(S1)
    vals = np.sum(patches[S1][:, None, ...] * weights[None, ...], axis=(-2, -1))    # swf all possible outputs (n, 4)
    idxs = np.argmin(np.abs(vals - denoised_img[S1][:, None]), axis=1)              # indexs of minimum sub sliding windows (n, )
    denoised_img[S1] = vals[np.arange(n), idxs]

    # 4. apply attenuation for S2
    denoised_img[S2] = denoised_img[S2] * Atten
    
    # 5. apply dilation operator
    denoised_img = morph_filter(denoised_img, 'max', se=cv2.MORPH_CROSS, size=size)

    return denoised_img


def denoise_with_cwm(img: np.ndarray, size: int=3):
    '''
        Denoise with combined wavelet transform and morphology.
        https://doi.org/10.27060/d.cnki.ghbcu.2020.001632
    '''
    img1 = denoise_with_wavelet(img, 'sym4', level=4, threshold=50)
    img2 = morph_filter(morph_filter(img, 'open', cv2.MORPH_ELLIPSE, size), 'close', cv2.MORPH_ELLIPSE, size)

    denoised_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    return denoised_img


def denoise_with_cnb(img: np.ndarray, size: int, sigma: int=10, sigma_color: int=20, sigma_space: int=20, threshold: int=150):
    '''
        Denoise with combined nlm and blf.
    '''

    def p_stable_hash(data: np.ndarray, w: int, l: int=3) -> dict[int, list] | dict[tuple, list]:
        '''
            P stable locality sensitive hash.
        '''
        _, d2 = data.shape

        a = np.random.normal(0, 1, (d2, l))          # standarad normal distribution
        b = np.random.uniform(0, w, (1, l))         # uniform distribution
        codes = np.floor((data@a+b)/w).astype(int)  # hash codes (n, l)

        tab = defaultdict(list)
        for idx, code in enumerate(codes):
            key = tuple(code.tolist())
            tab[key].append(idx)

        return tab

    def compute_nlm_weights(data: np.ndarray):
        '''
            Compute non-local mean weights.
        '''
        data = data.astype(float)
        sq = np.sum(data**2, axis=1)    # sum of squares (n,)
        dp = data @ data.T              # dot product (n, n)
        ssd = sq[:, None] + sq[None, :] - 2 * dp    # sum of squared differences (n, n)

        w = np.exp(-ssd/(2*sigma**2))   # weights
        ws = np.sum(w)                  # sum of weights

        # print(data, sq, dp, ssd)
        return w / ws

    # 1. Prepare data
    h, w = img.shape
    d = size                                  # the diameter of image patch
    r = size // 2                             # the radius of image patch
    offsets8 = get_neighbors(8, 2)

    denoised_img = np.zeros_like(img)
    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant')             # padded zero as the border of image
    mean_map = cv2.medianBlur(img, d)                                       # mean map
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))  

    # 2. Preselect possible star pixels by mean filter and gray check
    mask = (img == nd.maximum_filter(img, size=d)) & (mean_map > np.percentile(mean_map, 99))
    coords = np.argwhere(mask)                                   # the coordinates of possible star(central pixels)     
    coords = np.reshape(coords[:, None, :]+offsets8, (-1, 2))   # the coordinates of possible star(all pixels)     
    coords = coords[(coords[:, 0] >= 0) & (coords[:, 0] < h) & (coords[:, 1] >= 0) & (coords[:, 1] < w)] # boundary check
    mask[coords[:, 0], coords[:, 1]] = True
    coords = np.argwhere(mask)

    # 3. Do non local mean with each classified patch group
    n = np.sum(mask)
    fpatches = np.reshape(patches[mask], (n, -1))       # flatten possible star patches (n, d²)
    tab = p_stable_hash(fpatches, 10, l=2)              # lsh result
    grays = img[mask]                                   # the gray values of selected pixels (n,)

    if n < 5000:
        weights = compute_nlm_weights(fpatches)
        denoised_img[mask] = np.sum(weights * grays, axis=1)
    else:
        for key in tab:
            idxs = tab[key]
            if len(idxs) == 1:
                continue

            weights = compute_nlm_weights(fpatches[idxs])
            denoised_img[coords[idxs, 0], coords[idxs, 1]] = np.sum(weights * grays[idxs], axis=1)

    # 4. Do modified bilateral filter with other patches
    denoised_img[~mask] = denoise_with_blf(img, 21, sigma_color, sigma_space, threshold)[~mask]
    
    return denoised_img.astype(np.uint8)


def denoise_image(img: np.ndarray, method: str):
    '''
        Denoise the image.
    '''
    if method == 'CNB': # combined nlm and blf
        denoised_img = denoise_with_cnb(img, 3, sigma=10, sigma_color=2, sigma_space=np.inf)
    elif method == 'CWM':
        denoised_img = denoise_with_cwm(img)
    elif method == 'CMG':
        denoised_img = denoise_with_cmg(img)
    elif method == 'NLM_BLF':
        denoised_img = denoise_with_nlm(img, 3, 11) # cv2.addWeighted(denoise_with_nlm(img, 5, 11), 0.5, img, 0.5, 0)
        denoised_img = denoise_with_blf(denoised_img, 3, sigma_color=20, sigma_space=1)
    elif method == 'NLM': # non local mean
        denoised_img = denoise_with_nlm(img, 7, 49)
    elif method == 'MBLF': # modified bilateral filter
        denoised_img = denoise_with_blf(img, 7, sigma_color=20, sigma_space=1.5, threshold=200)
    elif method == 'WAVELET':
        denoised_img = denoise_with_wavelet(img, 'sym4', threshold=40)
    elif method in ['MEAN', 'GAUSSIAN', 'MEDIAN', 'BLF', 'GLF']:
        denoised_img = basic_filter(img, method)
    else:
        denoised_img = img
    return denoised_img


def gen_laplacian_pyramid(img: np.ndarray, levels: int=3):
    '''
        Generate the Laplacian pyramid of the image.
    Args:
        img: the image to be processed
        levels: the number of levels of the pyramid
    Returns:
        pyramid: the Laplacian pyramid
    '''

    gaussian_pyramid = [img]
    for i in range(levels-1):
        # down sample
        img = cv2.pyrDown(gaussian_pyramid[i])
        gaussian_pyramid.append(img)

    laplacian_pyramid = []
    for i in range(levels-1):
        img = cv2.subtract(gaussian_pyramid[i], cv2.pyrUp(gaussian_pyramid[i+1]))
        laplacian_pyramid.append(img)

    # last level
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid
