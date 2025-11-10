import cv2
import numpy as np
import pywt
import scipy.ndimage as nd
from scipy.optimize import minimize
from functools import partial
from numba import jit

from collections import defaultdict
from utils import get_offsets


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


def denoise_with_morph(img: np.ndarray, size: int=3):
    '''
        Denoise with Morphology filter.
    '''
    pass


def denoise_with_blf(img: np.ndarray, size: int, sigma_g: float, sigma_s: float, threshold: int=100):
    '''
        Improved bilateral filter denoising.
    Args:
        img: the image to be processed
        size: the size of template
        sigma_g: the standard deviation of the color space
        sigma_s: the standard deviation of the coordinate space
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
    color_weights = np.exp(-(color_diffs ** 2) / (2 * sigma_g ** 2))

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    space_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))

    # apply bilateral filter    
    bilateral_weights = color_weights * space_weights
    bilateral_weights = bilateral_weights / bilateral_weights.sum(axis=(-2, -1), keepdims=True)
    filtered_img1 = (bilateral_weights * patches).sum(axis=(-2, -1))

    # calculate weight sum of 8-connectivity neighbor
    neighbors = get_offsets(8)+r
    weightsum = bilateral_weights[..., neighbors[:, 0], neighbors[:, 1]].sum(axis=-1)

    # apply mean filter
    mean_weights = np.zeros((d, d))
    mean_weights[neighbors[:, 0], neighbors[:, 1]] = 1
    mean_weights = mean_weights / mean_weights.sum()
    filtered_img2 = (mean_weights[None, None, ...] * patches).sum(axis=(-2, -1))

    denoised_img = np.where((np.abs(weightsum) > 0.1) | (img < threshold), filtered_img1, filtered_img2).astype(np.uint8)

    return denoised_img


def denoise_with_amf(img: np.ndarray, size1: int, size2: int):
    '''
        Denoise with adaptive median filter.
    '''

    h, w = img.shape

    # generate all the windows sizes
    sizes = np.arange(size1, size2+1, 2)  #(n, )

    # get median, min and max of img under different the window sizes
    mids = np.stack([basic_filter(img, 'MEDIAN', size=s) for s in sizes])   # (n, h, w)
    mins = np.stack([morph_filter(img, 'min', size=s) for s in sizes])      # (n, h, w)
    maxs = np.stack([morph_filter(img, 'max', size=s) for s in sizes])      # (n, h, w)
    
    # get valid median values
    mask1 = (mins < mids) & (mids < maxs)                                   # (n, h, w)
    
    # get possible noisy pixels
    mask2 = (mins >= img[None, ...]) | (maxs <= img[None, ...])             # (n, h, w)

    # set possible noisy pixels to valid median values
    i, j = np.meshgrid(np.arange(0, h), np.arange(0, w))
    k = np.argmax(mask1 & mask2, axis=0)  # the first window size index whose median value is valid
    
    denoised_img = np.where(
        np.any(mask1 & mask2, axis=0),  # (h, w)
        mids[k, i, j],
        img        
    )
    
    return denoised_img


def denoise_with_emf(img: np.ndarray, size: int, threshold: int):
    '''
        Denoise with extreme median filter.
        https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2017&filename=DZYX201706017
    '''

    def compute_energy(src: np.ndarray, tgt: np.ndarray):
        '''
            Compute energy.
        '''
        es = np.abs(src-tgt)
        ed = np.zeros_like(es)
        ed[1:, :] += np.abs(tgt[1:, :] - src[:-1, :])   # top
        ed[:-1, :] += np.abs(tgt[:-1, :] - src[1:, :])  # bot
        ed[:, 1:] += np.abs(tgt[:, 1:] - src[:, :-1])   # left
        ed[:, :-1] += np.abs(tgt[:, :-1] - src[:, 1:])  # right
    
        return es, ed

    h, w = img.shape
    padded_img = np.pad(img, ((size, size), (size, size)))
    denoised_img = img.copy()

    # 1.initial check with extreme values
    mask = (np.abs(img - np.max(img))  < threshold) & (np.abs(img - np.min(img))  < threshold)
    denoised_img[mask] = basic_filter(img, 'MEDIAN', size)[mask]

    # 2.double check with energy
    es0, ed0 = compute_energy(img, img)
    es1, ed1 = compute_energy(img, denoised_img)
    mask = mask & (es1+ed1 < es0+ed0) & (np.sum(es0+ed0) / (h*w) < es0+ed0)

    # 3.adapative median filter 0/255 pixels
    mask1 = mask & ((img == 0) | (img == 255))
    denoised_img[mask1] = denoise_with_amf(img, size, size+5)[mask1]
    
    # 4.argmin engery
    mask2 = mask & (~mask1)
    coords = np.argwhere(mask2)
    offsets = get_offsets(4)
    neighbors = coords[:, None, :] + offsets[None, ...]

    n = np.sum(mask2)
    x0 = img[coords[:, 0], coords[:, 1]]                                # (n, )
    xx0 = padded_img[neighbors[..., 0]+size, neighbors[..., 1]+size]    # (n, 4)
    a = np.where((xx0 == 0) | (xx0 == 255), 1, 2)                       # factors for energy calculation

    x = np.arange(0, 255, threshold)                                    # possible values (m, )
    xx = np.ones((n, 1), dtype=int) * x                                 # (n, m)

    y1 = np.abs(xx - x0[:, None])                                                    # energy_s (n, m)
    y2 = np.sum(a[:, None, :] * np.power(np.abs(xx[..., None] - xx0[:, None, :]), 1.3), axis=-1)   # energy_d (n, m)
    y = y1 + y2

    i = np.argmin(y, axis=1)
    denoised_img[coords[:, 0], coords[:, 1]] = x[i]

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
    offsets = get_offsets(4)                                                  # 4-connectivity offsets (4, 2)
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


def denoise_with_cnb(img: np.ndarray, size: int, wind: int=7, sigma: int=10, sigma_g: int=20, sigma_s: int=20, sigma_i: int=20, trim: int=2):
    '''
        Denoise with combined nlm and blf.
    '''

    def preselect_similar(mean: np.ndarray, threshold: int=10):
        '''
            Preselect the similar patches and return the indexs.
        '''
        index_img = np.arange(0, h*w).reshape(h, w)                                         # index image for patch search
        padded_img = np.pad(index_img, ((k//2, k//2), (k//2, k//2)), constant_values=-1)    # padded index image
        windows = np.lib.stride_tricks.sliding_window_view(padded_img, (k, k))              # search windows (h, w, k, k)

        padded_mean = np.pad(mean, ((k//2, k//2), (k//2, k//2)), constant_values=0)
        grouped_mean = np.lib.stride_tricks.sliding_window_view(padded_mean, (k, k))

        # padded_devi = np.pad(devi, ((k//2, k//2), (k//2, k//2)), constant_values=0)
        # grouped_devi = np.lib.stride_tricks.sliding_window_view(padded_devi, (k, k))  

        mask = (np.abs(grouped_mean - mean[..., None, None]) < threshold) 
            # & (np.abs(grouped_devi - devi[..., None, None]) < threshold)  
        indexs = np.where(mask, windows, -1)

        return indexs

    def compute_nlm_weights(p: np.ndarray, sp: np.ndarray, idx: np.ndarray):
        '''
            Compute non-local mean weights.
        '''
        ssp = p[idx]                            # similar patches for star patches
        sim = np.where(                         # similarity between two patches
            idx==-1,                            # avoid invalid patch index(-1)
            np.inf,                         
            np.mean((sp[:, None, :] - ssp)**2, axis=-1),
        )
        wt = np.exp(-sim/(sigma**2))            # weights (n, k*k)
        ci = k**2//2                            # central pixel index
        wt[ci] = np.maximum(                    # central pixel weight
            np.max(wt[:ci]),
            np.max(wt[ci+1:])
        )
        wts = np.sum(wt, axis=-1, keepdims=True) # sum of weights (n, 1)
        return wt / wts
    
    def compute_blf_weights(g: np.ndarray, p: np.ndarray):
        '''
            Compute bilateral filter weights.
        '''
        gwt = np.exp(-(g[..., None, None] - p) ** 2 / (2 * sigma_g ** 2))   # gray weights

        r = d // 2
        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        swt = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))               # spatial weights

        ad = np.abs(g[..., None, None] - p).reshape(h, w, -1)                   # absolute differences between center pixels and other pixels
        road = np.sum(np.sort(ad, axis=-1)[..., :3], axis=-1)                   # rank ordered absolute differences
        iwt = np.exp(- road**2 / (2 * sigma_i ** 2))                      # impulse weights
        iwt = np.lib.stride_tricks.sliding_window_view(                         # (h, w) -> (h, w, d, d)
            np.pad(iwt, ((r, r), (r, r))), 
            (d, d)
        )             

        wt = gwt*swt*iwt                                                        # bilateral weights (h, w, d, d)
        wts = np.sum(wt, axis=(-1, -2), keepdims=True)                          # sum of weights (h, w, 1, 1)

        return wt / wts

    ## 1. Prepare data
    h, w = img.shape
    d = size                                    # the diameter of image patch
    k = wind                                    # the diameter of search window

    img = img.astype(float)
    denoised_img = np.zeros_like(img)                                               # denoised image
    padded_img = np.pad(img, ((d//2, d//2), (d//2, d//2)), mode='reflect')          # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))          # patches (h, w, d, d)
    
    tmmap = np.mean(np.sort(patches.reshape(h, w, -1))[..., trim:-trim], axis=-1)         # trimmed mean map
    mean = np.mean(img)                                                             # mean
    devi = np.std(img)                                                              # standard deviation

    ## 2. Segment image
    ot_mask = (img == nd.maximum_filter(img, size=d)) & (img > tmmap+5*devi) & (tmmap < mean+1.5*devi)       # outlier mask / peper noise
    fg_mask = (~ot_mask) & (img >= mean+3*devi)                                     # foreground mask / star pixels
    bg_mask = ~fg_mask                                                              # background mask / non-star pixels

    ## 3. Process star pixels with NLM
    fimg = img.reshape(-1)                                      # flatten image
    fpatches = patches.reshape(-1, d**2)                        # flatten patches (h*w, d*d)
    spatches = patches[fg_mask].reshape(-1, d**2)               # flatten star patches (n, d*d)
    indexs = preselect_similar(tmmap)[fg_mask].reshape(-1, k*k) # similar patch indexs (n, k*k)
    weights = compute_nlm_weights(fpatches, spatches, indexs)   # nlm weights (n, k*k)
    print(indexs.dtype, np.sum(indexs!=-1, axis=-1))
    print(fimg[indexs], weights)
    denoised_img[fg_mask] = np.sum(fimg[indexs] * weights, axis=-1)

    ## 4. Process outliers and non-star pixels with BLF
    weights = compute_blf_weights(img, patches)
    denoised_img[bg_mask] = np.sum(weights * patches, axis=(-1, -2))[bg_mask]

    return denoised_img.astype(np.uint8)


def denoise_image(img: np.ndarray, method: str):
    '''
        Denoise the image.
    '''
    if method == 'CNB': # combined nlm and blf
        denoised_img = denoise_with_cnb(img, 5, 7, sigma=10, sigma_g=20, sigma_s=3)
    elif method == 'CWM':
        denoised_img = denoise_with_cwm(img)
    elif method == 'CMG':
        denoised_img = denoise_with_cmg(img)
    elif method == 'AMF':
        denoised_img = denoise_with_amf(img, 3, 11)
    elif method == 'EMF':
        denoised_img = denoise_with_emf(img, 5, 10)
    elif method == 'NLM_BLF':
        denoised_img = denoise_with_nlm(img, 3, 11) # cv2.addWeighted(denoise_with_nlm(img, 5, 11), 0.5, img, 0.5, 0)
        denoised_img = denoise_with_blf(denoised_img, 3, sigma_g=20, sigma_s=1)
    elif method == 'NLM': # non local mean
        denoised_img = denoise_with_nlm(img, 3, 7, sigma=10)
    elif method == 'MBLF': # modified bilateral filter
        denoised_img = denoise_with_blf(img, 7, sigma_g=20, sigma_s=1.5, threshold=200)
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
