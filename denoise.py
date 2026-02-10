import cv2
import numpy as np
import pywt
import scipy.ndimage as ndi
from skimage import img_as_ubyte, img_as_float
import skimage.morphology as morph
import skimage.restoration as restoration

from utils import gen_gaussian_kernel
from detect import con_shift_map

eps = 1e-10


def basic_filter(img: np.ndarray, method: str='Gaussian', size: int=3) -> np.ndarray:
    '''
        Conventional noise reducing filters.
    '''

    d = size
    r = size // 2

    if method == 'Gaussian':
        filtered_img = ndi.gaussian_filter(img, sigma=0.7, truncate=(r)/0.7)
    elif method == 'Mean':
        filtered_img = ndi.uniform_filter(img, size=d)
    elif method == 'Median':
        filtered_img = ndi.median_filter(img, d)
    elif method == 'MMedian':
        padded_img = np.pad(img, ((r, r), (r, r)), mode='constant', constant_values=0)
        filtered_img = ndi.median_filter(padded_img, d)[r:-r, r:-r]
    elif method == 'Bilateral':
        # !NOTE: restoration.denoise_bilateral will convert the input image into float for better precision using the img_as_float function and thus the standard deviation (sigma_color) will be in range [0, 1].
        filtered_img = restoration.denoise_bilateral(img, win_size=9, sigma_color=0.1, sigma_spatial=1.5)
    elif method == 'GLF':
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        kernel = cv2.getGaussianKernel(d, 0.7)
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


def morph_filter(img: np.ndarray, size: int=3, method: str='Erode', selem: str='Rect') -> np.ndarray:
    '''
        Morphology filter.
    '''

    d = size
    r = size // 2

    if selem == 'Disk':
        kernel = morph.disk(r)
    elif selem == 'Cross':
        # no built-in function
        kernel = np.zeros((d, d), dtype=bool)
        kernel[:, r] = True
        kernel[r, :] = True
    else:
        kernel = np.ones((d, d), dtype=bool)
    
    if method == 'Erode':
        filtered_img = morph.erosion(img, kernel)
    elif method == 'Dilate':
        filtered_img = morph.dilation(img, kernel)
    elif method == 'Open':
        filtered_img = morph.opening(img, kernel)
    elif method == 'Close':
        filtered_img = morph.closing(img, kernel)
    else:
        print('Invalid morph method')
        return None
    
    return filtered_img


def denoise_with_nlm(img: np.ndarray, patch_size: int=7, wind_size: int=21, sigma: float=10):
    '''
        Non-local means denoising.
    '''

    if img.dtype == np.uint8:
        denoised_img = cv2.fastNlMeansDenoising(
            img, 
            h=sigma, 
            templateWindowSize=patch_size, 
            searchWindowSize=wind_size
        )
    else:
        denoised_img = restoration.denoise_nl_means(
            img,
            patch_size=patch_size,
            patch_distance=wind_size//2,
            h=sigma,
            # fast_mode=True
        )
    
    return denoised_img


def denoise_with_wavelet(img: np.ndarray, wavelet: str, level: int=3, threshold: float=2):
    '''
        Denoise with wavelet transform.
    '''

    if img.dtype == np.uint8:
        coeffs = pywt.wavedec2(img, wavelet, level=level)
        cA = coeffs[0]              # low freq coeff
        cD = coeffs[1:]             # high freq coeff
        denoised_coeffs = [cA]      # keep low freq coeff
        for cd in cD:
            denoised_cd = [pywt.threshold(band, threshold, mode='soft') for band in cd]
            denoised_coeffs.append(denoised_cd)
        denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    else:
        denoised_image = restoration.denoise_wavelet(img, wavelet=wavelet, wavelet_levels=level, rescale_sigma=True, sigma=0.065)

    return denoised_image


def denoise_with_shearlet(img: np.ndarray, shearlet: str, level: int=3, threshold: float=2):
    '''
        Denoise with shearlet transform.
    '''
    pass


def denoise_with_blf(img: np.ndarray, size: int, sigma_g: float, sigma_s: float, threshold: int=100):
    '''
        Denoise with modified bilateral filter.
    '''

    def custom_activation(x, threshold):
        '''
            Custom activation function.
        '''
        return np.where(x > threshold, np.inf, x)

    d = size
    if d % 2 == 0:
        d = d + 1
    r = d // 2

    # pad the image for color weight calculation and change the type in case negative overflow
    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant')

    # use a strided view to get the patches(h, w, d, d)
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))
    grays = img[..., None, None] # the central gray value of each image patch

    # calculate the color and space weights
    color_diffs = custom_activation(np.abs(grays - patches), threshold)
    color_weights = gen_gaussian_kernel(sigma=sigma_g, x=color_diffs, normalize=False)
    space_weights = gen_gaussian_kernel(sigma=sigma_s, size=d)

    # apply bilateral filter    
    bilateral_weights = color_weights * space_weights
    bilateral_weights = bilateral_weights / bilateral_weights.sum(axis=(-2, -1), keepdims=True)
    filtered_img1 = (bilateral_weights * patches).sum(axis=(-2, -1))

    # calculate weight sum of 8-connectivity neighbor
    mask = np.ones((d, d), dtype=bool)
    mask[r, r] = False
    weightsum = bilateral_weights[..., mask].sum(axis=-1)

    # apply mean filter
    mean_weights = np.full((d, d), 1 / (d**2 - 1))
    mean_weights[r, r] = 0
    filtered_img2 = (mean_weights[None, None, ...] * patches).sum(axis=(-2, -1))

    # when weightsum of BLF is small, the central pixel is probably a pepper noise point, then use the mean filter as output
    denoised_img = np.where((np.abs(weightsum) > 0.1) | (img < threshold), filtered_img1, filtered_img2)

    return denoised_img


def denoise_with_amf(img: np.ndarray, size1: int, size2: int):
    '''
        Denoise with adaptive median filter.
    '''

    h, w = img.shape

    # generate all the windows sizes
    sizes = np.arange(size1, size2+1, 2)  #(n, )

    # get median, min and max of img under different the window sizes
    mids = np.stack([basic_filter(img, 'Median', size=s) for s in sizes])   # (n, h, w)
    mins = np.stack([ndi.minimum_filter(img, size=s) for s in sizes])       # (n, h, w)
    maxs = np.stack([ndi.maximum_filter(img, size=s) for s in sizes])       # (n, h, w)
    
    # get valid median values
    mask1 = (mins < mids) & (mids < maxs)                                   # (n, h, w)
    
    # get possible noisy pixels
    mask2 = (mins >= img[None, ...]) | (maxs <= img[None, ...])             # (n, h, w)

    # set possible noisy pixels to valid median values
    i, j = np.meshgrid(np.arange(0, h), np.arange(0, w), indexing='ij')
    k = np.argmax(mask1 & mask2, axis=0)  # the first window size index whose median value is valid
    
    denoised_img = np.where(
        np.any(mask1 & mask2, axis=0),  # (h, w)
        mids[k, i, j],
        img        
    )
    
    return denoised_img


def denoise_with_emf(img: np.ndarray, size: int, threshold: float):
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

    d = size
    r = d // 2
    h, w = img.shape
    max_val = 255 if img.dtype == np.uint8 else 1.0
    
    padded_img = np.pad(img, ((r, r), (r, r)))
    denoised_img = img.copy()

    # 1.initial check with extreme values
    mask = (np.abs(img - np.max(img))  < threshold) & (np.abs(img - np.min(img))  < threshold)
    denoised_img[mask] = basic_filter(img, 'Median', d)[mask]

    # 2.double check with energy
    es0, ed0 = compute_energy(img, img)
    es1, ed1 = compute_energy(img, denoised_img)
    mask = mask & (es1 + ed1 < es0 + ed0) & (np.sum(es0 + ed0) / (h * w) < es0 + ed0)     # selected noise mask

    # 3.adapative median filter 0 or max_val pixels
    mask1 = mask & ((img == 0) | (img == max_val))                              # extreme noise
    denoised_img[mask1] = denoise_with_amf(img, d, d+5)[mask1]                  # denoise with adapative median filter
    
    # 4.argmin engery
    mask2 = mask & (~mask1)                                                     # other noise
    coords = np.argwhere(mask2)
    offsets = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=int)
    neighbors = coords[:, None, :] + offsets[None, ...]

    x0 = img[coords[:, 0], coords[:, 1]]                                        # (n, )
    xx0 = padded_img[neighbors[..., 0] + r, neighbors[..., 1] + r]              # (n, 4)

    a = np.where((xx0 == 0) | (xx0 == max_val), 1, 2)                           # factors for argmin energy calculation
    x = np.linspace(0, max_val, num=256)                                        # possible pixel values (m, )
    y = np.abs(x[None, :] - x0[:, None]) + np.sum(a[:, None, :] * np.power(np.abs(x[None, :, None] - xx0[:, None, :]), 1.3), axis=-1)

    i = np.argmin(y, axis=1)
    denoised_img[mask2] = x[i]

    return denoised_img


def denoise_with_cmg(img: np.ndarray, size: int=5, sigma: float=1):
    '''
        Denoise with combined morphology operation and modified gaussian filter.
        https://doi.org/10.27241/d.cnki.gnjgu.2024.001923
    '''

    T1 = 3
    T2 = 1.3
    T3 = 1.0
    Atten = 2

    d = size
    r = d // 2

    # 1. apply erosion operator
    denoised_img = morph_filter(img, size=d, method='Erode', selem='Disk') 

    # 2. select non-star pixels
    neighbor_4 = np.array([
        [False, True, False],
        [True,  False, True],
        [False, True, False],
    ], dtype=bool)
    denoised_img = np.maximum(denoised_img, eps)                                                    # avoid division by 0 in later operation
    min_map = ndi.minimum_filter(denoised_img, footprint=neighbor_4, mode='constant', cval=np.inf)  # min map
    max_map = ndi.maximum_filter(denoised_img, footprint=neighbor_4, mode='constant', cval=-np.inf) # max map
    
    Id = np.percentile(denoised_img, 95)
    S1 = (denoised_img < Id) & (denoised_img / min_map > T1)                                        # non star pixels
    S2 = (denoised_img > Id) & ((denoised_img / max_map < T2) | (max_map / denoised_img < T2)) & (max_map / min_map < T3) # star pixels

    # 3. apply sliding window guassian filter for S1
    padded_img = np.pad(denoised_img, ((r, r), (r, r)))                     # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))  # patches for guassian filter

    x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))              # offsets
    kernel = gen_gaussian_kernel(sigma=sigma, size=d)                       # default gaussian kernel
    weights = kernel * np.stack([
        x <= 0, x >= 0,                                                     # west and east
        y <= 0, y >= 0,                                                     # south and north
        (x <= 0) & (y <= 0),                                                # north west sub window
        (x >= 0) & (y <= 0),                                                # north east sub window
        (x <= 0) & (y >= 0),                                                # south west sub window
        (x >= 0) & (y >= 0),                                                # south east sub window
    ], axis=0)                                                              # all the sliding window weights (8, d, d)
    weights /= weights.sum(axis=(-2, -1), keepdims=True)                    # normalized weights
    # cv2.imwrite('S1.png', S1 * 255)
    # cv2.imwrite('S2.png', S2 * 255)

    n = np.sum(S1)
    vals = np.sum(patches[S1][:, None, ...] * weights[None, ...], axis=(-2, -1))    # swf all possible outputs
    idxs = np.argmin(np.abs(vals - denoised_img[S1][:, None]), axis=1)              # indexs of minimum sub sliding windows (n, )
    denoised_img[S1] = vals[np.arange(n), idxs]

    # 4. apply attenuation for S2
    denoised_img[S2] = denoised_img[S2] * Atten
    
    # 5. apply dilation operator
    denoised_img = morph_filter(denoised_img, size=d, method='Dilate', selem='Disk')

    return denoised_img


def denoise_with_swf(img: np.ndarray, size: int=5, sigma: float=1):
    '''
        Denoise with side window filter.
        https://doi.org/10.27543/d.cnki.gkgdk.2022.000001
    '''

    Tdmin = 20 / 255
    Tdmax = 1
    Td = 0

    ## 0. Initialize
    d = size
    r = d // 2
    padded_img = np.pad(img, ((r, r), (r, r)))                     # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))  # patches for guassian filter

    ## 1. Side window filter
    x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))              # offsets
    kernel = gen_gaussian_kernel(sigma=sigma, size=d)                       # default gaussian kernel
    weights = kernel * np.stack([
        (y <= x) & (y <= -x),
        (y >= x) & (y >= -x),
        (x <= y) & (x <= -y),
        (x >= y) & (x >= -y),
    ], axis=0)                                                              # all the sliding window weights (4, d, d)
    weights /= weights.sum(axis=(-2, -1), keepdims=True)                    # normalized weights
    filtered_imgs = np.sum(patches[None, ...] * weights[:, None, None, ...], axis=(-1, -2)) # side window filter response(4, h, w)

    ## 2. Segment the image
    diffs = np.abs(filtered_imgs - img[None, ...])
    min_idxs, max_idxs = np.argmin(diffs, axis=0), np.argmax(diffs, axis=0)
    min_img = np.take_along_axis(filtered_imgs, min_idxs[None, :, :], axis=0).squeeze(0)
    max_img = np.take_along_axis(filtered_imgs, max_idxs[None, :, :], axis=0).squeeze(0)

    # mask_outlier = np.all(img[None, ...] > filtered_imgs, axis=0) & np.all(diffs < Tdmin, axis=0) & (img - min_img > Tdmin)
    # mask_target = np.all(img[None, ...] > filtered_imgs, axis=0) & ((img - min_img > Tdmax) | (max_img - min_img > Td))
    max_map, min_map = ndi.maximum_filter(img, size=d), ndi.minimum_filter(img, size=d)
    mean_map, mean2_map = ndi.uniform_filter(img, size=d), ndi.uniform_filter(img**2.0, size=d)
    devi_map = np.sqrt(np.maximum(mean2_map - mean_map**2.0, eps))
    mask_outlier = ((img == max_map) | (img == min_map)) & (np.abs(img - mean_map) > 3 * devi_map)
    mask_target = ndi.maximum_filter(img, size=d) == img

    cv2.imwrite('m_outlier.png', mask_outlier * 255)
    cv2.imwrite('m_target.png', mask_target * 255)

    ## 3. Output
    denoised_img = img.copy()
    denoised_img[mask_outlier] = 0.1 * max_img[mask_outlier]
    denoised_img[~(mask_outlier & mask_target)] = min_img[~(mask_outlier & mask_target)]

    return denoised_img


def denoise_with_cwm(img: np.ndarray, size: int=3):
    '''
        Denoise with combined wavelet transform and morphology.
        https://doi.org/10.27060/d.cnki.ghbcu.2020.001632
    '''
    img1 = denoise_with_wavelet(img, 'sym4', level=4, threshold=20)
    img2 = morph_filter(morph_filter(img, size, 'Open', selem='Disk'), size, 'Close', selem='Disk')

    frac = 0.85
    denoised_img = frac * img1 + (1 - frac) * img2

    return denoised_img


def denoise_with_cnb(img: np.ndarray, patch_size: int, wind_size: int=11, sigma: int=10, sigma_g: int=20, sigma_s: int=20, sigma_i: int=20, sigma_j: int=20, road_kth: int=5):
    '''
        Denoise with combined nlm and blf.
    '''

    def preselect_similar(mean_map: np.ndarray, threshold: np.ndarray):
        '''
            Preselect the similar patches and return the mask.
        '''
        assert mean_map.shape == threshold.shape
        grouped_mean_map = np.lib.stride_tricks.sliding_window_view(
            np.pad(mean_map, ((kk, kk), (kk, kk))), 
            (k, k)
        )
        mask = np.abs(grouped_mean_map - mean_map[..., None, None]) < threshold[..., None, None]
        return mask

    def compute_nlm_weights(fpatches, foreground: np.ndarray, valid: np.ndarray):
        '''
            Compute non-local mean weights.
        '''
        winds = np.lib.stride_tricks.sliding_window_view(                       # search windows (h, w, k, k, d²)
            np.pad(fpatches, ((kk, kk), (kk, kk), (0, 0)), mode='constant'),
            window_shape=(k, k),
            axis=(0, 1)
        ).transpose(0, 1, 3, 4, 2)                                              #! NOTE: By default, sliding_window_view pushes the newly generated window dimensions to the end of the array.

        swinds, spatches = winds[foreground], fpatches[foreground]
        sims = np.where(                                                        # similarities
            valid[foreground],
            np.mean((swinds - spatches[:, None, None, :])**2, axis=-1),
            np.inf, 
        )
        wt = np.exp(-sims / sigma**2)                                           # nlm weights (n, k, k)
        wt[:, kk, kk] = eps      
        wt[:, kk, kk] = np.max(wt, axis=(-1, -2))                               # central nlm weight
        wt = wt / np.maximum(np.sum(wt, axis=(-1, -2), keepdims=True), eps)     # normalize the weights

        return wt
    
    def compute_blf_weights(winds: np.ndarray, tmean_map: np.ndarray, devi_map: np.ndarray):
        '''
            Compute bilateral filter weights.
        '''

        ad = np.abs(winds[..., kk, kk, None, None] - winds)                     # absolute differences
        ead = np.exp(-ad ** 2 / (2 * sigma_i ** 2))                             # exponential absolute differences
        road = np.partition(ad.reshape(h, w, -1), kth=road_kth, axis=-1)        # rank ordered absolute differences
        stroad = np.sum(road[..., :road_kth], axis=-1)                          # sum of trimmed rank ordered absolute differences
        grouped_stroad = np.lib.stride_tricks.sliding_window_view(              # grouped stroad
            np.pad(stroad, ((kk, kk), (kk, kk))), 
            (k, k)
        )
        grouped_tmean_map = np.lib.stride_tricks.sliding_window_view(
            np.pad(tmean_map, ((kk, kk), (kk, kk))), 
            (k, k)
        )
        grouped_devi_map = np.lib.stride_tricks.sliding_window_view(
            np.pad(devi_map, ((kk, kk), (kk, kk))), 
            (k, k)
        )
        diff = grouped_tmean_map - tmean_map[..., None, None] + grouped_devi_map - devi_map[..., None, None]

        gwt = np.exp(-diff ** 2.0 / (2 * sigma_g ** 2))                         # gray weights (h, w, d, d)
        iwt = np.exp(-stroad ** 2.0 / (2 * sigma_i ** 2))                       # impulse weights based on ead or exponential stroad (h, w)
        # iwt = np.minimum((np.sum(ead, axis=(-1, -2)) - 1) / k, 1)
        iwt = np.lib.stride_tricks.sliding_window_view(                         # impulse weights (h, w) -> (h, w, d, d)
            np.pad(iwt, ((kk, kk), (kk, kk))), 
            (k, k)
        )
        
        jcoef = 1 - np.exp(-(stroad[..., None, None] + grouped_stroad) ** 2 / (8 * sigma_j ** 2)) # joint modulation coefficient between gray weights and impulse weights
        wt = gwt ** (1 - jcoef) * iwt ** jcoef                                  # weights (h, w, d, d)
        wt = wt / np.maximum(np.sum(wt, axis=(-1, -2), keepdims=True), eps)     # normalize the weights

        return wt

    ## 0. Prepare data
    h, w = img.shape
    d = patch_size                              # the diameter of image patch
    k = wind_size                               # the diameter of search window
    r = d // 2                                  # the radius of image patch, namely half of d
    kk = k // 2                                 # the radius of search window, namely half of k
    
    denoised_img = np.copy(img)                                                     # denoised image
    patches = np.lib.stride_tricks.sliding_window_view(
        np.pad(img.astype(float), ((r, r), (r, r))), 
        (d, d)
    )                                                                               # image patches (h, w, d, d)
    winds = np.lib.stride_tricks.sliding_window_view(
        np.pad(img.astype(float), ((kk, kk), (kk, kk))), 
        (k, k)
    )                                                                               # search windows (h, w, d, d)
    flatten_patches = np.reshape(patches, (h, w, -1))
    sorted_patches = np.sort(flatten_patches, axis=-1)

    ## 1. Construct local contrast measure
    max_map, min_map = ndi.maximum_filter(img, size=d), ndi.minimum_filter(img, size=d) # max and min map of image patches
    mean_map, mean2_map = ndi.uniform_filter(img, size=d), ndi.uniform_filter(img**2.0, size=d) # mean and mean sqaure map of image patches
    devi_map = np.sqrt(np.maximum(mean2_map - mean_map**2.0, eps))                  # standard deviation map of image patches
    tmean_map = np.mean(sorted_patches[..., 2:-2], axis=-1)                         # trimmed mean map of image patches
    shift_mask, shifted_tmean_map = con_shift_map(tmean_map, size=d)
    bright_mask = tmean_map[None, ...] - shifted_tmean_map > 0
    residual_map = np.where(shift_mask & bright_mask, tmean_map[None, ...] - shifted_tmean_map, np.nan)
    products = residual_map[0:8:2] * residual_map[1:8:2]
    measure = np.nanmin(products, axis=0, initial=np.inf)
    measure[np.isinf(measure)] = eps
    
    ## 2. Segment image
    outlier_mask = ((img == max_map) | (img == min_map)) & (np.abs(img - mean_map) > 3 * devi_map) # outlier mask / peper noise
    target_mask = (measure > np.mean(measure) + 3 * np.std(measure)) & (~outlier_mask)             # target mask / star pixels
    target_mask = morph_filter(target_mask, method='Dilate', selem='Rect')
    cv2.imwrite('m_outlier.png', outlier_mask * 255)
    cv2.imwrite('m_target.png', target_mask * 255)

    ## 3. Replace outliers with median values
    grouped_outlier_mask = np.lib.stride_tricks.sliding_window_view(
        np.pad(outlier_mask, ((r, r), (r, r))),
        (d, d)
    )
    denoised_img[outlier_mask] = np.nanmedian(
        np.where(grouped_outlier_mask, np.nan, patches),  
        axis=(-1, -2)
    )[outlier_mask]

    ## 4. Process star pixels with NLM
    # similar_mask = preselect_similar(tmean_map, devi_map)                           # similar patch mask (h, w, k, k)
    # weights = compute_nlm_weights(patches.reshape(h, w, -1), target_mask, similar_mask) # nlm weights (n, k, k)
    # denoised_img[target_mask] = np.sum(winds[target_mask] * weights, axis=(-1, -2))
    denoised_img[target_mask] = denoise_with_nlm(img, patch_size, wind_size, sigma)[target_mask]

    ## 5. Process all the pixels with MBLF
    weights = compute_blf_weights(winds, tmean_map, devi_map)
    denoised_img[~target_mask] = np.sum(winds * weights, axis=(-1, -2))[~target_mask]

    return denoised_img


def denoise_image(img: np.ndarray, method: str):
    '''
        Denoise the image.
    '''
    # convert the input image for better precision
    img = img_as_float(img)

    # Check and force scaling (if necessary)
    if img.max() > 1.0:
        img = img / img.max()

    if method == 'CNB': # combined nlm and blf
        denoised_img = denoise_with_cnb(img, 5, 11, sigma=0.1, sigma_g=0.05, sigma_s=3, sigma_i=0.2, sigma_j=0.5, road_kth=6)
    elif method == 'CWM':
        denoised_img = denoise_with_cwm(img, size=3)
    elif method == 'CMG':
        denoised_img = denoise_with_cmg(img, size=5, sigma=2)
    elif method == 'SWF':
        denoised_img = denoise_with_swf(img, size=5, sigma=1.2)
    elif method == 'AMF':
        denoised_img = denoise_with_amf(img, size1=3, size2=11)
    elif method == 'EMF':
        denoised_img = denoise_with_emf(img, size=5, threshold=10)
    elif method == 'NLM_BLF':
        denoised_img = denoise_with_nlm(img, patch_size=3, wind_size=11, sigma=0.05)
        denoised_img = denoise_with_blf(denoised_img, size=3, sigma_g=0.05, sigma_s=1, threshold=0.4)
    elif method == 'NLM': # non local mean
        denoised_img = denoise_with_nlm(img, 5, 21, sigma=0.08)
    elif method == 'MBLF': # modified bilateral filter
        denoised_img = denoise_with_blf(img, 7, sigma_g=0.05, sigma_s=1.5, threshold=0.6)
    elif method == 'Wavelet':
        denoised_img = denoise_with_wavelet(img, 'db4', threshold=10)
    elif method in ['Mean', 'Gaussian', 'Median', 'MMedian', 'Bilateral', 'GLF']:
        denoised_img = basic_filter(img, method)
    else:
        denoised_img = img

    # clip and astype
    denoised_img = img_as_ubyte(np.clip(denoised_img, 0, 1))

    return denoised_img

