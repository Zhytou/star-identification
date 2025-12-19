import cv2
import numpy as np
import pywt
import scipy.ndimage as ndi
import skimage.morphology as morph
import skimage.restoration as restoration

# from utils import find_close_pair
eps = 1e-10

def basic_filter(img: np.ndarray, method: str='Gaussian', size: int=3) -> np.ndarray:
    '''
        Conventional noise reducing filters.
    Args:
        img: the image to be processed
        method: the method of filtering
    Returns:
        filtered_img: the image after filtering
    '''
    if method == 'Gaussian':
        filtered_img = ndi.gaussian_filter(img, sigma=0.7, truncate=(size//2)/0.7)
    elif method == 'Mean':
        filtered_img = ndi.uniform_filter(img, size=size)
    elif method == 'Median':
        # d = size//2
        # padded_img = np.pad(img, ((d, d), (d, d)), mode='constant')
        filtered_img = ndi.median_filter(img, size)
    elif method == 'Bilateral':
        filtered_img = restoration.denoise_bilateral(img, win_size=9, sigma_color=20, sigma_spatial=1.5)
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


def morph_filter(img: np.ndarray, size: int=3, method: str='Erode', selem: str='Rect') -> np.ndarray:
    '''
        Morphology filter.
    '''
    if selem == 'Disk':
        kernel = morph.disk(size//2)
    elif selem == 'Cross':
        # no built-in function
        kernel = np.zeros((size, size), dtype=bool)
        kernel[:, size//2] = True
        kernel[size//2, :] = True
    else:
        kernel = np.ones((size, size), dtype=bool)
    
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
    Args:
        img: the image to be processed
        wavelet: the name of wavelet
        level: the decomposition level
    Returns:
        denoised_img: the image after filtering
    '''

    if True:
        coeffs = pywt.wavedec2(img, wavelet, level=level)
        cA = coeffs[0]              # low freq coeff
        cD = coeffs[1:]             # high freq coeff
        denoised_coeffs = [cA]      # keep low freq coeff
        for cd in cD:
            denoised_cd = [pywt.threshold(band, threshold, mode='soft') for band in cd]
            denoised_coeffs.append(denoised_cd)
        denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    else:
        denoised_image = restoration.denoise_wavelet(img, wavelet=wavelet, wavelet_levels=level)

    return denoised_image


def denoise_with_shearlet(img: np.ndarray, shearlet: str, level: int=3, threshold: float=2):
    '''
        Denoise with shearlet transform.
    '''
    pass


def denoise_with_blf(img: np.ndarray, size: int, sigma_g: float, sigma_s: float, threshold: int=100):
    '''
        Denoise with modified bilateral filter.
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
    color_weights = np.exp(-(color_diffs ** 2) / (2 * sigma_g ** 2))

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    space_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))

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
    i, j = np.meshgrid(np.arange(0, h), np.arange(0, w))
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

    Id = np.percentile(img, 90)
    T1 = 1.4
    T2 = 1.2
    T3 = 3
    Atten = 4

    d = size
    r = d // 2

    # 1. apply erosion operator
    denoised_img = morph_filter(img, size=d, method='Erode', selem='Disk') 

    # 2. select non-star pixels
    cross = np.array([
        [False, True, False],
        [True,  True, True],
        [False, True, False],
    ], dtype=bool)
    img = np.maximum(img, eps)                                                          # avoid division by 0 in later operation
    min_map = ndi.minimum_filter(img, footprint=cross, mode='constant', cval=np.inf)    # min map
    max_map = ndi.maximum_filter(img, footprint=cross, mode='constant', cval=-np.inf)   # max map
    
    S1 = (img < Id) & (img / min_map > T1)                                                      # non star pixels
    S2 = (img > Id) & ((img / max_map < T2) | (max_map / img < T2)) & (max_map / min_map < T3)  # star pixels

    # 3. apply sliding window guassian filter for S1
    padded_img = np.pad(img, ((r, r), (r, r)))                              # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))  # patches for guassian filter

    x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))              # offsets
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))                            # default gaussian kernel
    weights = kernel * np.stack([
        x <= 0, x >= 0,                                                     # west and east
        y <= 0, y >= 0,                                                     # south and north
        (x <= 0) & (y <= 0),                                                # north west sub window
        (x >= 0) & (y <= 0),                                                # north east sub window
        (x <= 0) & (y >= 0),                                                # south west sub window
        (x >= 0) & (y >= 0),                                                # south east sub window
    ], axis=0)                                                              # all the sliding window weights (8, d, d)
    weights /= weights.sum(axis=(-2, -1), keepdims=True)                    # normalized weights

    n = np.sum(S1)
    vals = np.sum(patches[S1][:, None, ...] * weights[None, ...], axis=(-2, -1))    # swf all possible outputs
    idxs = np.argmin(np.abs(vals - denoised_img[S1][:, None]), axis=1)              # indexs of minimum sub sliding windows (n, )
    denoised_img[S1] = vals[np.arange(n), idxs]

    # 4. apply attenuation for S2
    denoised_img[S2] = denoised_img[S2] * Atten
    
    # 5. apply dilation operator
    denoised_img = morph_filter(denoised_img, size=d, method='Dilate', selem='Disk')

    return denoised_img


def denoise_with_cwm(img: np.ndarray, size: int=3):
    '''
        Denoise with combined wavelet transform and morphology.
        https://doi.org/10.27060/d.cnki.ghbcu.2020.001632
    '''
    img1 = denoise_with_wavelet(img, 'sym4', level=4, threshold=40)
    img2 = morph_filter(morph_filter(img, size, 'Open'), size, 'Close')

    frac = 0.85
    denoised_img = frac * img1 + (1 - frac) * img2

    return denoised_img


def denoise_with_cnb(img: np.ndarray, size: int, wind: int=7, sigma: int=10, sigma_g: int=20, sigma_s: int=20, sigma_i: int=20, sigma_j: int=20, count: int=5, trim: int=3):
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
    
    def compute_blf_weights(g: np.ndarray, p: np.ndarray, k: int, f: bool=False):
        '''
            Compute bilateral filter weights.
        '''
        gwt = np.exp(-(g[..., None, None] - p) ** 2 / (2 * sigma_g ** 2))       # gray weights

        x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
        swt = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))                   # spatial weights

        ad = np.abs(g[..., None, None] - p).reshape(h, w, -1)                   # absolute differences between center pixels and other pixels
        ead = np.exp(-ad ** 2 / (2 * sigma_i ** 2))                             # exponential absolute differences
        road = np.sort(ad, axis=-1)                                             # rank ordered absolute differences
        stroad = np.sum(road[..., :k], axis=-1)                                 # sum of trimmed rank ordered absolute differences

        if f:                                                                   # calculate impulse weights with ead
            iwt = np.minimum((np.sum(ead, axis=-1) - 1) / k, 1)                 
        else:                                                                   # calculate impulse weights with exponential stroad
            iwt = np.exp(-stroad ** 2 / (2 * sigma_i ** 2))     
        iwt = np.lib.stride_tricks.sliding_window_view(                         # impulse weights (h, w) -> (h, w, d, d)
            np.pad(iwt, ((r, r), (r, r))), 
            (d, d)
        )
        
        stroadp = np.lib.stride_tricks.sliding_window_view(                     # stroad patches
            np.pad(stroad, ((r, r), (r, r))), 
            (d, d)
        )
        jcoef = 1 - np.exp(-(stroad[..., None, None] + stroadp) ** 2 / (8 * sigma_j ** 2))  # joint modulation coefficient between gray weights and impulse weights
        
        if f:
            wt = swt * gwt * iwt                                                # bilateral weights (h, w, d, d)
        else:
            wt = swt * gwt ** (1 - jcoef) * iwt ** jcoef                        # bilateral weights (h, w, d, d)
        wt = wt / np.maximum(np.sum(wt, axis=(-1, -2), keepdims=True), eps)    # normalize the weights

        # check double star situation
        # max_val = 255 if img.dtype == np.uint8 else 1.0 
        # coords = np.column_stack(np.where(g == max_val))
        # coord, _ = find_close_pair(coords, 4)
        # x, y = coord
        # print(x, y, p[x, y])
        # print(iwt[x, y], gwt[x, y], wt[x, y])

        return wt

    ## 1. Prepare data
    h, w = img.shape
    d = size                                    # the diameter of image patch
    k = wind                                    # the diameter of search window
    r = d // 2                                  # the radius of image patch

    denoised_img = np.copy(img)                                                     # denoised image
    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant')                     # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))          # patches (h, w, d, d)

    tmmap = np.mean(np.sort(patches.reshape(h, w, -1))[..., trim:-trim], axis=-1)   # trimmed mean map
    mean = np.mean(img)                                                             # mean
    devi = np.std(img)                                                              # standard deviation

    ## 2. Segment image
    ot_mask = (img == ndi.maximum_filter(img, size=d)) \
           & (np.abs(img-tmmap) > 15*devi) \
           & (np.abs(tmmap-mean) < 1.5*devi)                                        # outlier mask / peper noise
    fg_mask = (~ot_mask) & (img >= mean+1.1*devi)                                   # foreground mask / star pixels

    ## 3. Process outlier
    if False: # replace with median
        pot_mask = np.pad(ot_mask, ((r, r), (r, r)), constant_values=True)          # padded outlier mask
        got_mask = np.lib.stride_tricks.sliding_window_view(
            pot_mask, (d, d)
        )[ot_mask]                                                                  # grouped outlier mask  
        denoised_img[ot_mask] = np.nanmedian(
            np.where(got_mask, np.NAN, patches[ot_mask]),  
            axis=(1, 2)
        )
    else: # replace with trimmed mean
        denoised_img[ot_mask] = tmmap[ot_mask]

    ## 4. Process star pixels with NLM
    fimg = img.reshape(-1)                                      # flatten image
    fpatches = patches.reshape(-1, d**2)                        # flatten patches (h*w, d*d)
    spatches = patches[fg_mask].reshape(-1, d**2)               # flatten star patches (n, d*d)
    indexs = preselect_similar(tmmap, devi)[fg_mask].reshape(-1, k*k) # similar patch indexs (n, k*k)
    weights = compute_nlm_weights(fpatches, spatches, indexs)   # nlm weights (n, k*k)
    denoised_img[fg_mask] = np.sum(fimg[indexs] * weights, axis=-1)

    ## 5. Process outliers and non-star pixels with BLF
    padded_img[r:-r, r:-r] = denoised_img
    weights = compute_blf_weights(denoised_img, patches, count)
    denoised_img = np.sum(weights * patches, axis=(-1, -2))

    return denoised_img


def denoise_image(img: np.ndarray, method: str):
    '''
        Denoise the image.
    '''
    # maximum intensity
    max_val = 255 if img.dtype == np.uint8 else 1.0

    if method == 'CNB': # combined nlm and blf
        # when the deviation of the guassian noise is lower than 0.6
        denoised_img = denoise_with_cnb(img, 7, 17, sigma=0.05*max_val, sigma_g=0.15*max_val, sigma_s=3, sigma_i=0.8*max_val, sigma_j=0.6*max_val, count=7, trim=3)
    
        # when the deviation of the guassian noise is higher than 0.8
        # denoised_img = denoise_with_cnb(img, 7, 17, sigma=0.03*max_val, sigma_g=0.1*max_val, sigma_s=3, sigma_i=0.4*max_val, sigma_j=0.3*max_val, count=7, trim=3)
    elif method == 'CWM':
        denoised_img = denoise_with_cwm(img, size=5)
    elif method == 'CMG':
        denoised_img = denoise_with_cmg(img, size=5, sigma=2)
    elif method == 'AMF':
        denoised_img = denoise_with_amf(img, size1=3, size2=11)
    elif method == 'EMF':
        denoised_img = denoise_with_emf(img, size=5, threshold=10)
    elif method == 'NLM_BLF':
        denoised_img = denoise_with_nlm(img, patch_size=3, wind_size=11, sigma=0.05*max_val)
        denoised_img = denoise_with_blf(denoised_img, size=3, sigma_g=0.05*max_val, sigma_s=1, threshold=0.4*max_val)
    elif method == 'NLM': # non local mean
        denoised_img = denoise_with_nlm(img, 5, 21, sigma=0.1*max_val)
    elif method == 'MBLF': # modified bilateral filter
        denoised_img = denoise_with_blf(img, 7, sigma_g=0.05*max_val, sigma_s=1.5, threshold=0.6*max_val)
    elif method == 'Wavelet':
        denoised_img = denoise_with_wavelet(img, 'db4', threshold=10)
    elif method in ['Mean', 'Gaussian', 'Median', 'Bilateral', 'GLF']:
        denoised_img = basic_filter(img, method)
    else:
        denoised_img = img

    # clip and astype
    denoised_img = np.clip(denoised_img, 0, max_val).astype(img.dtype)

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
