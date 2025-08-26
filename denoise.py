import cv2
import numpy as np
import pywt
from scipy.signal import convolve2d


def filter_image(img: np.ndarray, method: str='GAUSSIAN', size: int=3) -> np.ndarray:
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
        filtered_img = cv2.bilateralFilter(img, 7, 10, 0.7)
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


def denoise_with_mle(y: np.ndarray, psf: np.ndarray, noise_std: float, max_iter: int=100, tol: float=1e-6):
    """
        Maximum likelihood estimation denoising.
    Args:
        y: noised image
        psf: point spread function
        noise_std: noise standard deviation
        max_iter: max iteration
        tol: convergence threshold
    Returns:
        x: denoised image
    """
    x = y.copy().astype(float)
    
    for k in range(max_iter):
        x_conv = convolve2d(x, psf, mode='same', boundary='symm')
        residual = y - x_conv        
        x_new = x + (1 / noise_std**2) * convolve2d(residual, psf[::-1, ::-1], mode='same', boundary='symm')
        diff = np.linalg.norm(x_new - x) / np.linalg.norm(x)
        if diff < tol:
            print(k)
            break
        x = x_new
    
    return x.astype(np.uint8)


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


def denoise_with_wavelet(img, wavelet: str='sym4', level: int=3, threshold: float=2):
    '''
        Denoise with wavelet.
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


def denoise_with_blf(img: np.ndarray, size: int = 9, atten: float = 0.1, threshold: int = 150, sigma_color: float = 30, sigma_space: float = 1):
    '''
        Improved bilateral filter denoising.
    Args:
        img: the image to be processed
        size: the size of template
        atten: the attenuation factor
        threshold: the threshold
        sigma_color: the standard deviation of the color space
        sigma_space: the standard deviation of the coordinate space
    Returns:
        filtered_img: the image after filtering
    '''

    def custom_activation(x, threshold):
        '''
            Custom activation function.
        Args:
            x: the input value
            threshold: the threshold
        Returns:
            y: the output value
        '''
        return np.where(x > threshold, np.inf, x)

    h, w = img.shape
    d = size
    if d % 2 == 0:
        d = d + 1
    r = d // 2

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    space_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    # pad the image for color weight calculation and change the type in case negative overflow
    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant').astype(np.int16)

    # use a strided view to get the neighborhoods(h, w, d, d)
    neighborhoods = np.lib.stride_tricks.as_strided(padded_img, shape=(h, w, d, d), strides=padded_img.strides + padded_img.strides[:2])
    center_pixels = img[..., np.newaxis, np.newaxis]

    # calculate the color difference
    color_diff = custom_activation(np.abs(center_pixels - neighborhoods), threshold)
    color_kernel = np.exp(-(color_diff ** 2) / (2 * sigma_color ** 2))

    # calculate the bilateral weight
    bilateral_kernel = color_kernel * space_kernel
    weight_sum = bilateral_kernel.sum(axis=(-2, -1))
    center_weight = bilateral_kernel[..., r, r]

    # calculate the filtered image
    filtered_img = (bilateral_kernel * neighborhoods).sum(axis=(-2, -1)) / weight_sum

    # apply the attenuation factor
    filtered_img = np.where(weight_sum == center_weight, center_pixels.squeeze() * atten, filtered_img).astype(np.uint8)

    return filtered_img


def denoise_with_emf(img: np.ndarray):
    '''
        Denoise with the extreme median filter.
    '''
    h, w = img.shape

    denoised_img = img
    mean_map = cv2.blur(img, (3, 3))
    median_map = cv2.medianBlur(img, 3)

    max_img3 = morph_filter(img, 'max', size=3)
    min_img3 = morph_filter(img, 'min', size=3)
    is_extreme1 = (img == max_img3) | (img == min_img3)
    ds = [-1, 0, 1]
    xs, ys = np.meshgrid(ds, ds, indexing='ij')
    offsets3 = np.stack([xs.flatten(), ys.flatten()], axis=1)

    max_img5 = morph_filter(img, 'max', size=5)
    min_img5 = morph_filter(img, 'min', size=5)
    is_extreme2 = (img == max_img5) | (img == min_img5)
    ds = [-2, -1, 0, 1, 2]
    xs, ys = np.meshgrid(ds, ds, indexing='ij')
    offsets5 = np.stack([xs.flatten(), ys.flatten()], axis=1)
    
    ## first step
    patches = np.argwhere(is_extreme1)[:, None, :] + offsets3                       # coordinates of patches (n, 8, 2)
    xs, ys = np.clip(patches[..., 0], 0, h-1), np.clip(patches[..., 1], 0, w-1)     # clipped coordinates
    count = np.sum(is_extreme1[xs, ys], axis=1)                                     # count of extreme pixels in each patch(n, )
    mask1 = count < 9                                                               # not all pixel are extreme values
    coords = np.argwhere(is_extreme1)[mask1]
    denoised_img[coords[:, 0], coords[:, 1]] = median_map[coords[:, 0], coords[:, 1]]

    ## second step
    coords = np.argwhere(is_extreme1)[~mask1]
    patches = coords[:, None, :] + offsets5
    xs, ys = np.clip(patches[..., 0], 0, h-1), np.clip(patches[..., 1], 0, w-1)
    count = np.sum(is_extreme2[xs, ys], axis=1)   
    mask2 = count == 25
    coords = coords[mask2]
    denoised_img[coords[:, 0], coords[:, 1]] = mean_map[coords[:, 0], coords[:, 1]]

    return denoised_img
    

def denoise_with_cwm(img: np.ndarray, size: int=3):
    '''
        Denoise with combined wavelet transform and morphology.
    '''
    denoised_img = denoise_with_wavelet(img)

    img = morph_filter(img, 'open', cv2.MORPH_ELLIPSE, size)
    img = morph_filter(img, 'close', cv2.MORPH_ELLIPSE, size)

    denoised_img = cv2.addWeighted(denoised_img, 0.5, img, 0.5, 0)

    return denoised_img


def denoise_with_cnb(img: np.ndarray, patch_size: int=3, match_num: int=49, sigma: int=10, threshold: int=50, atten: float=0.1, sigma_color: int=50, sigma_space: int=20):
    '''
        Denoise with combined nlm and blf.
    '''

    def compute_ssd(patches: np.ndarray):
        '''
            Compute the sum of squared differences.
        '''
        n = patches.shape[0]            # number of patches
        pf = patches.reshape(n, -1)     # flatten patches
        
        sq = np.sum(pf**2, axis=1)      # sum of squares (n,)
        dp = pf @ pf.T                  # dot product (n, n)
        ssd = sq[:, None] + sq[None, :] - 2 * dp

        return ssd        

    def custom_activation(x, threshold):
        '''
            Custom activation function.
        '''
        return np.where(x > threshold, np.inf, x)

    # 1. Prepare data
    d = patch_size                                  # the diameter of image patch
    r = patch_size // 2                             # the radius of image patch
    k = match_num                                   # the number of match patches
    img = img.astype(np.float32)
    denoised_img = img.copy()

    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant')             # padded zero as the border of image
    # integral_sq = cv2.integral(img**2)                                    # square integral map
    mean_map = cv2.medianBlur(img, d)                                       # mean map
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))  

    # 2. Preselect possible star pixels and similar patches for each pixles
    T = np.percentile(mean_map, 99.9) 
    mask = mean_map > T
    print('Bound:', T, 'NLM count:', np.sum(mask))

    coords = np.argwhere(mask)                                      # the center coordinates of possible star patches
    grays = img[coords[:, 0], coords[:, 1]]                         # the gray values of selected pixels (n, k)
    star_patches = patches[coords[:, 0], coords[:, 1]]              # the possible star patches (n, d, d)
    ssds = compute_ssd(star_patches)                                # the sum of squared differences (n, k)
    
    # 3. Do non local mean with the selected patches
    ## calculate the nlm weights
    nlm_weights = np.exp(-ssds / (sigma **2))  
    nlm_weightsum = np.sum(nlm_weights, axis=-1)

    ## apply non local mean
    denoised_img[coords[:, 0], coords[:, 1]] = (nlm_weights * grays).sum(axis=-1) / nlm_weightsum

    # 4. Do modified bilateral filter with other patches
    coords = np.argwhere(~mask)                                         # the center coordinates of unselected patches
    grays = img[coords[:, 0], coords[:, 1], np.newaxis, np.newaxis]     # the gray values of unselected pixels
    other_patches = patches[coords[:, 0], coords[:, 1]]                 # the unselected patches
    
    ## calculate the blf weights
    color_diffs = custom_activation(np.abs(grays - other_patches), threshold)
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    blf_weights = np.exp(-(color_diffs ** 2) / (2 * sigma_color ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))
    blf_weightsum = blf_weights.sum(axis=(-2, -1))

    ## apply modified blf
    denoised_img[coords[:, 0], coords[:, 1]] = (blf_weights * other_patches).sum(axis=(-2, -1)) / blf_weightsum
    denoised_img[coords[:, 0], coords[:, 1]] = np.where(blf_weightsum == blf_weights[..., r, r], grays.squeeze() * atten, denoised_img[coords[:, 0], coords[:, 1]])

    return denoised_img.astype(np.uint8)


def denoise_image(img: np.ndarray, method: str):
    '''
        Denoise the image.
    '''
    if method == 'CNB': # combined nlm and blf
        denoised_img = denoise_with_cnb(img)
    elif method == 'CWM':
        denoised_img = denoise_with_cwm(img)
    elif method == 'EMF':
        denoised_img = denoise_with_emf(img)
    elif method == 'NLM_BLF':
        denoised_img = denoise_with_nlm(img, 3, 11) # cv2.addWeighted(denoise_with_nlm(img, 5, 11), 0.5, img, 0.5, 0)
        denoised_img = denoise_with_blf(denoised_img, 3, 0.1, sigma_color=10)
    elif method == 'NLM': # non local mean
        denoised_img = denoise_with_nlm(img, 7, 49)
    elif method == 'MBLF': # modified bilateral filter
        denoised_img = denoise_with_blf(img, 3, 0.1, sigma_color=10)
    elif method in ['MEAN', 'GAUSSIAN', 'MEDIAN', 'BLF', 'GLF']:
        denoised_img = filter_image(img, method)
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
