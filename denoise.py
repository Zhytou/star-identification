import cv2
import numpy as np
import pywt


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
    patches = np.lib.stride_tricks.as_strided(padded_img, shape=(h, w, d, d), strides=padded_img.strides + padded_img.strides[:2])
    grays = img[..., np.newaxis, np.newaxis] # the central gray value of each image patch

    # calculate the color and space weights
    color_diffs = custom_activation(np.abs(grays - patches), threshold)
    color_weights = np.exp(-(color_diffs ** 2) / (2 * sigma_color ** 2))

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    space_weights = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    # apply bilateral filter    
    bilateral_weights = color_weights * space_weights
    bilateral_weights = bilateral_weights / bilateral_weights.sum(axis=(-2, -1), keepdims=True)
    filtered_img = (bilateral_weights * patches).sum(axis=(-2, -1))

    # apply the attenuation factor
    offsets = np.array([(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)])
    neighbors = [r, r] + offsets
    weightsum = bilateral_weights[..., neighbors[:, 0], neighbors[:, 1]].sum(axis=-1)
    
    filtered_img = np.where(np.abs(weightsum) < 0.1, np.zeros_like(img), filtered_img).astype(np.uint8)

    return filtered_img


def denoise_with_emf(img: np.ndarray, size: int=3, threshold: int=5):
    '''
        Denoise with the extreme median filter.
    '''
    
    def compute_energy(img0: np.ndarray, img1: np.ndarray,):
        '''
            Compute the energy of each pixel in the image.
        '''
        padded_img = np.pad(img0, r, 'edge')

        left = np.abs(padded_img - np.roll(padded_img,  1, axis=1))     
        right = np.abs(padded_img - np.roll(padded_img, -1, axis=1))    
        up = np.abs(padded_img - np.roll(padded_img,  1, axis=0))       
        down = np.abs(padded_img - np.roll(padded_img, -1, axis=0))     

        energy = (left + right + up + down)[1:-1, 1:-1]
        energy += np.abs(img1 - img0)

        return energy

    h, w = img.shape
    d = size
    r = d // 2

    denoised_img = img
    mean_map = cv2.blur(img, (3, 3))
    median_map = cv2.medianBlur(img, 3)

    # 1.initial check with extreme values
    max_img = morph_filter(img, 'max', size=size)
    min_img = morph_filter(img, 'min', size=size)
    mask = (np.abs(img - max_img)  < threshold) & (np.abs(img - min_img)  < threshold)
    denoised_img[mask] = median_map[mask]

    # 2.double check with energy
    e0 = compute_energy(img, img)
    e1 = compute_energy(img, denoised_img)
    mask = e1 < e0 & np.sum(e0) / (h*w) < e0

    # 3.predicate
    grays = img[mask]
    coords = np.argwhere(mask)
    offsets = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])


    return denoised_img
    

def denoise_with_cwm(img: np.ndarray, size: int=3):
    '''
        Denoise with combined wavelet transform and morphology.
    '''
    denoised_img = denoise_with_wavelet(img, 'sym4')

    img = morph_filter(img, 'open', cv2.MORPH_ELLIPSE, size)
    img = morph_filter(img, 'close', cv2.MORPH_ELLIPSE, size)

    denoised_img = cv2.addWeighted(denoised_img, 0.5, img, 0.5, 0)

    return denoised_img


def denoise_with_cnb(img: np.ndarray, patch_size: int=3, sigma: int=10, size: int=5, sigma_color: int=20, sigma_space: int=20, threshold: int=50):
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

    # 1. Prepare data
    d = patch_size                                  # the diameter of image patch
    r = patch_size // 2                             # the radius of image patch
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

    grays = img[mask]                                               # the gray values of selected pixels (n, k)
    coords = np.argwhere(mask)                                      # the center coordinates of possible star patches
    star_patches = patches[coords[:, 0], coords[:, 1]]              # the possible star patches (n, d, d)
    ssds = compute_ssd(star_patches)                                # the sum of squared differences (n, k)
    
    # 3. Do non local mean with the selected patches
    ## calculate the nlm weights
    nlm_weights = np.exp(-ssds / (sigma **2))  
    nlm_weightsum = np.sum(nlm_weights, axis=-1)

    ## apply non local mean
    denoised_img[coords[:, 0], coords[:, 1]] = (nlm_weights * grays).sum(axis=-1) / nlm_weightsum

    # 4. Do modified bilateral filter with other patches
    coords = np.argwhere(~mask)
    denoised_img[coords[:, 0], coords[:, 1]] = denoise_with_blf(img, size, sigma_color, sigma_space, threshold)[coords[:, 0], coords[:, 1]]

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
        denoised_img = denoise_with_blf(img, 5, sigma_color=20, sigma_space=1)
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
