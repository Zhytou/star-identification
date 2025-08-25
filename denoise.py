import cv2
import numpy as np
from scipy.signal import convolve2d


def filter_image(img: np.ndarray, method: str='GAUSSIAN', size: int=3, sigma: float=0.5) -> np.ndarray:
    '''
        Conventional noise reducing filters.
    Args:
        img: the image to be processed
        method: the method of filtering
    Returns:
        filtered_img: the image after filtering
    '''
    if method == 'GAUSSIAN':
        filtered_img = cv2.GaussianBlur(img, (size, size), sigma)
    elif method == 'MEAN':
        filtered_img = cv2.blur(img, (size, size))
    elif method == 'MEDIAN':
        # d = size//2
        # padded_img = np.pad(img, ((d, d), (d, d)), mode='constant')
        # filtered_img = cv2.medianBlur(padded_img, size)
        # filtered_img = filtered_img[d:-d, d:-d]
        filtered_img = cv2.medianBlur(img, size)
    elif method == 'BLF':
        filtered_img = cv2.bilateralFilter(img, size, 30, sigma)
    elif method == 'GLF':
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        kernel = cv2.getGaussianKernel(size, sigma)
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


def denoise_with_emf(img: np.ndarray, half_size: int=2):
    '''
        Denoise with the extreme median filter.
    '''
    h, w = img.shape

    max_img = morph_filter(img, 'max')
    min_img = morph_filter(img, 'min')

    is_extreme = (img == max_img) | (img == min_img)
    coords = np.transpose(np.nonzero(is_extreme)) # n x 2

    # 3x3 window check 
    offset3 = np.array(
        [[-1, -1],
         [-1, 0],
         [-1, 1],
         [0, 1],
         [1, 1],
         [1, 0],
         [1, -1],
         [0, -1]]
    )
    coords3 = coords + offset3[None, :] # n x 8 x 2
    coords3[:, :, 0] < h and coords3[:, :, 1] < w


def denoise_with_cnb(img: np.ndarray, patch_size: int=3, match_num: int=49, sigma: int=10, threshold: int=150, atten: float=0.1, sigma_color: int=50, sigma_space: int=20):
    '''
        Denoise with combined nlm and blf.
    '''

    def compute_ssd(patches: np.ndarray, sim_patches: np.ndarray):
        '''
            Compute the sum of squared differences.
        '''
        sq = np.sum(star_patches**2, axis=(1,2))          # sum of squares (n,)
        sim_sq = np.sum(sim_patches**2, axis=(2,3))       # sum of squares (n, k)
        dot_product = np.einsum('nij, nkij->nk', patches, sim_patches)  # dot product
        ssd = sq[:, None] + sim_sq - 2 * dot_product

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
    T = np.percentile(mean_map, 98) 
    mask = mean_map > T
    print('Bound:', T, 'NLM count:', np.sum(mask))

    coords = np.argwhere(mask)                                      # the center coordinates of possible star patches
    avg_grays = mean_map[coords[:, 0], coords[:, 1]]                # the average gray values of selected pixels (n, )
    diff = np.abs(avg_grays[:, None] - avg_grays[None, :])          # the differences between pixels (n, n)
    topk_idxs = np.argpartition(diff, kth=k, axis=1)[:, :k]         # the indexs of top kth similar pixel (n, k)
    sim_coords = coords[topk_idxs]                                  # the center coordinates of similar patches

    # 3. Do non local mean with the selected patches
    sim_grays = img[sim_coords[..., 0], sim_coords[..., 1]]             # the gray values of similar pixels (n, k)
    star_patches = patches[coords[:, 0], coords[:, 1]]                  # the possible star patches (n, d, d)
    sim_patches = patches[sim_coords[..., 0], sim_coords[..., 1]]       # the similar patches of each possible star patch (n, k, d, d)

    ## calculate the nlm weights
    ssds = compute_ssd(star_patches, sim_patches)                       # the sum of squared differences (n, k)
    
    nlm_weights = np.exp(-ssds / (sigma **2))  
    nlm_weightsum = np.sum(nlm_weights, axis=-1)

    ## apply non local mean
    denoised_img[coords[:, 0], coords[:, 1]] = (nlm_weights * sim_grays).sum(axis=-1) / nlm_weightsum

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
