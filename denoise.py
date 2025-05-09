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
        d = size//2
        padded_img = np.pad(img, ((d, d), (d, d)), mode='constant')
        filtered_img = cv2.medianBlur(padded_img, size)
        filtered_img = filtered_img[d:-d, d:-d]
    elif method == 'GLP':
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


def denoise_with_nlm(img: np.ndarray, h: int=10, K: int=7, L: int=21):
    '''
        Non-local means denoising.
    Args:
        img: the image to be processed
        h: the parameter regulating filter strength
        K: the size of the template window
        L: the size of the search window
    Returns:
        denoised_img: the image after filtering
    '''
    denoised_img = cv2.fastNlMeansDenoising(img, None, h, K, L)

    return denoised_img


def denoise_with_blf(img: np.ndarray, d: int=9, atten: float=0.1, threshold: int=150, sigma_color: float=30, sigma_space: float=1):
    '''
        Improved bilateral filter denoising.
    Args:
        img: the image to be processed
        d: the diameter of the pixel neighborhood
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
    filtered_img = np.zeros_like(img)

    if d % 2 == 0:
        d = d + 1
    r = d // 2

    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    space_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_space ** 2))

    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant')
    for y in range(h):
        for x in range(w):
            # get the neighborhood
            neighborhood = padded_img[y:y+d, x:x+d].astype(np.int16)
            center_pixel = img[y, x].astype(np.int16)

            # calculate the color difference
            color_diff = custom_activation(np.abs(center_pixel - neighborhood), threshold)
            color_kernel = np.exp(-(color_diff ** 2) / (2 * sigma_color ** 2))

            # calculate the bilateral weight
            bilateral_weight = color_kernel * space_kernel

            # set the center as attenuation factor
            if np.sum(bilateral_weight) == bilateral_weight[r, r]:
                filtered_img[y, x] = center_pixel * atten
            else:
                filtered_img[y, x] = (bilateral_weight * neighborhood).sum() / bilateral_weight.sum()

    return filtered_img.astype(np.uint8)


def denoise_with_blf_new(img: np.ndarray, d: int = 9, atten: float = 0.1, threshold: int = 150, sigma_color: float = 30, sigma_space: float = 1):
    '''
        Improved bilateral filter denoising.
    Args:
        img: the image to be processed
        d: the diameter of the pixel neighborhood
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


def denoise_with_star_distri(img: np.ndarray, half_size: int=2):
    '''
        Get the seed coordinates with the star distribution.
    '''
    # filter operation
    max_img = morph_filter(img, 'max', cv2.MORPH_RECT, half_size*2+1)
    min_img = morph_filter(img, 'min', cv2.MORPH_ELLIPSE, 5)
    open_img = morph_filter(img, 'open', cv2.MORPH_ELLIPSE, 3)
    
    # get local max pixels
    mask1 = (img == max_img).astype(np.uint8)
    coords1 = np.transpose(np.nonzero(mask1))

    # potential star pixels
    T = np.mean(open_img) + 2*np.std(open_img)
    _, mask2 = cv2.threshold(open_img, T, 255, cv2.THRESH_BINARY)
    coords2 = np.transpose(np.nonzero(mask2))

    # intersection
    star_coords = np.array(list((set(map(tuple, coords1)) & set(map(tuple, coords2)))))

    # iterate through the local max pixels
    denoised_img = min_img
    for row, col in star_coords:
        denoised_img[row-half_size:row+half_size+1, col-half_size:col+half_size+1] = img[row-half_size:row+half_size+1, col-half_size:col+half_size+1]
    
    return denoised_img


def denoise_with_multi_scale_nlm(img: np.ndarray, levels: int=3, h: int=10, K: int=7, L: int=21):
    '''
        Multi-scale NLM denoising.
    Args:
        img: the image to be processed
        h: the parameter regulating filter strength
        K: the size of the template window
        L: the size of the search window
        levels: the number of levels of the pyramid
    '''
    pyramid = gen_laplacian_pyramid(img, levels)
    pyramid = pyramid[::-1]

    for i in range(len(pyramid)):
        if i == 0:
            denoised_img = pyramid[i]
        else:
            denoised_img = cv2.pyrUp(denoised_img) + pyramid[i]
        denoised_img = denoise_with_nlm(denoised_img, h, K, L)

    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

    return denoised_img


def denoise_image(img: np.ndarray, method: str):
    '''
        Denoise the image.
    '''

    if method == 'NLM_BLF':
        denoised_img = cv2.addWeighted(denoise_with_nlm(img, 10, 7, 49), 0.5, denoise_with_nlm(img, 10, 5, 25), 0.5, 0)
        denoised_img = denoise_with_blf_new(denoised_img, 3, 0.1, sigma_color=10)
    elif method == 'NLM':
        denoised_img = denoise_with_nlm(img, 10, 7, 49)
    elif method == 'BLF':
        denoised_img = denoise_with_blf_new(img, 3, 0.1, sigma_color=10)
    else:
        denoised_img = filter_image(img, method)
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
