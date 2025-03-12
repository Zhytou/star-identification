import cv2
import pywt
import numpy as np
from scipy.signal import convolve2d


def filter_image(img: np.ndarray, method: str='gaussian', size: int=3, sigma: float=0.5) -> np.ndarray:
    '''
        Conventional noise reducing filters.
    Args:
        img: the image to be processed
        method: the method of filtering
    Returns:
        filtered_img: the image after filtering
    '''
    if method == 'gaussian':
        filtered_img = cv2.GaussianBlur(img, (size, size), sigma)
    elif method == 'mean':
        filtered_img = cv2.blur(img, (size, size))
    elif method == 'median':
        filtered_img = cv2.medianBlur(img, size)
    elif method == 'gaussian low pass':
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
        print('Invalid method')
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


def denoise_with_wavelet(img: np.ndarray, wavelet='sym4', thr_method='bayes'):
    '''
        Wavelet denoising.
    '''

    def cal_threshold(coeffs, noise_std):
        '''
            Calculate the threshold for wavelet denoising.
        '''

        # use the highest level subband to estimate noise standard deviation
        noise_std = np.median(np.abs(coeffs[1][-1])) / 0.6745
        
        if thr_method == 'visu':
            threshold = noise_std * np.sqrt(2 * np.log(len(coeffs)))
        elif thr_method == 'bayes':    
            threshold = (noise_std**2) / np.sqrt(np.var(coeffs) + 1e-6)
        elif thr_method == 'sure':
            threshold = np.sqrt(2 * np.log(len(coeffs))) * noise_std
        else:
            threshold = 0
        
        print(threshold)

        return threshold

    # wavelet decomposition
    coeffs = pywt.wavedec2(img, wavelet, level=2)

    # iterate through the wavelet coefficients, and skip the lowest frequency subband
    for i in range(1, len(coeffs)):
        tr_coeff = pywt.threshold(coeffs[i], 0, mode='soft')
        coeffs[i] = (tr_coeff[0], tr_coeff[1], tr_coeff[2])

    denoised_img = pywt.waverec2(coeffs, wavelet)
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

    return denoised_img


def denoise_image(img: np.ndarray, method: str='nlm'):
    '''
        Denoise the image.
    '''
    denoised_img = filter_image(img, method)
    if denoised_img is not None:
        return denoised_img
    elif method == 'nlm':
        return denoise_with_nlm(img)
    elif method == 'wavelet':
        return denoise_with_wavelet(img)
    else:
        return None


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


