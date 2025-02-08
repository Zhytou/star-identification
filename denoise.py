import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity
from math import radians

from simulate import create_star_image


def filter_image(img: np.ndarray, method='gaussian', method_args=None) -> np.ndarray:
    '''
        Conventional noise reducing filters.
    Args:
        img: the image to be processed
        method: the method of filtering
        method_args: the arguments of the method
            gaussian: sigma
            bilateral: sigma_s, sigma_c
            gaussian low pass: sigma/cutoff frequency
    Returns:
        filtered_img: the image after filtering
    '''
    if method == 'gaussian':
        filtered_img = cv2.GaussianBlur(img, (3, 3), method_args)
    elif method == 'mean':
        filtered_img = cv2.blur(img, (3, 3))
    elif method == 'median':
        filtered_img = cv2.medianBlur(img, 3)
    elif method == 'bilateral':
        filtered_img = cv2.bilateralFilter(img, 5, *method_args)
    elif method == 'gaussian low pass':
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        kernel = cv2.getGaussianKernel(3, method_args)
        kernel = np.outer(kernel, kernel.transpose())
        kernel_padded = np.pad(kernel, ((0, img.shape[0] - kernel.shape[0]), (0, img.shape[1] - kernel.shape[1])), mode='constant')
        kernel_f = np.fft.fft2(kernel_padded)
        kernel_fshift = np.fft.fftshift(kernel_f)

        filtered_fshift = fshift * kernel_fshift
        filtered_f = np.fft.ifftshift(filtered_fshift)
        filtered_img = np.fft.ifft2(filtered_f)
        filtered_img = np.abs(filtered_img)
    else:
        print('wrong filter method!')
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
        filtered_img: the image after filtering
    '''
    filtered_img = cv2.fastNlMeansDenoising(img, None, h, K, L)

    return filtered_img


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

    print(coeffs[0].shape, coeffs[1][0].shape, coeffs[2][0].shape)

    # iterate through the wavelet coefficients, and skip the lowest frequency subband
    for i in range(1, len(coeffs)):
        tr_coeff = pywt.threshold(coeffs[i], 0, mode='soft')
        coeffs[i] = (tr_coeff[0], tr_coeff[1], tr_coeff[2])

    filtered_img = pywt.waverec2(coeffs, wavelet)
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

    return filtered_img


def denoise(img: np.ndarray):
    '''
        Proposed denoising method.
    '''

    # multi-scale nlm denoising
    pyramid = gen_laplacian_pyramid(img)
    pyramid = pyramid[::-1]

    for i in range(len(pyramid)-1):
        # img = denoise_with_nlm(pyramid[i], 12-2*i, 7, 21)
        img = filter_image(pyramid[i], 'gaussian', 1)
        img = cv2.pyrUp(img) + pyramid[i+1]

    # merge
    img = np.clip(img, 0, 255).astype(np.uint8)

    # img = denoise_with_wavelet(img)    

    return img


def draw_freq_spectrum(imgs: list[np.ndarray]):
    '''
        Draw the frequency spectrum of the image.
    Args:
        imgs: the images to be processed
    '''
    n = len(imgs)
    for i, img in enumerate(imgs):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)    
        fdb = 20 * np.log(np.abs(fshift))

        plt.subplot(n, 2, i*2+1)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        plt.axis('off')
    
        plt.subplot(n, 2, i*2+2)
        plt.imshow(fdb, cmap='gray')
        plt.title('Frequency Spectrum')
        plt.axis('off')
    
    plt.show()


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


def cal_psnr_ssim(img: np.ndarray, filtered_img: np.ndarray):
    '''
        Calculate peak signal-to-noise ratio and the structural similarity between the original image and the filtered image.
    Args:
        img: the original image
        filtered_img: the image after filtering
    Returns:
        psnr: the peak signal-to-noise ratio
        mssim: the mean structural similarity
    '''
    # caculate the MSE
    mse = np.mean((img - filtered_img)**2)
    
    # caculate the PSNR
    psnr = 10 * np.log10(255**2 / mse)
    
    # caculate the SSIM
    mssim = structural_similarity(img, filtered_img, data_range=255)

    return psnr, mssim


if __name__ == '__main__':
    ra, de, roll = radians(29.2104), radians(-12.0386), radians(0)
    original_img, stars = create_star_image(ra, de, roll, sigma_g=0, prob_p=0)

    noised_img, _ = create_star_image(ra, de, roll, sigma_g=0.05, prob_p=0.001)
    real_coords = np.array([star[1] for star in stars])

    # imgs = gen_laplacian_pyramid(noised_img, 3)
    # for i, img in enumerate(imgs):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(img, cmap='gray')
    # plt.show()
    
    # snr
    snr = cal_snr(original_img, noised_img)
    print(snr)

    denoised_imgs = {}
    # conventional filters
    denoised_imgs['mean'] = filter_image(noised_img, 'mean')
    denoised_imgs['median'] = filter_image(noised_img, 'median')
    denoised_imgs['gaussian'] = filter_image(noised_img, 'gaussian', 1)

    # mle
    kernel = cv2.getGaussianKernel(3, 1.0)
    kernel = np.outer(kernel, kernel.transpose())
    denoised_imgs['mle'] = denoise_with_mle(noised_img, kernel, 10)

    # wavelet
    denoised_imgs['wavelet'] = denoise_with_wavelet(noised_img)
    
    # median+nlm
    denoised_imgs['proposed'] = denoise(noised_img)

    # psf nlm
    # denoised_imgs['psf_nlm'] = denoise_with_psf_nlm(noised_img)

    cv2.imwrite('example/star.png', original_img)
    cv2.imwrite('example/n_star.png', noised_img)
    cv2.imwrite('example/m_star.png', denoised_imgs['median'])
    cv2.imwrite('example/g_star.png', denoised_imgs['gaussian'])
    cv2.imwrite('example/w_star.png', denoised_imgs['wavelet'])
    cv2.imwrite('example/p_star.png', denoised_imgs['proposed'])

    print('noised', *cal_psnr_ssim(original_img, noised_img))

    for name, denoised_img in denoised_imgs.items():
        pnr, ssim = cal_psnr_ssim(original_img, denoised_img)
        print(name, pnr, ssim)