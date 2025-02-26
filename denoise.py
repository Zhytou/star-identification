import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from skimage.metrics import structural_similarity
from math import radians

from simulate import create_star_image


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
    elif method == 'max':
        filtered_img = cv2.dilate(img, np.ones((size, size), np.uint8))
    elif method == 'min':
        filtered_img = cv2.erode(img, np.ones((size, size), np.uint8))
    elif method == 'open':
        filtered_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    elif method == 'close':
        filtered_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
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


def denoise_with_star_distri(img: np.ndarray, half_size: int=2):
    '''
        Star distrition based denoising.
    '''
    quarter_size = half_size // 2

    # filter operation
    open_img = filter_image(img, 'open', quarter_size*2+1)
    max_img = filter_image(img, 'max', half_size*2+1)
    
    # get local max pixels
    mask1 = (img == max_img).astype(np.uint8)
    coords1 = np.transpose(np.nonzero(mask1))

    # potential star pixels
    _, mask2 = cv2.threshold(open_img, 10, 255, cv2.THRESH_BINARY)
    coords2 = np.transpose(np.nonzero(mask2))

    cv2.imshow('1', mask1*255)
    cv2.waitKey(-1)

    # intersection
    star_coords = np.array(list((set(map(tuple, coords1)) & set(map(tuple, coords2)))))

    # iterate through the local max pixels
    denoised_img = np.zeros_like(img, dtype=np.uint8)
    for row, col in star_coords:
        denoised_img[row-half_size:row+half_size+1, col-half_size:col+half_size+1] = img[row-half_size:row+half_size+1, col-half_size:col+half_size+1]
    
    return denoised_img, star_coords


def denoise_with_nlm_and_star_distri(img: np.ndarray):
    '''
        Proposed denoising method.
    '''
    # multi patch size nlm denoising
    img1 = denoise_with_nlm(img, 10, 5, 21)
    img2 = denoise_with_nlm(img, 10, 7, 39)

    # merge
    denoised_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # star distribution based denoising
    denoised_img = denoise_with_star_distri(denoised_img, 2)

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
        return denoise_with_nlm_and_star_distri(img)


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
        fdb_img = 20 * np.log(np.abs(fshift))

        plt.subplot(n, 2, i*2+1)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        plt.axis('off')
    
        plt.subplot(n, 2, i*2+2)
        plt.imshow(fdb_img, cmap='gray')
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


def cal_mse_psnr_ssim(img: np.ndarray, filtered_img: np.ndarray):
    '''
        Calculate peak signal-to-noise ratio and the structural similarity between the original image and the filtered image.
    Args:
        img: the original image
        filtered_img: the image after filtering
    Returns:
        psnr: the peak signal-to-noise ratio
        mssim: the mean structural similarity
    '''
    # diff = (img - filtered_img)**2
    # max_val = np.max(diff)
    # rows, cols = np.where(diff == max_val)
    # print(rows, cols)

    # for row, col in zip(rows, cols):
    #     t, b = max(0, row-2), min(img.shape[0], row+3)
    #     l, r = max(0, col-2), min(img.shape[1], col+3)

    #     print(img[t:b, l:r])
    #     print(filtered_img[t:b, l:r])

    # caculate the MSE
    mse = np.mean((img - filtered_img)**2)
    
    # caculate the PSNR
    psnr = 10 * np.log10(255**2 / mse)
    
    # caculate the SSIM
    mssim = structural_similarity(img, filtered_img, data_range=255)

    mse, psnr, mssim = round(mse, 2), round(psnr, 2), round(mssim, 2)

    return mse, psnr, mssim


if __name__ == '__main__':
    imgs = {}

    ra, de, roll = radians(29.2104), radians(-12.0386), radians(0)
    imgs['original'], stars = create_star_image(ra, de, roll, sigma_g=0, prob_p=0)
    imgs['noised'], _ = create_star_image(ra, de, roll, sigma_g=0.05, prob_p=0.001)
    real_coords = np.array([star[1] for star in stars])

    # freq spectrum
    # draw_freq_spectrum([imgs['original'], imgs['noised']])
    # snr
    snr = cal_snr(imgs['original'], imgs['noised'])
    print(snr)

    # pyramid = gen_laplacian_pyramid(imgs['noised'], 3)
    # for i, img in enumerate(pyramid):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(img, cmap='gray')
    # plt.show()
    
    # conventional filters
    imgs['mean'] = filter_image(imgs['noised'], 'mean')
    imgs['median'] = filter_image(imgs['noised'], 'median', size=3)
    imgs['gaussian'] = filter_image(imgs['noised'], 'gaussian', sigma=1)
    # imgs['glp'] = filter_image(imgs['noised'], 'gaussian low pass', sigma=1)

    # multi-scale non-local mean
    # imgs['ms_nlm'] = denoise_with_multi_scale_nlm(imgs['noised'], 3, 10, 7, 21)

    # wavelet
    # imgs['wavelet'] = denoise_with_wavelet(imgs['noised'])
    
    # imgs['distri'] = denoise_with_star_distri(imgs['noised'], half_size=3)

    imgs['proposed'] = denoise_with_nlm_and_star_distri(imgs['noised'])

    for name in imgs:
        if name != 'original' and name != 'noised':
            mse, pnr, ssim = cal_mse_psnr_ssim(imgs['original'], imgs[name])
            print(name, mse, pnr, ssim)
        cv2.imwrite(f'example/star/{name}.png', imgs[name])

        save_scale_image = False
        if save_scale_image:
            # half length
            d = 64
            x, y = 188, 169
            cv2.imwrite(f'example/star/scale_{name}.png', imgs[name][x-d:x+d, y-d:y+d])
        