import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from skimage.metrics import structural_similarity


def convert_ra_dec(ra: float, dec: float):
    '''
        Convert the RA and DEC from degree to timezone.
    Args:
        ra: the right ascension in degree
        dec: the declination in degree
    Returns:
        ra_rad: the right ascension in hour
        dec_rad: the declination in radians
    '''
    # convert the RA and DEC from degree to radians

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    return coord.to_string('hmsdms')


def draw_gray_3d(img: np.ndarray):
    '''
        Draw the 3D gray image.
    Args:
        img: the image to be processed
    '''
    # get the image size
    h, w = img.shape

    # generate the coordinates
    # x = np.linspace(-w/2, w/2, w)
    x = np.linspace(0, w, w)
    # y = np.linspace(-h/2, h/2, h)
    y = np.linspace(0, h, h)
    X, Y = np.meshgrid(x, y)

    # create 3D image
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, img, cmap='gray')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (gray value)')
    fig.colorbar(surf)
    plt.show()


def draw_freq_spectrum(img: np.ndarray):
    '''
        Draw the frequency spectrum of the image.
    Args:
        img: the image to be processed
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)    
    fdb_img = 20 * np.log(np.abs(fshift))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fdb_img, cmap='gray')
    plt.title('Frequency Spectrum')
    plt.axis('off')
    plt.show()


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
    # caculate the MSE
    mse = np.mean((img - filtered_img)**2)
    
    # caculate the PSNR
    psnr = 10 * np.log10(255**2 / mse)
    
    # caculate the SSIM
    mssim = structural_similarity(img, filtered_img, data_range=255)

    mse, psnr, mssim = round(mse, 2), round(psnr, 2), round(mssim, 2)

    return mse, psnr, mssim