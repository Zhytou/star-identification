import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from skimage.metrics import structural_similarity


def get_angdist(points: np.ndarray):
    '''
        Get the angular distance of the points.
    '''
    norm = np.linalg.norm(points, axis=1)
    angd = np.dot(points, points.T) / np.outer(norm, norm)

    return angd


def convert_rade2deg(ra: float, dec: float):
    '''
        Convert the RA and DE from degree to timezone.
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


def draw_img_with_id_label(img: np.ndarray, coords: np.ndarray, ids: np.ndarray, grid_on: bool=False, grid_step: int=10, output_path: str=None):
    '''
        Draw the image with the id label.
    '''
    h, w = img.shape[:2]

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray', origin='lower')  
    
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('on')
    ax.invert_yaxis()
    if grid_on:
        ax.set_xticks(np.arange(0, w, grid_step))
        ax.set_yticks(np.arange(0, h, grid_step))
        ax.grid(grid_on, color='r', linewidth=2)

    for id, (row, col) in zip(ids, coords):
        row, col = min(int(row)+5, h-10), min(int(col)+5, w-10)
        ax.text(col, row, str(id), fontsize=8, color='white', backgroundcolor=(0, 0, 0, 0.5), ha='left', va='top')

    plt.show()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=1, dpi=300)
    
    plt.close()


def describe_database(db: pd.DataFrame):
    '''
        Describe the database.
    '''

    db.columns = db.columns.astype(int)
    db_info = np.sum(db.notna().to_numpy(), axis=1)
    max_cnt, min_cnt, avg_cnt = np.max(db_info), np.min(db_info), np.sum(db_info)/len(db)

    print(
        'Max count of 1 in pattern matrix', max_cnt, 
        '\nMin count of 1 in pattern matrix', min_cnt, 
        '\nAvg count of 1 in pattern matrix', avg_cnt
    )

    plt.hist(db_info, bins=max_cnt, edgecolor='black')
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