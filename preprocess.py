import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from math import radians

from simulate import create_star_image, add_stary_light_noise, ROI


def draw_gray_3d(img: np.ndarray):
    '''
        Draw the 3D gray image.
    Args:
        img: the image to be processed
    '''
    # get the image size
    h, w = img.shape

    # generate the coordinates
    x = np.linspace(-w/2, w/2, w)
    y = np.linspace(-h/2, h/2, h)
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


def filter_image(img: np.ndarray, method='gaussian') -> np.ndarray:
    '''
        Use gaussian filter to reduce the noise in the image.
    Args:
        img: the image to be processed
    Returns:
        filtered_img: the image after filtering
    '''
    if method == 'gaussian':
        # low pass filter for noise
        filtered_img = cv2.GaussianBlur(img, (2*ROI+1, 2*ROI+1), 1)
    elif method == 'median':
        filtered_img = cv2.medianBlur(img, 3)
    elif method == 'wavelet':
        cA, (cH, cV, cD) = pywt.dwt2(img, 'sym8')
        filtered_img = pywt.idwt2((cA, (cH, cV, cD)), 'sym8')
    else:
        print('wrong filter method!')
        return None
    
    return filtered_img


def cal_mse_psnr(img: np.ndarray, filtered_img: np.ndarray):
    '''
        Calculate the mean square error and peak signal-to-noise ratio between the original image and the filtered image.
    Args:
        img: the image to be processed
        filtered_img: the image after filtering
    Returns:
        mse: the mean square error between the original image and the filtered image
        psnr: the peak signal-to-noise ratio between the original image and the filtered image
    '''
    # get the image size
    h, w = img.shape

    # caculate the MSE
    mse = np.sum((img - filtered_img)**2) / (h * w)
    
    # caculate the PSNR
    psnr = 10 * np.log10(255**2 / mse)
    
    return mse, psnr


def cal_multiwind_threshold(img: np.ndarray, wind_len: int=40, num_wind: int=20) -> int:
    """
        Calculate the threshold of the image using the method "multi-window threshold division" from https://ieeexplore.ieee.org/abstract/document/1008988.
    Args:
        wind_len: the length of the window
        num_wind: the number of the windows
    Returns:
        threshold: the threshold of the image
    """        
    # initialize random windows
    threshold = 0

    # get the image size
    h, w = img.shape

    for i in range(num_wind):
        x = np.random.randint(0, w - wind_len)
        y = np.random.randint(0, h - wind_len)

        wind = img[y:y+wind_len, x:x+wind_len]    
        mean = np.mean(wind)  
        std = np.std(wind)
        threshold += mean + 5 * std

    return round(threshold/num_wind)


def group_star(img: np.ndarray, threshold: int, connectivity: int) -> list[list[tuple[int, int]]]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        connectivity: method of connectivity
    Returns:
    """
    # if img[u, v] < threshold: 0, else: img[u, v]
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    # if img[u, v] > 0: 1, else: 0
    _, binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

    # label connected regions of the same value in the binary image
    labeled_img, label_num = measure.label(binary_img, return_num=True, connectivity=connectivity)

    group_coords = []
    for label in range(1, label_num + 1):
        # get the coords for each label
        rows, cols = np.nonzero(labeled_img == label)
        # two small to be a star
        if len(rows) < 3 and len(cols) < 3:
            continue
        group_coords.append((rows, cols))

    return group_coords


def get_star_centroids(img: np.ndarray, method: str, wind_size: int=9) -> list[tuple[float, float]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
        method: centroid algorithm
        wind_size: the size of the window used to calculate the centroid
    Returns:
        centroids: the centroids of the stars in the image
    '''

    # get the image size
    h, w = img.shape

    # low pass filter for noise
    filtered_img = filter_image(img)

    # calaculate the threshold
    threshold = cal_multiwind_threshold(filtered_img, wind_len=int(max(h*0.7, w*0.7)), num_wind=10)

    # rough group star using connectivity
    group_coords = group_star(filtered_img, threshold, 2)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for (rows, cols) in group_coords:
        vals = filtered_img[rows, cols]
        # get the brightest pixel
        idx = np.argmax(vals)
        row, col = rows[idx], cols[idx]
        # construct window
        top = max(0, row - wind_size//2)
        bottom = min(h, row + wind_size//2)
        left = max(0, col - wind_size//2)
        right = min(w, col + wind_size//2)
        # calculate the centroid in the window
        row_sum = 0
        col_sum = 0
        gray_sum = 0
        for r in range(top, bottom):
            for c in range(left, right):
                if method == 'default centroid':
                    row_sum += r * filtered_img[r][c]
                    col_sum += c * filtered_img[r][c]
                    gray_sum += filtered_img[r][c]
                elif method == 'square centroid':
                    row_sum += r * pow(filtered_img[r][c], 2)
                    col_sum += c * pow(filtered_img[r][c], 2)
                    gray_sum += pow(filtered_img[r][c], 2)
                elif method == 'centroid with threshold':
                    row_sum += r * (filtered_img[r][c] - threshold)
                    col_sum += c * (filtered_img[r][c] - threshold)
                    gray_sum += filtered_img[r][c] - threshold
                else:
                    print('wrong centroid method!')
                    return []

        centroids.append((round(row_sum/gray_sum, 3), round(col_sum/gray_sum, 3)))

    return centroids


if __name__ == '__main__':
    filter_test = False
    segment_test = False
    centroid_test = True

    if filter_test:
        ra, de, roll = radians(161.7048), radians(4.2806), radians(0)
        img, _ = create_star_image(ra, de, roll, white_noise_std=0)
        noised_img, _ = create_star_image(ra, de, roll, white_noise_std=10)

        for method in  ['gaussian', 'median', 'wavelet']:
            filtered_img = filter_image(noised_img, method)
            mse, psnr = cal_mse_psnr(img, filtered_img)
            cv2.imwrite(f'filtered_img_{method}.png', filtered_img)
            cv2.imwrite(f'filtered_img_{method}_scale.png', filtered_img[50:200, 350:500])
            print(method, 'mse:', mse, 'psnr:', psnr)

    if segment_test:
        ra, de, roll = radians(161.7048), radians(4.2806), radians(0)
        img, _ = create_star_image(ra, de, roll, white_noise_std=10)
        h, w = img.shape
        filtered_img = filter_image(img)
        threshold = cal_multiwind_threshold(filtered_img, wind_len=int(max(h*0.7, w*0.7)), num_wind=10)
        print(threshold)
        binary_img = cv2.threshold(filtered_img, threshold, 255, cv2.THRESH_TOZERO)[1]
        cv2.imwrite('after_seg.png', binary_img)
        cv2.imwrite('after_seg_scale.png', binary_img[50:200, 350:500])

    if centroid_test:
        white_noise_stds = [10]
        arr_pos_err = {
            'default centroid': [],
            'square centroid': [],
            'centroid with threshold': []
        }
        for white_noise_std in white_noise_stds:
            # times of estimation using centroid algorithm
            num_esti_times = 3
            # random ra & de test
            num_test = 10
            # generate random right ascension[0, 360] and declination[-90, 90]
            ras = np.random.uniform(0, 2*np.pi, num_test)
            des = np.arcsin(np.random.uniform(-1, 1, num_test))
            # centroid position error
            pos_err = {
                'default centroid': 0,
                'square centroid': 0,
                'centroid with threshold': 0
            }
            # generate the star image
            for i in range(num_test):
                img, star_info = create_star_image(ras[i], des[i], 0, white_noise_std=0)
                real_coords = np.array([x[1] for x in star_info])
                # estimate the centroids of the stars in the image
                for method in pos_err.keys():
                    arr_esti_coords = []
                    for _ in range(num_esti_times):
                        esti_coords = np.array(get_star_centroids(img, method))
                        if len(real_coords) != len(esti_coords):
                            print('Wrong star number extracted: ', len(real_coords), len(esti_coords))
                            continue
                        arr_esti_coords.append(esti_coords)
                    if len(arr_esti_coords) == 0:
                        continue
                    avg_esti_coords = np.mean(arr_esti_coords, axis=0)
                    # calculate the position error
                    diff = real_coords[:, None] - avg_esti_coords
                    dist = np.linalg.norm(diff, axis=-1)
                    pos_err[method] += np.sum(np.min(dist, axis=0))

            for method in pos_err.keys():
                arr_pos_err[method].append(pos_err[method]/num_test)
        
        print(arr_pos_err)

        for method in arr_pos_err.keys():
            plt.plot(white_noise_stds, arr_pos_err[method], label=method)
        plt.grid()
        plt.show()
