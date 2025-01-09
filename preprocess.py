import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from math import radians

from simulate import create_star_image, cal_avg_star_num_within_fov, roi


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
        filtered_img = cv2.GaussianBlur(img, (2*roi+1, 2*roi+1), 0.5)
    elif method == 'median':
        filtered_img = cv2.medianBlur(img, 3)
    elif method == 'wavelet':
        cA, (cH, cV, cD) = pywt.dwt2(img, 'sym8')
        filtered_img = pywt.idwt2((cA, (cH, cV, cD)), 'sym8')
    elif method == 'pca':
        pass
    else:
        print('wrong filter method!')
        return None
    
    return filtered_img


def denoise_window_with_pca(img: np.ndarray, center: tuple[int, int], K: int, L: int, epsilon: float=20) -> np.ndarray:
    '''
        Use PCA to reduce the noise in the window of a image.
    Args:
        img: the image to be processed
        center: the center of the window
        K: the size of denoising window
        L: the size of training window
        epsilon: the threshold of the similarity
    Returns:
        filtered_img: the image after filtering
    '''
    # get the image size
    h, w = img.shape

    # get both the target window and training window
    t1, b1, l1, r1 = cal_wind_boundary(center, K, h, w)
    t2, b2, l2, r2 = cal_wind_boundary(center, L, h, w)

    # LPG
    # target block
    target_block = img[t1:b1+1, l1:r1+1].reshape(1, -1).astype(np.float64)
    # get all the blocks similar to the target block in the training window
    similar_blocks = []
    corner = t2, l2
    while corner[0] + K <= b2:
        while corner[1] + K <= r2:
            # calculate the similarity between the target block and the sample block
            if corner[0] == t1 and corner[1] == l1:
                corner = corner[0], corner[1] + 1
                continue

            sample_block = img[corner[0]:corner[0]+K, corner[1]:corner[1]+K].reshape(1, -1).astype(np.float64)
            diff = np.linalg.norm(target_block - sample_block)
            if diff < epsilon * K * K:
                similar_blocks.append(sample_block.reshape(-1))
            corner = corner[0], corner[1] + 1
        corner = corner[0] + 1, l2
    
    if len(similar_blocks) == 0:
        return target_block.reshape(K, K)

    # PCA
    similar_blocks = np.array(similar_blocks).astype(np.float64)
    # each block is a row vector with size equal to K*K, which means the number of features is K*K
    mean = np.mean(similar_blocks, axis=0)
    # centralize the similar blocks
    similar_blocks -= mean
    cov = np.cov(similar_blocks, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # sort the eigenvectors by the absolute value of eigenvalues
    idx = np.argsort(eigvals)
    # select the top 90% eigenvectors
    num = int(0.80 * K * K)
    eigvals, eigvecs = eigvals[idx[-num:]], eigvecs[:, idx[-num:]]

    # project the target block to the eigenvector space
    eigdomain_target_block = np.dot(target_block - mean, eigvecs)
    
    # reconstruct the target block
    denoise_target_block = np.dot(eigdomain_target_block, eigvecs.T) + mean

    #? make sure every pixel is in the range of [0, 255]
    denoise_target_block = np.clip(denoise_target_block, 0, 255)

    return denoise_target_block.reshape(K, K)


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


def cal_threshold(img: np.ndarray, method: str, delta: float=0.1, wind_size: int=5, gray_diff: int=4) -> int:
    """
        Calculate the threshold for image segmentation.
    Args:
        img: the image to be processed
        method: the method used to calculate the threshold
            'Otsu': Otsu thresholding(which minimizes the within-class variances for threshold selection)
                https://ieeexplore.ieee.org/document/4310076/?arnumber=4310076
            'Liebe': adaptive thresholding
                http://ieeexplore.ieee.org/document/1008988/
            'Xu': weighted iterative thresholding
                https://linkinghub.elsevier.com/retrieve/pii/S0030402613002490
            'Abutaleb': automatic thresholding of gray-level pictures using two-dimensional entropy
                https://www.sciencedirect.com/science/article/abs/pii/0734189X89900510?via%3Dihub
            'Xiao': entropic thresholding based on GLSC 2D histogram
                https://ieeexplore.ieee.org/document/4761626/?arnumber=4761626
        delta: scale parameter used for new threshold iterative calculation in 'Xu' method
        wind_size: the size of the window used to calculate the threshold in 'Abutaleb'/'Xiao' method
        gray_diff: the max difference of the gray value to count the similarity in 'Xiao' method
    Returns:
        T: the threshold of the image
    """
    h, w = img.shape

    # initialize threshold
    T = 0

    if method == 'Otsu':
        # use cv2 threshold function to get otsu threshold
        T, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'Liebe':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 3 * std
    elif method == 'Xu':
        #? Xu method is not validated yet
        # get the avg number of fov
        avg_star_num = cal_avg_star_num_within_fov()
        # get the gray distribution of the image(histgram)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.ravel()
        # use 'Liebe' method to get the initial threshold
        T = cal_threshold(img, method='Liebe')
        while True:
            num1, num2 = 0, 0
            denom1, denom2 = 0, 0
            for i in range(256):
                if i < T:
                    num1 += i * hist[i]
                    denom1 += hist[i]
                else:
                    num2 += i * hist[i]
                    denom2 += hist[i]
            # background and foreground mean
            mean1 = num1/denom1 if denom1 != 0 else 0
            mean2 = num2/denom2 if denom2 != 0 else 0
            # get the number of stars
            _, star_num = group_star(img, T, connectivity=2)
            if star_num < 0.5 * avg_star_num:
                T = (1-delta) * mean1 + (1+delta) * mean2
            elif star_num > 1.5 * avg_star_num:
                T = (1+delta) * mean1 + (1-delta) * mean2
            else:
                break
    elif method == 'Abutaleb':
        # average gray level matrix for each pixel's window
        avg_glw = np.zeros_like(img, dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                # window
                t, b, l, r = cal_wind_boundary((i, j), wind_size, h, w)
                wind = img[t:b + 1, l:r + 1]
                # quantize the gray level
                avg_glw[i, j] = round(np.average(wind), 0)
        
        # get the 2d histogram
        hist = np.zeros((256, 256), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                hist[img[i, j], avg_glw[i, j]] += 1
        hist /= h*w

        # iterate to get the threshold with max entropy
        max_entropy = 0.0
        S = 0
        for t in range(256):
            for s in range(256):
                # background and object entropy(edge not concerned)
                Pb, Po = np.sum(hist[:t, :s]), np.sum(hist[t:, s:])
                if Pb == 0.0 or Po == 0.0:
                    continue
                Hb = -np.sum(hist[:t, :s]/Pb * np.log(hist[:t, :s]/Pb, where=(hist[:t, :s]/Pb>= 1e-7)))
                Ho = -np.sum(hist[t:, s:]/Po * np.log(hist[t:, s:]/Po, where=(hist[t:, s:]/Po>= 1e-7)))
                entropy = Hb + Ho
                if entropy < 0:
                    print('error', entropy, Hb, Ho)
                if entropy > max_entropy:
                    max_entropy = entropy
                    T = t
                    S = s
                print('T', t, 'S', s, 'entropy', entropy)
    elif method == 'Xiao':
        # !still error, and need to be fixed
        # gray similarity matrix for each pixel
        sim = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                # window
                t, b, l, r = cal_wind_boundary((i, j), wind_size, h, w)
                wind = img[t:b + 1, l:r + 1]
                sim[i, j] = np.sum(np.abs(wind - img[i, j]) <= gray_diff)
        
        # get the 2d histogram
        hist = np.zeros((256, wind_size**2), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                hist[img[i, j], sim[i, j]-1] += 1
        hist /= h*w
        # draw_gray_3d(hist)

        max_entropy = 0
        weights = np.exp(-9 * (np.arange(wind_size ** 2) + 1) / (wind_size ** 2))
        weights = (1 + weights) / (1 - weights)
        # iterate to get the threshold with max entropy
        for t in range(256):
            Pb = np.sum(hist[:t, :])
            if Pb == 0.0 or Pb == 1.0:
                continue
            Pf = 1 - Pb
            # background and foreground entropy
            Hb = -np.sum(hist[:t, :]/Pb * np.log(hist[:t, :]/Pb, where=(hist[:t, :]/Pb>= 1e-7)) * weights)
            Hf = -np.sum(hist[t:, :]/Pf * np.log(hist[t:, :]/Pf, where=(hist[t:, :]/Pf>= 1e-7)) * weights)
            entropy = Hb + Hf
            if entropy < 0:
                print('error', entropy, Hb, Hf)
            if entropy > max_entropy:
                max_entropy = entropy
                T = t
            print('T', t, 'entropy', entropy)
    else:
        print('wrong threshold method!')
    
    return T


def group_star(img: np.ndarray, threshold: int, connectivity: int, pixel_limit: int=5) -> tuple[list[list[tuple[int, int]]], int]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        threshold: the threshold used to segment the image
        connectivity: method of connectivity
        pixel_limit: the minimum number of pixels for a group
    Returns:
        group_coords: the coordinates of the grouped pixels(which are the potential stars)
        num_group: the number of the grouped
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
        if len(rows) < pixel_limit and len(cols) < pixel_limit:
            continue
        group_coords.append((rows, cols))

    return group_coords, len(group_coords)


def cal_wind_boundary(center: tuple[int, int], wind_size: int, h: int, w: int) -> tuple[int, int, int, int]:
    '''
        Calculate the boundary of the window.
    Args:
        center: the center of the window
        wind_size: the size of the window
    Returns:
        t: the top boundary of the window
        b: the bottom boundary of the window
        l: the left boundary of the window
        r: the right boundary of the window
    '''
    # construct window
    t = max(0, center[0] - wind_size//2)
    b = min(h-1, center[0] + wind_size//2)
    l = max(0, center[1] - wind_size//2)
    r = min(w-1, center[1] + wind_size//2)

    return t, b, l, r


def cal_center_of_guassian_curve(img: np.ndarray, center: tuple[int,int], wind_size: int) -> tuple[float, float]:
    '''
        Calculate the centroid of the star using the gaussian fitting.
    Args:
        img: the image to be processed
        center: the center of the window
        wind_size: the size of the window
    Returns:
        centroid: the centroid of the star
    '''
    h, w = img.shape
    t, b, l, r = cal_wind_boundary(center, wind_size, h, w)

    # get the window
    x, y = np.meshgrid(range(t, b+1), range(l, r+1))
    x, y = x.flatten()+0.5, y.flatten()+0.5

    # construct the matrix A
    A = np.column_stack([
        x**2 + y**2,
        x,
        y,
        -np.ones_like(x)
    ]).astype(np.float64)

    # ln(I(x, y))
    Y = np.log(img[t:b+1, l:r+1].flatten(order='F')).astype(np.float64)

    # solve the linear equation X = ||Y - AX||min
    X, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return round(-X[1]/(2*X[0]), 3), round(-X[2]/(2*X[0]), 3)


def cal_center_of_gravity(img: np.ndarray, coords: list[tuple[int,int]], method: str, T: int=-1, center: tuple[int, int]=None, A: float=200, sigma: float=1.0, n: int=1) -> tuple[float, float]:
    '''
        Calculate the centroid of the star in the window.
    Args:
        img: the image to be processed
        coords: the coordinates of the star
        method: centroid algorithm
            'CoG': center of gravity
            'MCoG': modified center of gravity
            'WCoG': weighed center of gravity
            'IWCoG': iterative weighed center of gravity
        T: the threshold of the star
        center: initial centroid used in WCoG and IWCoG
        sigma: the sigma used in WCoG and IWCoG
    Returns:
        centroid: the centroid of the star
    '''
    # number of iterations for methods except 'IWCoG' is set to 1
    if method != 'IWCoG':
        n = 1

    coords = np.array(coords)
    # move the initial centroid to the center of the pixel
    if center is not None:
        center = center[0]+0.5, center[1]+0.5
    # row and column add 0.5 to get the center of the pixel
    r, c = coords[:, 0]+0.5, coords[:, 1]+0.5
    # gray
    g = img[coords[:, 0], coords[:, 1]]

    # iterate n times
    while n > 0:
        n -= 1
        if method == 'CoG':
            rs, cs = np.sum(r * g), np.sum(c * g)
            gs = np.sum(g)
        elif method == 'MCoG':
            rs, cs = np.sum(r * (g - T)), np.sum(c * (g - T))
            gs = np.sum(g - T)
        elif method == 'WCoG' or method == 'IWCoG':
            # weight for each pixel used in WCoG and IWCoG
            d = (r-center[0])**2 + (c-center[1])**2
            w = A*np.exp(-d/(2*sigma**2))
            rs, cs = np.sum(r * g * w), np.sum(c * g * w)
            gs = np.sum(g * w)
            print(rs, cs, gs)
            if gs == 0.0:
                return 0.0, 0.0
        else:
            print('wrong gravity method!')
            return 0.0, 0.0
        center = round(rs/gs, 3), round(cs/gs, 3)
    return center


def get_star_centroids(img: np.ndarray, thr_method: str, cen_method: str, wind_size: int=-1) -> list[tuple[float, float]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
        thr_method: threshold calculation method
        cen_method: centroid algorithm
        wind_size: the size of the window used to calculate the centroid
    Returns:
        centroids: the centroids of the stars in the image
    '''

    # get the image size
    h, w = img.shape

    # low pass filter for noise
    filtered_img = img#filter_image(img)

    # calaculate the threshold
    T = cal_threshold(filtered_img, thr_method)

    # rough group star using connectivity
    group_coords, _ = group_star(filtered_img, T, connectivity=2, pixel_limit=5)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for (rows, cols) in group_coords:
        vals = filtered_img[rows, cols]
        # get the brightest pixel and use it as the center
        idx = np.argmax(vals)
        brightest = rows[idx], cols[idx]

        if wind_size != -1:    
            # construct window
            t, b, l, r = cal_wind_boundary(brightest, wind_size, h, w)
            coords = [(x, y) for x in range(t, b+1) for y in range(l, r+1)]
            centroid = cal_center_of_gravity(filtered_img, coords, cen_method, T, center=brightest, n=10)
        else:
            coords = [(r, c) for r, c in zip(rows, cols)]
            centroid = cal_center_of_gravity(filtered_img, coords, cen_method, T, center=brightest, n=10)

        centroids.append(centroid)

    return centroids


def get_star_centroids_with_pca(img: np.ndarray, thr_method: str, cen_method: str, K: int=2*roi+1, L: int=4*roi+1) -> list[tuple[float, float]]:
    '''
        Get the centroids of the stars in the image using PCA-LPG(principal component analysis with local pixel grouping).
    Args:
        img: the image to be processed
        method: centroid algorithm
        K: the size of denoising window
        L: the size of training window
    Returns:
        centroids: the centroids of the stars in the image
    '''
    # get the image size
    h, w = img.shape

    # calaculate the threshold
    T = cal_threshold(img, thr_method)

    # get the local maxium pixel
    group_coords, _ = group_star(img, T, connectivity=2, pixel_limit=5)

    print(T, len(group_coords))

    centroids = []
    for rows, cols in group_coords:
        # get the brightest pixel
        idx = np.argmax(img[rows, cols])
        # set the brightest pixel as the center of window
        center = (rows[idx], cols[idx])
        # get the boundary of the window
        t, _, l, _ = cal_wind_boundary(center, K, h, w)
        # denoise the target block
        denoise_target_block = denoise_window_with_pca(img, center, K, L)
        
        #! subtract the offset for later centroid calculation and add the offset back after calculation
        center = center[0]-t, center[1]-l
        coords = [(r, c) for r in range(0, K) for c in range(0, K)]
        centroid = cal_center_of_gravity(denoise_target_block, coords, cen_method, T=np.mean(img), center=center, n=10)
        centroids.append((centroid[0]+t, centroid[1]+l))
    
    return centroids


if __name__ == '__main__':

    white_noise_stds = [10]
    arr_pos_err = {
        'CoG': [],
        'MCoG': [],
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
            'CoG': 0,
            'MCoG': 0
        }
        # generate the star image
        for i in range(num_test):
            img, star_info = create_star_image(ras[i], des[i], 0, white_noise_std=0)
            real_coords = np.array([x[1] for x in star_info])
            # estimate the centroids of the stars in the image
            for method in pos_err.keys():
                arr_esti_coords = []
                for _ in range(num_esti_times):
                    esti_coords = np.array(get_star_centroids(img, 'Liebe', method, wind_size=10))
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
                pos_err[method] += np.sum(np.min(dist, axis=1))

        for method in pos_err.keys():
            arr_pos_err[method].append(pos_err[method]/num_test)
    
    print(arr_pos_err)

    # for method in arr_pos_err.keys():
    #     plt.plot(white_noise_stds, arr_pos_err[method], label=method)
    # plt.grid()
    # plt.show()
