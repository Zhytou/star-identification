import cv2
import numpy as np
import matplotlib.pyplot as plt

from simulate import create_star_image
from denoise import filter_image, denoise_image


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
            'Yang': proposed method
                2d histogram otsu(adding a max value axis)
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
    elif method == 'Abutaleb':
        # average gray level matrix for each pixel's window
        avg_img = cv2.medianBlur(img, wind_size)
        
        # get the 2d histogram
        hist = np.zeros((256, 256), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                hist[img[i, j], avg_img[i, j]] += 1
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
    elif method == 'Yang':
        fimg = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)

        # get the 2d histogram
        hist = np.zeros((256, 256), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                hist[img[i, j], fimg[i, j]] += 1
        hist /= h*w
        draw_gray_3d(hist)

        # iterate to get the threshold with max entropy
        # for t in range(256):
        #     Pb = np.sum(hist[:t, 255])
        #     if Pb == 0.0 or Pb == 1.0:
        #         continue
        #     Pf = 1 - Pb
        #     # background and foreground entropy
        #     Hb = -np.sum(hist[:t, 255]/Pb * np.log(hist[:t, 255]/Pb, where=(hist[:t, 255]/Pb>= 1e-7)) * weights)
        #     Hf = -np.sum(hist[t:, 255]/Pf * np.log(hist[t:, 255]/Pf, where=(hist[t:, 255]/Pf>= 1e-7)) * weights)
        #     entropy = Hb + Hf
        #     if entropy < 0:
        #         print('error', entropy, Hb, Hf)
        #     if entropy > max_entropy:
        #         max_entropy = entropy
        #         T = t
        #     print('T', t, 'entropy', entropy)
    else:
        print('wrong threshold method!')
    
    return T


def get_seed_coords_with_star_distri(img: np.ndarray, half_size: int=2):
    '''
        Get the seed coordinates with the star distribution.
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


def region_grow(img: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    '''
        Region grow the image.
    '''
    h, w = img.shape

    # initialize the segmented image
    queue = [seed]

    # offsets
    ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # coords
    xs, ys = [], []

    while len(queue) > 0:
        x, y = queue.pop(0)
        if img[x, y] == 0:
            continue
        img[x, y] = 0
        xs.append(x)
        ys.append(y)
        for dx, dy in ds:
            if x + dx < 0 or x + dx >= h or y + dy < 0 or y + dy >= w:
                continue
            queue.append((x + dx, y + dy))

    return np.array(xs), np.array(ys)


def group_star(img: np.ndarray, threshold: int, seeds: list[tuple]=None, connectivity: int=-1, pixel_limit: int=5) -> tuple[list[list[tuple[int, int]]], int]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        threshold: the threshold used to segment the image
        seeds: the seeds used to grow the region
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

    group_coords = []

    # label connected regions of the same value in the binary image
    if seeds is not None:
        for seed in seeds:
            rows, cols = region_grow(binary_img, seed)
            if len(rows) < pixel_limit and len(cols) < pixel_limit:
                continue
            group_coords.append((rows, cols))
    else:
        label_num, label_img = cv2.connectedComponents(binary_img, connectivity=connectivity)

        for label in range(1, label_num + 1):
            # get the coords for each label
            rows, cols = np.nonzero(label_img == label)
            # two small to be a star
            if len(rows) < pixel_limit and len(cols) < pixel_limit:
                continue
            group_coords.append((rows, cols))

    return group_coords


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


def cal_center_of_guassian_curve(img: np.ndarray, rows, cols) -> tuple[float, float]:
    '''
        Calculate the centroid of the star using the gaussian fitting.
    Args:
        img: the image to be processed
    Returns:
        centroid: the centroid of the star
    '''
    x, y = rows+0.5, cols+0.5

    # construct the matrix A
    A = np.column_stack([
        x**2 + y**2,
        x,
        y,
        -np.ones_like(x)
    ]).astype(np.float64)

    if np.linalg.cond(A) > 1e12:
        print('A is ill-conditioned')
        return 0.0, 0.0 

    # ln(I(x, y))
    Y = np.log(img[rows, cols].flatten(order='F')+1e-11).astype(np.float64)

    # solve the linear equation X = ||Y - AX||min
    X, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    return round(-X[1]/(2*X[0]), 3), round(-X[2]/(2*X[0]), 3)


def cal_center_of_gravity(img: np.ndarray, rows: np.ndarray, cols: np.ndarray, method: str, T: int=-1, center: tuple[int, int]=None, A: float=200, sigma: float=1.0, n: int=1) -> tuple[float, float]:
    '''
        Calculate the centroid of the star in the window.
    Args:
        img: the image to be processed
        rows/cols: the coordinates of the pixels in the window
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

    # move the initial centroid to the center of the pixel
    if center is not None:
        center = center[0]+0.5, center[1]+0.5
    
    # gray
    g = img[rows, cols]
    # row and column add 0.5 to get the center of the pixel
    x, y = rows+0.5, cols+0.5

    # iterate n times
    while n > 0:
        n -= 1
        if method == 'CoG':
            # row multiply gray and sum, column multiply gray and sum
            xgs, ygs = np.sum(x * g), np.sum(y * g)
            gs = np.sum(g)
        elif method == 'MCoG':
            xgs, xgs = np.sum(x * (g - T)), np.sum(y * (g - T))
            gs = np.sum(g - T)
        elif method == 'WCoG' or method == 'IWCoG':
            # weight for each pixel used in WCoG and IWCoG
            d = (x-center[0])**2 + (y-center[1])**2
            w = A*np.exp(-d/(2*sigma**2))
            xgs, ygs = np.sum(x * g * w), np.sum(y * g * w)
            gs = np.sum(g * w)
            if gs == 0.0:
                return 0.0, 0.0
        else:
            print('wrong gravity method!')
            return 0.0, 0.0
        center = round(xgs/gs, 3), round(ygs/gs, 3)
    return center


def get_star_centroids(img: np.ndarray, den_method: str, thr_method: str, cen_method: str, wind_size: int=-1) -> list[tuple[float, float]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
        den_method: denoising method
        thr_method: threshold calculation method
        cen_method: centroid algorithm
        wind_size: the size of the window used to calculate the centroid
    Returns:
        centroids: the centroids of the stars in the image
    '''

    # get the image size
    h, w = img.shape

    # denoise
    filtered_img = denoise_image(img, den_method)
    filtered_img, seed_coords = get_seed_coords_with_star_distri(filtered_img, half_size=2)
    
    # calaculate the threshold
    T = cal_threshold(filtered_img, thr_method)

    # rough group star using connectivity
    group_coords = group_star(filtered_img, T, seeds=seed_coords, connectivity=4, pixel_limit=5)

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
            nrows, ncols = np.meshgrid(np.arange(t, b+1), np.arange(l, r+1))
            nrows, ncols = nrows.flatten(), ncols.flatten()
            centroid = cal_center_of_gravity(filtered_img, nrows, ncols, cen_method, T, center=brightest, n=10)
        else:
            centroid = cal_center_of_gravity(filtered_img, rows, cols, cen_method, T, center=brightest, n=10)

        centroids.append(centroid)

    return centroids


if __name__ == '__main__':
    # times of estimation using centroid algorithm
    num_esti_times = 3
    # random ra & de test
    num_test = 10
    # generate random right ascension[0, 360] and declination[-90, 90]
    ras = np.random.uniform(0, 2*np.pi, num_test)
    des = np.arcsin(np.random.uniform(-1, 1, num_test))
    # centroid position error
    pos_err = {
        'proposed': 0.0,
        'gaussian': 0.0,
        'mean': 0.0,
        'median': 0.0,
    }
    # generate the star image
    for i in range(num_test):
        img, star_info = create_star_image(ras[i], des[i], 0, sigma_g=0.05, prob_p=0.0001)
        real_coords = np.array([x[1] for x in star_info])
        # estimate the centroids of the stars in the image
        for method in pos_err.keys():
            arr_esti_coords = []
            for _ in range(num_esti_times):
                esti_coords = np.array(get_star_centroids(img, method, 'Liebe', 'CoG', wind_size=10))
                if len(real_coords) != len(esti_coords):
                    print('Wrong star number extracted: ', len(real_coords), len(esti_coords))
                    continue
                arr_esti_coords.append(esti_coords)
            if len(arr_esti_coords) == 0:
                continue
            avg_esti_coords = np.mean(arr_esti_coords, axis=0)
            
            # calculate the position error
            diff = real_coords[:, None] - avg_esti_coords
            dist = np.min(np.sum(diff**2, axis=-1), axis=-1)
            pos_err[method] += np.mean(dist)

    print(pos_err)
