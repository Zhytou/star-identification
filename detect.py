import cv2
import numpy as np
import skimage.feature as skf


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
        # # !still error, and need to be fixed
        # # gray similarity matrix for each pixel
        # sim = np.zeros_like(img)
        # for i in range(h):
        #     for j in range(w):
        #         # window
        #         t, b, l, r = cal_wind_boundary((i, j), wind_size, h, w)
        #         wind = img[t:b + 1, l:r + 1]
        #         sim[i, j] = np.sum(np.abs(wind - img[i, j]) <= gray_diff)
        
        # # get the 2d histogram
        # hist = np.zeros((256, wind_size**2), dtype=np.float64)
        # for i in range(h):
        #     for j in range(w):
        #         hist[img[i, j], sim[i, j]-1] += 1
        # hist /= h*w

        # max_entropy = 0
        # weights = np.exp(-9 * (np.arange(wind_size ** 2) + 1) / (wind_size ** 2))
        # weights = (1 + weights) / (1 - weights)
        # # iterate to get the threshold with max entropy
        # for t in range(256):
        #     Pb = np.sum(hist[:t, :])
        #     if Pb == 0.0 or Pb == 1.0:
        #         continue
        #     Pf = 1 - Pb
        #     # background and foreground entropy
        #     Hb = -np.sum(hist[:t, :]/Pb * np.log(hist[:t, :]/Pb, where=(hist[:t, :]/Pb>= 1e-7)) * weights)
        #     Hf = -np.sum(hist[t:, :]/Pf * np.log(hist[t:, :]/Pf, where=(hist[t:, :]/Pf>= 1e-7)) * weights)
        #     entropy = Hb + Hf
        #     if entropy < 0:
        #         print('error', entropy, Hb, Hf)
        #     if entropy > max_entropy:
        #         max_entropy = entropy
        #         T = t
        pass
    else:
        print('wrong threshold method!')
    
    return T


def get_seed_coords(img: np.ndarray):
    '''
        Get the seed coordinates with the star distribution.
    '''
    coords = skf.blob_doh(img, min_sigma=1, max_sigma=20, threshold=0.001, num_sigma=10)
    coords = np.array([[int(coord[0]), int(coord[1])] for coord in coords])

    return coords


def region_grow(img: np.ndarray, seed: tuple[int, int], connectivity: int=4) -> np.ndarray:
    '''
        Region grow the image.
    '''
    h, w = img.shape

    # initialize the segmented image
    queue = [seed]

    # offsets
    if connectivity == 4:
        ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        ds = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        print('wrong connectivity!')
        return np.array([]), np.array([])

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


def group_star(img: np.ndarray, method: str, threshold: int, connectivity: int=-1, pixel_limit: int=5) -> tuple[list[list[tuple[int, int]]], int]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        method: RC('Region Grow') or CCL('Connected Components Labeling')
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

    group_coords = []

    # label connected regions of the same value in the binary image
    if method == 'RC':
        seeds = get_seed_coords(img)
        for seed in seeds:
            rows, cols = region_grow(binary_img, seed, connectivity)
            if len(rows) < pixel_limit and len(cols) < pixel_limit:
                continue
            group_coords.append((rows, cols))
    elif method == 'CCL':
        label_num, label_img = cv2.connectedComponents(binary_img, connectivity=connectivity)

        for label in range(1, label_num + 1):
            # get the coords for each label
            rows, cols = np.nonzero(label_img == label)
            # two small to be a star
            if len(rows) < pixel_limit and len(cols) < pixel_limit:
                continue
            group_coords.append((rows, cols))
    else:
        print('wrong method!')
        return []

    return group_coords
