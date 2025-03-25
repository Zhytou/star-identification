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


def get_seed_coords(img: np.ndarray, method: str='doh') -> np.ndarray:
    '''
        Get the seed coordinates with the star distribution.
    '''
    if method == 'doh':
        coords = skf.blob_doh(img, min_sigma=2, max_sigma=3, threshold=0.001, num_sigma=2)
    elif method == 'log':
        coords = skf.blob_log(img, min_sigma=2, max_sigma=3, threshold=0.05, num_sigma=10)
    elif method == 'dog':
        coords = skf.blob_dog(img, min_sigma=2, max_sigma=3, threshold=1, sigma_ratio=1.5)
    else:
        return []
    
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


class UnionSet:
    '''
        Union set for connected components label.
    '''
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_x] = root_y
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_y] += 1

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0


def connected_components_label(img: np.ndarray, connectivity: int=4) -> tuple[int, np.ndarray]:
    '''
        Label the connected components in the image.
    '''
    h, w = img.shape

    # initialize the label image
    label_img = np.zeros_like(img)
    label_cnt = 0
    label_tab = UnionSet()

    # offsets
    if connectivity == 4:
        ds = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif connectivity == 8:
        ds = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        print('wrong connectivity!')
        return 0, np.array([])

    # first pass
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                continue
            connected_labels = []
            for dx, dy in ds:
                if i + dx < 0 or i + dx >= h or j + dy < 0 or j + dy >= w:
                    continue
                if label_img[i + dx, j + dy] > 0:
                    connected_labels.append(label_img[i + dx, j + dy])
            
            if len(connected_labels) == 0:
                label_cnt += 1
                label_img[i, j] = label_cnt
            else:
                min_label = min(connected_labels)
                for label in connected_labels:
                    label_tab.union(min_label, label)
                label_img[i, j] = min_label
    
    label_num = 0

    # second pass
    for i in range(h):
        for j in range(w):
            if label_img[i, j] == 0:
                continue
            label_img[i, j] = label_tab.find(label_img[i, j])
            label_num = max(label_num, label_img[i, j])

    return label_num, label_img


def run_length_code_label(img: np.ndarray, connectivity: int=4) -> list[dict]:
    '''
        Label the connected components in the image using run length code.
    '''

    # label counter & label table for merging
    label_cnt = 0
    label_tab = UnionSet()

    # row, start, end, label
    runs = []

    # generate runs
    for i, row in enumerate(img):
        for (start, end) in find_ranges(row):
            runs.append({
                'row': i,
                'start': start,
                'end': end,
                'label': -1
            })

    # iterate the runs by row
    prev_row_runs = []
    curr_row_runs = []
    for run in runs:
        
        if len(curr_row_runs) > 0 and curr_row_runs[0]['row'] != run['row']:
            if curr_row_runs[0]['row'] == run['row'] - 1:
                prev_row_runs = curr_row_runs
            else:
                prev_row_runs = []
            curr_row_runs = []
        curr_row_runs.append(run)

        connected_labels = []
        for prev_run in prev_row_runs:
            overlap = False
            if connectivity == 4:
                # 4-connectivity
                overlap = (prev_run['start'] <= run['end']) and (prev_run['end'] >= run['start'])
            else:
                # 8-connectivity
                overlap = (prev_run['start'] <= run['end'] + 1) and (prev_run['end'] >= run['start'] - 1)
            
            if overlap:
                connected_labels.append(prev_run['label'])
        
        if len(connected_labels) == 0:
            label_cnt += 1
            label_tab.add(label_cnt)
            run['label'] = label_cnt
        else:
            min_label  = min(connected_labels)
            for label in connected_labels:
                label_tab.union(min_label, label)
            run['label'] = min_label

    # merge the labels
    for run in runs:
        run['label'] = label_tab.find(run['label'])
    
    return runs


def find_ranges(nums, threshold=0) -> list[tuple[int, int]]:
    '''
        Find the ranges of the continuous values in the list.
    Args:
        nums: the list of values
        threshold: the threshold to segment the values
    Returns:
        ranges: the ranges of the continuous values
    '''

    ranges = []
    start = -1
    end = -1
    for i, value in enumerate(nums):
        if value > threshold:
            if start == -1:
                start = i
            end = i
        else:
            if start != -1:
                ranges.append((start, end))
                start = -1
                end = -1
    if start != -1:
        ranges.append((start, end))
    return ranges


def group_star(img: np.ndarray, method: str, threshold: int, connectivity: int=-1, pixel_limit: int=5) -> tuple[list[list[tuple[int, int]]], int]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        method: 
            RG Region Grow
            CCL Connected Components Label
            RLC_CCL Run Length Code Connected Components Label
            CPL Cross Project Label(https://www.sciengine.com/CJSS/doi/10.11728/cjss2006.03.209)
        threshold: the threshold used to segment the image
        connectivity: method of connectivity
        pixel_limit: the minimum number of pixels for a group
    Returns:
        group_coords: the coordinates of the grouped pixels(which are the potential stars)
        num_group: the number of the grouped
    """
    h, w = img.shape

    # if img[u, v] < threshold: 0, else: img[u, v]
    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)

    # if img[u, v] > 0: 1, else: 0
    _, binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

    group_coords = []

    # label connected regions of the same value in the binary image
    if method == 'RG':
        seeds = get_seed_coords(img, 'doh')
        for seed in seeds:
            rows, cols = region_grow(binary_img, seed, connectivity)
            if len(rows) < pixel_limit and len(cols) < pixel_limit:
                continue
            group_coords.append((rows, cols))
    elif method == 'CCL':
        label_num, label_img = connected_components_label(binary_img, connectivity)
        # label_num, label_img = cv2.connectedComponents(binary_img, connectivity=connectivity)

        for label in range(1, label_num + 1):
            # get the coords for each label
            rows, cols = np.nonzero(label_img == label)
            # two small to be a star
            if len(rows) < pixel_limit and len(cols) < pixel_limit:
                continue
            group_coords.append((rows, cols))
    elif method == 'RLC_CCL':
        runs = run_length_code_label(binary_img, connectivity)
        sorted_runs = sorted(runs, key=lambda x: x['label'])
        
        curr_label = 1
        curr_rows, curr_cols = [], []
        for run in sorted_runs:
            if run['label'] != curr_label and len(curr_rows) > pixel_limit and len(curr_cols) > pixel_limit:
                group_coords.append((np.array(curr_rows), np.array(curr_cols)))
                curr_rows, curr_cols = [], []

            curr_label = run['label']
            row, start, end = run['row'], run['start'], run['end']

            curr_rows.extend([row] * (end - start + 1))
            curr_cols.extend(list(range(start, end+1)))
        
        if len(curr_rows) > pixel_limit and len(curr_cols) > pixel_limit:
            group_coords.append((np.array(curr_rows), np.array(curr_cols)))
    elif method == 'CPL':
        vertical_project = np.zeros(w) #np.sum(binary_img, axis=0)
        for i in range(w):
            for j in range(h):
                vertical_project[i] += binary_img[j, i]
        vranges = find_ranges(vertical_project)

        for (y1, y2) in vranges:
            if y1 == y2:
                continue

            horizontal_project = np.zeros(h) #np.sum(binary_img[:, y1:y2], axis=1)
            for i in range(h):
                for j in range(y1, y2+1):
                    horizontal_project[i] += binary_img[i, j]
            hranges = find_ranges(horizontal_project)

            for (x1, x2) in hranges:
                if x1 == x2:
                    continue
                for x in range(x1, x2+1):
                    for y in range(y1, y2+1):
                        if binary_img[x, y] == 0:
                            continue

                        rows, cols = region_grow(binary_img, (x, y), connectivity)
                        if len(rows) < pixel_limit and len(cols) < pixel_limit:
                            continue
                        group_coords.append((rows, cols))
    else:
        print('wrong method!')
        return []

    return group_coords
