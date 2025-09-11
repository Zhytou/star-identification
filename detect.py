import cv2
import bisect as bis
import numpy as np
import scipy.ndimage as nd


class UnionSet:
    '''
        Union set for connected components label.
    '''
    def __init__(self, size: int=0):
        self.parent = {}
        self.rank = {}
        self.cnt = size
        for i in range(size):
            self.parent[i+1] = i+1
            self.rank[i+1] = 0

    def find(self, x: int):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.cnt -= 1
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.cnt += 1
    
    def count(self):
        return self.cnt


def cal_threshold(img: np.ndarray, method: str, factor: float=0.1, wind_size: int=5, gray_diff: int=4) -> int:
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
    elif method == 'Liebe1.8':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 1.8 * std
    elif method == 'Liebe2':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 2 * std
    elif method == 'Liebe2.5':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 2.5 * std
    elif method == 'Liebe3':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 3 * std
    elif method == 'Liebe4':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 4 * std
    elif method == 'Liebe5':
        # calculate the threshold using the mean and standard deviation of multiple windows
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 5 * std
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
        print('Invalid threshold method!')
    
    return T


def get_seed_coords(img: np.ndarray, threshold: int, wind_size: int=3, connectivity: int=4) -> tuple[int, np.ndarray]:
    '''
        Get the seed coordinates with the star distribution.
    Args:
        img: the image to be processed
        threshold: the background threshold
        wind_size: the size of the window
        connectivity
    Returns:
        seeds: the coordinates and labels of the seeds
    '''

    def check_doh(img: np.ndarray, coords: np.ndarray, stride: int):
        '''
            Check the possible seeds with determination of hessian operator.
        '''

        # new offsets including center pixel itself (5, 2) or (9, 2)
        noffsets = np.vstack([[0, 0], offsets])

        # neighbors (n, 5, 2) or (n, 9, 2)
        neighbors = coords[:, None, :] + noffsets

        # second derivative calculation (n, 5) or (n, 9)
        dxx = (img[neighbors[..., 0], neighbors[..., 1]-stride] + img[neighbors[..., 0], neighbors[..., 1]+stride] - 2*img[neighbors[..., 0], neighbors[..., 1]]) / stride
        dyy = img[neighbors[..., 0]-stride, neighbors[..., 1]] + img[neighbors[..., 0]+stride, neighbors[..., 1]] - 2*img[neighbors[..., 0], neighbors[..., 1]]
        dxy = img[neighbors[..., 0]-stride, neighbors[..., 1]-stride] + img[neighbors[..., 0]+stride, neighbors[..., 1]+stride] - img[neighbors[..., 0]-stride, neighbors[..., 1]+stride] + img[neighbors[..., 0]+stride, neighbors[..., 1]-stride]

        # determination of hessian
        doh = dxx*dyy-dxy**2

        # check if the center is local maximum doh results
        mask = np.argmax(doh, axis=1) == 0

        return mask

    if connectivity == 4:
        offsets = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    else: # connectivity == 8:
        offsets = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])

    # window and half window
    wind_size = wind_size+1 if wind_size%2==0 else wind_size
    half_size = wind_size//2

    # get the coordinates of the local maximum
    coords = np.argwhere((img == nd.maximum_filter(img, size=wind_size)) & (img >= threshold))

    # pad the image
    padded_img = np.pad(img, ((wind_size, wind_size), (wind_size, wind_size)), mode='constant').astype(np.int16)

    # select seeds with determination of hessian operator
    mask = np.full(len(coords), False)
    for stride in range(1, half_size+1):
        mask = mask | check_doh(padded_img, coords+wind_size, stride)  #!because of the pad
    coords = coords[mask]

    # save to seeds(row, col, label)
    n = len(coords)
    seeds = np.zeros((n, 3), dtype=int)    
    seeds[:, 0] = coords[:, 0]
    seeds[:, 1] = coords[:, 1]
    seeds[:, 2] = np.arange(1, n+1)

    # make sure seeds are separate
    cnt = 0
    tab = UnionSet(n) 
    for seed in seeds:
        if tab.find(seed[2]) != seed[2]:
            continue

        # neighboring seeds (4, 2)
        nseeds = seed[:2] + offsets

        # the indexs of connected seeds
        idxs = np.where(np.isin(seeds[:, 0], nseeds[:, 0]) & np.isin(seeds[:, 1], nseeds[:, 1]))[0]

        # not connected
        if len(idxs) == 0:
            continue

        for idx in idxs:
            tab.union(seed[2], seeds[idx, 2])
            seed[2] = min(seed[2], seeds[idx, 2])

    # update label
    lab = {}
    for seed in seeds:
        seed2 = tab.find(seed[2])
        if seed2 in lab:
            seed[2] = lab[seed2]
        else:
            cnt += 1
            lab[seed2] = cnt
            seed[2] = cnt
    
    assert cnt == tab.count(), f'{cnt}, {tab.count()}'

    return cnt, seeds


def region_grow(img: np.ndarray, seeds: np.ndarray, connectivity: int=4, steps: int=4) -> np.ndarray:
    '''
        Region grow the image.(Careful, image will change)
    '''
    assert seeds.shape[1] == 3

    # img size
    h, w = img.shape

    # offsets
    if connectivity == 4:
        ds = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])
    elif connectivity == 8:
        ds = np.array([[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]])
    else:
        print('wrong connectivity!')
        return np.array([])

    # breadth first search
    trace = [seeds]

    # maximum search steps
    while steps > 0 and len(seeds) > 0:
        # set to visited
        assert np.all(img[seeds[:, 0], seeds[:, 1]] == 1)
        img[seeds[:, 0], seeds[:, 1]] = 0

        # get the neighboring seeds
        seeds = seeds[:, None, :] + ds # (n, 4, 3)
        seeds = seeds.reshape(-1, 3) # (4*n, 3)

        # boundary check
        mask = (seeds[:, 0] >= 0) & (seeds[:, 0] < h) & (seeds[:, 1] >= 0) & (seeds[:, 1] < w) #(4*n, )
        seeds = seeds[mask]

        # candidate check
        mask = img[seeds[:, 0], seeds[:, 1]] == 1
        seeds = seeds[mask]

        # add to search trace
        trace.append(seeds)

        steps -= 1

    return np.concatenate(trace)


def connected_components_label(img: np.ndarray, connectivity: int=4) -> tuple[int, np.ndarray]:
    '''
        Label the connected components in the image.
    '''
    h, w = img.shape

    # initialize the label image
    # avoid using int8(overflow)
    label_img = np.zeros_like(img, dtype=np.int32)
    label_cnt = 0
    label_tab = UnionSet()

    # offsets
    if connectivity == 4:
        ds = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    elif connectivity == 8:
        ds = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    else:
        print('wrong connectivity!')
        return np.array([]), np.array([])

    # first pass
    xs, ys = np.nonzero(img)
    for x, y in zip(xs, ys):
        # # get neighbors
        # neighbors = ds + (x, y)
        # # boundary check
        # mask = (neighbors[:,0] >= 0) & (neighbors[:,0] < h) & (neighbors[:,1] >= 0) & (neighbors[:,1] < w)
        # neighbors = neighbors[mask]

        # # get connected neighbors' label
        # connected_labels = label_img[neighbors[:, 0], neighbors[:, 1]]
        # connected_labels = connected_labels[connected_labels!=0]
                
        connected_labels = []
        for dx, dy in ds:
            if x + dx < 0 or x + dx >= h or y + dy < 0 or y + dy >= w:
                continue
            if label_img[x + dx, y + dy] > 0:
                connected_labels.append(label_img[x + dx, y + dy])
        
        if len(connected_labels) == 0:
            label_cnt += 1
            label_img[x, y] = label_cnt
            label_tab.add(label_cnt)
        else:
            min_label = min(connected_labels)
            for label in connected_labels:
                label_tab.union(min_label, label)
            label_img[x, y] = min_label
    

    # second pass
    labels = label_img[xs, ys]
    for xi, yi, labeli in zip(xs, ys, labels):
        # find the root label
        label_img[xi, yi] = label_tab.find(labeli)
    
    return label_img


def run_length_code_label(img: np.ndarray, connectivity: int=4) -> list[dict]:
    '''
        Label the connected components in the image using run length code.
    '''
    
    # label counter & label table for merging
    label_cnt = 0
    label_tab = UnionSet()

    def gen_curr_run(row: int, beg: int, end: int):
        '''
            Generate the current run.
        Args:
            row: the number of the row
            beg: the begin column of the run
            end: the end column of the run
        Returns:
            run: the run
        '''
        nonlocal label_cnt, label_tab
        run = {
            'row': row,
            'beg': beg,
            'end': end,
            'label': -1
        }
        connected_labels = []
        # use binary search to find the potential connected labels in the previous runs
        idx = bis.bisect_left(prev_runs, run['beg'], key=lambda x: x['end'])
        if idx < len(prev_runs):
            for prev_run in prev_runs[idx:]:
                # no longer connected
                if prev_run['beg'] > end:
                    break

                # 4-connectivity
                if connectivity == 4:
                    overlap = (prev_run['beg'] <= end) and (prev_run['end'] >= beg)
                else:
                    # 8-connectivity
                    overlap = (prev_run['beg'] <= end + 1) and (prev_run['end'] >= beg - 1)
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
        
        return run

    h, w = img.shape

    # row, beg, end, label
    runs = []

    # preverse row runs
    prev_runs = []

    # iterate on axis 0 of img to generate runs
    for row in range(h):
        if len(prev_runs) > 0 and prev_runs[0]['row'] != row-1:
            prev_runs = []
        
        # current row runs
        curr_runs = []

        # generate current row runs
        col_ranges = find_ranges(img[row])
        if col_ranges is None:
            continue

        curr_runs.extend([gen_curr_run(row, beg, end) for (beg, end) in col_ranges])
        prev_runs = curr_runs
        runs.extend(curr_runs)

    runs = [dict(run, label=label_tab.find(run['label'])) for run in runs]
    runs.sort(key=lambda x: x['label'])

    return runs


def find_ranges(nums: np.ndarray, threshold: int=0) -> list[tuple[int, int]]:
    '''
        Find the ranges of the continuous values in the list.
    '''
    # get the ranges that meet the requirement
    mask = nums > threshold
    
    # add False on both ends of array to deal with boundary
    mask = np.concatenate(([False], mask, [False])).astype(np.int8)

    # compute the discrete difference between consecutive elements
    # transitions from False to True (start of a range) yield 1
    # transitions from True to False (end of a range) yield -1
    begs = np.where(np.diff(mask) > 0)[0]
    ends = np.where(np.diff(mask) < 0)[0] - 1

    if begs.size == 0 or ends.size == 0:
        return None

    return np.vstack([begs, ends]).transpose()


def group_star(img: np.ndarray, method: str, threshold: int, connectivity: int=-1, pixel_limit: int=5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        method: 
            RG Region Grow
            CCL Connected Components Label
            RLC Run Length Code Connected Components Label
            CPL Cross Project Label(https://www.sciengine.com/CJSS/doi/10.11728/cjss2006.03.209)
        pixel_limit: the minimum number of connected pixels in the group
    Returns:
        group_coords: the coordinates of the grouped pixels(which are the potential stars)
        num_group: the number of the grouped
    """
    binary_img = np.zeros_like(img)
    binary_img[img >= threshold] = 1

    group_coords = []

    # label connected regions of the same value in the binary image
    if method == 'RG':
        # do region grow
        n, seeds = get_seed_coords(img, threshold, wind_size=5)
        trace = region_grow(binary_img, seeds)

        # get group coords for each root seed
        for i in range(1, n+1): 
            mask = trace[:, 2] == i
            if np.sum(mask) < pixel_limit:
                continue
            group_coords.append((trace[mask, 0], trace[mask, 1]))

    elif method == 'CCL' or method == 'DCCL':
        label_img = connected_components_label(binary_img, connectivity) if method == 'CCL' else cv2.connectedComponents(binary_img, connectivity=connectivity)[1]

        rows, cols = np.nonzero(label_img)
        labels = label_img[rows, cols]

        # labels pixel > pixel_limit
        ulabels, ucnts = np.unique(labels, return_counts=True)
        ulabels = ulabels[ucnts >= pixel_limit]

        for label in ulabels:
            coords = rows[labels == label], cols[labels == label]
            group_coords.append(coords)
    elif method == 'RLC':
        runs = run_length_code_label(binary_img, connectivity)
        
        curr_label = 1
        curr_rows, curr_cols = [], []
        for run in runs:
            if run['label'] != curr_label and len(curr_rows) > pixel_limit and len(curr_cols) > pixel_limit:
                group_coords.append((np.array(curr_rows), np.array(curr_cols)))
                curr_rows, curr_cols = [], []

            curr_label = run['label']
            row, beg, end = run['row'], run['beg'], run['end']

            curr_rows.extend([row] * (end - beg + 1))
            curr_cols.extend(list(range(beg, end+1)))
        
        if len(curr_rows) > pixel_limit and len(curr_cols) > pixel_limit:
            group_coords.append((np.array(curr_rows), np.array(curr_cols)))
    elif method == 'CPL':
        # vertical projection
        vproj = np.sum(binary_img, axis=0)
        vranges = find_ranges(vproj)
        if vranges is None:
            return []

        for (y1, y2) in vranges:
            # horizontal projection
            hproj = np.sum(binary_img[:, y1:y2+1], axis=1)
            hranges = find_ranges(hproj)

            for (x1, x2) in hranges:
                # get the region of interest
                roi = binary_img[x1:x2+1, y1:y2+1]
                _, label_roi = cv2.connectedComponents(roi, connectivity=connectivity)
                rows, cols = np.nonzero(label_roi)
                labels = label_roi[rows, cols]

                # labels pixel > pixel_limit
                ulabels, ucnts = np.unique(labels, return_counts=True)
                ulabels = ulabels[ucnts >= pixel_limit]

                # add offset
                rows, cols = rows+x1, cols+y1

                # add to group coords
                for label in ulabels:
                    coords = rows[labels == label], cols[labels == label]
                    group_coords.append(coords)
    else:
        print('Invalid segmentation method!')
        return []

    return group_coords
