import cv2
import bisect as bis
import numpy as np
import scipy.ndimage as nd

from utils import get_offsets, cal_doh, cal_ly, cal_sobel


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


def cal_threshold(img: np.ndarray, method: str, wind_size: int=5) -> int:
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
    else:
        print('Invalid threshold method!')
    
    return T


def get_seeds_with_doh(img: np.ndarray, threshold: int, connectivity: int=4) -> tuple[int, np.ndarray]:
    '''
        Get the seeds by doh.
    Args:
        img: the image to be processed
        threshold: the background threshold
        connectivity
    Returns:
        seeds: the coordinates and labels of the seeds
    '''

    # offsets
    offsets = get_offsets(connectivity)

    # window size
    d = 5

    # get the coordinates of the local maximum
    coords = np.argwhere((img == nd.maximum_filter(img, size=d)) & (img >= threshold))

    # pad the image
    padded_img = np.pad(img, ((d, d), (d, d)), mode='constant').astype(np.int16)

    # determination of hessian
    neighbors = coords[:, None, :] + np.vstack([[0, 0], offsets]) # (n, 5, 2) or (n, 9, 2)
    doh1 = cal_doh(padded_img, neighbors[..., 0]+d, neighbors[..., 1]+d, 1) # !because of the pad
    doh2 = cal_doh(padded_img, neighbors[..., 0]+d, neighbors[..., 1]+d, 2)

    # select seeds with determination of hessian operator
    mask = (np.argmax(doh1, axis=1) == 0) | (np.argmax(doh2, axis=1) == 0)
    coords = coords[mask]

    # save to seeds(row, col, label) and make sure seeds are separate
    labels = np.arange(1, len(coords)+1).reshape(-1, 1)
    n, seeds = merge_seeds(np.hstack([coords, labels]))

    return n, seeds


def get_seeds_with_ly(img: np.ndarray, threshold: int):
    '''
        Get the seeds by ly.
        https://doi.org/10.16251/j.cnki.1009-2307.2012.01.033
    '''

    h, w = img.shape
    d = 3
    r = d // 2
    padded_img = np.pad(img, ((d, d), (d, d)), mode='edge').astype(float)
    
    # 1. Preselect
    coords = np.stack(np.meshgrid(np.arange(d, h+d), np.arange(d, w+d)), axis=-1)   # (h, w, 2)
    offsets = get_offsets(4)  # (4, 2)
    acoords = coords[..., None, :] + offsets[None, None, ...] # (h, w, 4, 2)
    mask = np.sum(img[..., None] - padded_img[acoords[..., 0], acoords[..., 1]] > threshold, # (h, w, 4)
                  axis=-1) >= 2
    
    # 2. Check
    coords = np.argwhere(mask)  # (n, 2)
    offsets = np.stack(np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1)), axis=-1).reshape(-1, 2) # (d², 2)
    acoords = coords[:, None, :] + offsets[None, ...]
    _, weights = cal_ly(padded_img, acoords[..., 0].ravel()+d, acoords[..., 1].ravel()+d, d)
    weights = weights.reshape(-1, d**2)
    mask = np.argmax(weights, axis=1) == d**2//2
    
    # 3. Save
    coords = coords[mask]
    labels = np.arange(1, len(coords)+1).reshape(-1, 1)
    n, seeds = merge_seeds(np.hstack([coords, labels]))

    return n, seeds


def get_seeds_with_sobel(img: np.ndarray, threshold: int):
    '''
        Get the seeds by modified sobel.
        https://doi.org/10.27060/d.cnki.ghbcu.2020.001632
    '''
    d = 3
    r = d // 2
    padded_img = np.pad(img, ((r, r), (r, r)), mode='edge').astype(float)
    coords = np.argwhere(img > threshold)
    mask = np.sum(cal_sobel(padded_img, coords[:, 0]+r, coords[:, 1]+r) > threshold, axis=1) >= 2

    coords = coords[mask]
    labels = np.arange(1, len(coords)+1).reshape(-1, 1)
    n, seeds = merge_seeds(np.hstack([coords, labels]))

    return n, seeds


def merge_seeds(seeds: np.ndarray):
    '''
        Merge possible connected seeds.
    '''

    n = len(seeds)
    tab = UnionSet(n)

    for seed in seeds:
        x, y, _ = seed
        # the indexs of connected seeds
        idxs = np.where(
            ((seeds[:, 0] == x+1) & (seeds[:, 1] == y)) |
            ((seeds[:, 0] == x-1) & (seeds[:, 1] == y)) | 
            ((seeds[:, 0] == x) & (seeds[:, 1] == y+1)) | 
            ((seeds[:, 0] == x) & (seeds[:, 1] == y-1))
        )[0]

        # not connected
        if len(idxs) == 0:
            continue

        # merge the connected label
        for idx in idxs:
            tab.union(seed[2], seeds[idx, 2])
            seed[2] = min(seed[2], seeds[idx, 2])

    # update label
    cnt = 0
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
        Region grow the image.(Careful, binary image will change)
    '''
    assert seeds.shape[1] == 3

    # img size
    h, w = img.shape

    # offsets
    offsets = get_offsets(connectivity)                                   # (connectivity, 2)
    offsets = np.hstack([offsets, np.zeros((connectivity, 1), dtype=int)])  # (connectivity, 3)

    # breadth first search
    trace = [seeds]

    # maximum search steps
    while steps > 0 and len(seeds) > 0:
        # set to visited
        assert np.all(img[seeds[:, 0], seeds[:, 1]] == 1)
        img[seeds[:, 0], seeds[:, 1]] = 0

        # get the neighboring seeds
        seeds = seeds[:, None, :] + offsets # (n, 4, 3)
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
    ds = get_offsets(connectivity)

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


def group_star(img: np.ndarray, method: str, threshold: int, connectivity: int=4, pixel_limit: int=5) -> list[tuple[np.ndarray, np.ndarray]]:
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
    if method == 'RG_DOH' or method == 'RG_LY' or method == 'RG_SOBEL':
        # do region grow
        if method == 'RG_DOH':
            n, seeds = get_seeds_with_doh(img, threshold)
        elif method == 'RG_LY':
            n, seeds = get_seeds_with_ly(img, threshold)
        else:
            n, seeds = get_seeds_with_sobel(img, threshold)
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
