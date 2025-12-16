import cv2
import bisect as bis
import numpy as np
import scipy.ndimage as ndi
import skimage.filters as filters
import skimage.morphology as morph

from utils import cal_doh, cal_log, cal_ly, cal_sobel, cal_gcm, is_local_max

eps = 1e-10


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


def cal_threshold(img: np.ndarray, method: str) -> int:
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

    if method == 'Otsu':
        T = filters.threshold_otsu(img)
    elif method == 'Liebe3' or method == 'Liebe5':
        mean = np.mean(img)
        std = np.std(img)
        T = mean + 3 * std if method == 'Liebe3' else mean + 5 * std
    elif method == 'Abutaleb':
        avg_img = ndi.uniform_filter(img, size=3)
        
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
        return np.nan
    
    return T


def enhance_image(img: np.ndarray, method: str, patch_size: int=3):
    '''
        Enhance the image.
    Args:
        img: the input image.
        method: the method used to enhance the image.
            'LCM': local contrast measure and other upgraded versions
            'SDM': structural difference measure
            'GCM': gradient directional consistency measure
        patch_size: the size of patch
    Return:
        enhanced_img
    '''

    h, w = img.shape
    img = img.astype(np.float64)                                                                # change data type to avoid overflow
    d = patch_size                                                                              # patch size
    r = patch_size // 2                                                                         # half of patch size
    
    padded_img = np.pad(img, ((r, r), (r, r)))                                                  # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d)).reshape(h, w, -1)    # raw patches (h, w, d²)
    sorted_patches = np.sort(patches, axis=-1)                                                  # sorted patches (h, w, d²)
    
    i, j =  np.indices((h, w))
    shift_mask = np.stack(
        [i >= d, i < h - d, j >= d, j < w - d, 
        (i >= d) & (j >= d), (i < h - d) & (j < w - d),
        (i < h - d) & (j >= d), (i >= d) & (j < w - d),], axis=0)                               # valid shift mask
    shitfs = [(d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d), (-d, d), (d, -d)]             # shift offsets 

    kth = 3
    kmax_map = sorted_patches[..., d**2 - kth]                                                  # kth max map (h, w)
    max_map = sorted_patches[..., -1]                                                           # max map (h, w)

    trim = 2
    tmean_map = np.mean(sorted_patches[..., trim:-trim], axis=-1)                               # trimmed mean map
    mean_map = np.mean(patches, axis=-1)                                                        # mean map (h, w)
    
    shifted_mean_map = np.stack([np.roll(mean_map, shift, axis=(0, 1)) for shift in shitfs])    # shifted mean map(neighbor patches' mean) (8, h, w)
    shifted_mean_map = np.maximum(shifted_mean_map, eps)
    shifted_tmean_map = np.stack([np.roll(tmean_map, shift, axis=(0, 1)) for shift in shitfs])  # shifted trimmed mean map(neighbor patches' mean) (8, h, w)
    shifted_tmean_map = np.maximum(shifted_tmean_map, eps)
    shifted_patches = np.stack([np.roll(patches, shift, axis=(0, 1)) for shift in shitfs])      # shifted patches (8, h, w, d²)

    enhanced_img = np.zeros_like(img, dtype=np.float64)
    if method == 'LCM':
        'A Local Contrast Method for Small  Infrared Target Detection http://ieeexplore.ieee.org/document/6479296/'
        measure = max_map[None, ...]**2 / shifted_mean_map                                      # local contrast measure (8, h, w)
        enhanced_img = np.nanmin(np.where(shift_mask, measure, np.nan),axis=0)                  # enhanced image based on local contrast measure
    elif method == 'ILCM':
        'A Robust Infrared Small Target Detection Algorithm  Based on Human Visual System https://ieeexplore.ieee.org/document/6819810/'
        measure = max_map[None, ...] * mean_map[None, ...] / shifted_mean_map                   # improved local contrast measure (8, h, w)
        enhanced_img = np.nanmin(np.where(shift_mask, measure, np.nan),axis=0)                  # enhanced image based on improved local contrast measure
    elif method == 'NLCM':
        'Effective Infrared Small Target Detection Utilizing a Novel Local Contrast Method http://ieeexplore.ieee.org/document/7725517/'
        kmean_map = np.mean(sorted_patches[..., -kth:], axis=-1)                                 # kth mean map (h, w)
        kvar_map = np.sum((sorted_patches[..., -kth:] - kmean_map[..., None])**2, axis=-1)       # kth variance map (h, w)
        shifted_kmean_map = np.stack([np.roll(kmean_map, shift, axis=(0, 1)) for shift in shitfs]) # shifted kth mean map(neighbor patches' mean) (8, h, w)
        measure = kmean_map[None, ...] * kvar_map[None, ...] / np.maximum(shifted_kmean_map, eps)
        enhanced_img = np.nanmin(np.where(shift_mask, measure, np.nan),axis=0)                  # enhanced image based on novel local contrast measure
    elif method == 'SDM':         
        measure = np.linalg.norm(patches[None, ...] - shifted_patches, axis=-1) / shifted_mean_map # euclidean difference measure (8, h, w)
        enhanced_img = np.nanmin(np.where(shift_mask, measure, np.nan),axis=0)                  # enhanced image based on structural difference measure
    elif method == 'Top-Hat':
        selem = morph.rectangle(d, d)                                                             # structural element
        enhanced_img = morph.white_tophat(img, footprint=selem)                                   # enhanced image based on white top-hat
    else: # method == 'None'
        enhanced_img = img
    
    return enhanced_img


def enhance_image_multiscale(img: np.ndarray, method: str, patch_sizes: list[int]):
    '''
        Enhance the image under multiscale.
    '''
    enhanced_imgs = np.stack([enhance_image(img, method, size) for size in patch_sizes])

    return np.max(enhanced_imgs, axis=0)


def region_grow(img1: np.ndarray, img2: np.ndarray, opt_meth: str, thr_meth: str, patch_size: int=5, wind_size: int=25, connectivity: int=4, steps: int=4) -> tuple[int, np.ndarray]:
    '''
        Do region grow.
    Args:
        img1: the input image for seed detection
        img2: the input image for region growing 
        opt_meth: the method of operator
        thr_meth: the method of threshol
        patch_size: the size used for operator calculation
        wind_size: the size used for local threshold calculation
    Returns:
        n: number of seeds
        seeds: the coordinates and labels of the seeds
        bimg: binary image
    '''
    assert img1.shape == img2.shape

    h, w = img1.shape
    d = patch_size
    r = patch_size // 2
    padded_img = np.pad(img1, ((r, r), (r, r)), mode='constant')                # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))      # image patches (h, w, d, d)
    max_intensity = 255 if img1.dtype == 255 else 1.0                           # max intensity for image data type

    ## 1. Preselect
    #! only use maximum_filter, because the background threshold might be too high under starry light interference
    #! and the size of maximum_filter must be 3, otherwise stars mighbe be missing
    mask = ndi.maximum_filter(img1, 3) == img1                                  # possible seed mask (h, w)
    if d >= 5:                                                                  # check the mean gray of inner ring is higher than the outter one
        rr = d // 4
        inner = np.zeros((d, d), dtype=bool)
        inner[r-rr:r+rr+1, r-rr:r+rr+1] = True
        omean = np.maximum(np.mean(patches[..., ~inner], axis=-1), eps)         # outer ring mean map
        inner[r, r] = False                                                     # exclude central pixel
        imean = np.maximum(np.mean(patches[..., inner], axis=-1), eps)          # inner ring mean map 
        mask = mask & ((img1 == max_intensity) | (img1 - 1.1 * imean >= 0)) & (imean - 1.1 * omean >= 0)
    n = np.sum(mask)                                                            # number of possible seeds

    print('Number of seeds after preselection:', n)

    ## 2. Double check with different operators 
    if opt_meth == 'DOH' or opt_meth == 'LY':
        # https://doi.org/10.16251/j.cnki.1009-2307.2012.01.033
        res = cal_doh(img1, sigma=1) if opt_meth == 'DOH' else cal_ly(img1, sigma=1)[0] # operator results (h, w)
        mask = is_local_max(res, mask, connectivity)
    elif opt_meth == 'SOBEL': 
        # https://doi.org/10.27060/d.cnki.ghbcu.2020.001632
        res = cal_sobel(img1, sigma=1)
        threshold = cal_threshold(res, thr_meth)
        mask = mask & (res > threshold)
    elif opt_meth == 'GCM':
        res1, res2 = cal_gcm(img1, size=5), cal_doh(img1, sigma=0.2)
        mask = (res1 > 0.5) & is_local_max(res2, mask, connectivity)
    else:
        pass

    ## 3. Generate unique label for each seed
    coords = np.argwhere(mask)                                                  # coordinates of preselected seeds (_, 2)
    labels = np.arange(1, len(coords)+1).reshape(-1, 1)                         # initilize labels for each seeds (_, 1)
    n, seeds = merge_seeds(np.hstack([coords, labels]))                         # merge the label of connected seeds (_, 3)

    print('Number of seeds after operator double check:', n)

    ## 4. Gnerate binary image for later growth
    binary_img = np.zeros((h, w), dtype=np.uint8)

    if True: # use local threshold if flag is True
        rr = wind_size // 2

        y, x = coords[:, 0], coords[:, 1]
        ymin, ymax = np.maximum(0, y - rr), np.minimum(h, y + rr + 1)           # top and bottom boundary
        xmin, xmax = np.maximum(0, x - rr), np.minimum(w, x + rr + 1)           # left and right boundary

        for y1, y2, x1, x2 in zip(ymin, ymax, xmin, xmax):    
            threshold = cal_threshold(img2[y1:y2, x1:x2], thr_meth)
            binary_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2] > threshold
    else:
        threshold = cal_threshold(img2, thr_meth)
        binary_img[img2 > threshold] = 1

    ## 5. Retain only foreground seeds
    valid = binary_img[seeds[:, 0], seeds[:, 1]] == 1
    seeds = seeds[valid]

    ## 6. Do region growth on binary image, namely breadth first search
    trace = [seeds]
    if connectivity == 4:
        offs = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=int)
    else:  # connectivity == 8
        offs = np.array([[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0]], dtype=int)
    while steps > 0:                                                            # maximum search steps
        assert np.all(binary_img[seeds[:, 0], seeds[:, 1]] == 1)                # must be foreground seeds
        binary_img[seeds[:, 0], seeds[:, 1]] = 0                                # set to visited

        seeds = np.reshape(seeds[:, None, :] + offs, (-1, 3))                   # get the neighboring seeds (4 * n, 3) or (8 * n, 3)
        valid = (seeds[:, 0] >= 0) & (seeds[:, 0] < h) & (seeds[:, 1] >= 0) & (seeds[:, 1] < w) # boundary check
        seeds = seeds[valid]

        valid = binary_img[seeds[:, 0], seeds[:, 1]] == 1                       # !careful must validate foreground pixels only after bounds check; otherwise, binary_img[seeds] may raise IndexError for out-of-range coordinates.
        seeds = seeds[valid]

        trace.append(seeds)                                                     # add current seeds to search trace
        steps -= 1

    return n, np.concatenate(trace)


def merge_seeds(seeds: np.ndarray):
    '''
        Merge possible connected seeds.
    '''

    tot = len(seeds)
    tab = UnionSet(tot)

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


def connected_components_label(img: np.ndarray, connectivity: int=4) -> tuple[int, np.ndarray]:
    '''
        Label the connected components in the image.
    '''
    h, w = img.shape

    # initialize the label image
    # avoid using int8(overflow)
    label_img = np.zeros_like(img, dtype=np.uint32)
    label_cnt = 0
    label_tab = UnionSet()

    # offsets
    ds = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int) if connectivity == 4 else np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=int)

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


def group_star(img: np.ndarray, method: list[str], connectivity: int=4, pixel_limit: int=5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
        Group the facula(potential star) in the image.
    Args:
        img: the image to be processed
        method: 
            RG Region Grow
            DOH Determination of Hessian
            LCM Local Contrast Measure
            CCL Connected Components Label
            RLC Run Length Code Connected Components Label
            CPL Cross Projection Label
        pixel_limit: the minimum number of connected pixels in the group
    Returns:
        group_coords: the coordinates of the grouped pixels(which are the potential stars)
    """
    ehc_meth, thr_meth, lab_meth, opt_meth = method
    
    binary_img = np.zeros_like(img, dtype=np.uint8)
    enhanced_img = enhance_image(img, ehc_meth, patch_size=5)
    # enhanced_img = enhance_image_multiscale(img, ehc_meth, patch_sizes=[3, 5, 7, 9])
    
    ethreshold = cal_threshold(enhanced_img, thr_meth)
    binary_img[enhanced_img > ethreshold] = 1 # except RG-based methods might use local thresholds                              
    print('Global threshold of enhanced image:', ethreshold)

    group_coords = []
    # label connected regions of the same value in the binary image
    if lab_meth == 'RG':
        # region growth on enhanced image
        n, trace = region_grow(img, enhanced_img, opt_meth, thr_meth, patch_size=5, wind_size=25)

        # get group coords for each root seed
        for i in range(1, n+1): 
            mask = trace[:, 2] == i
            if np.sum(mask) < pixel_limit:
                continue
            group_coords.append((trace[mask, 0], trace[mask, 1]))
    elif lab_meth == 'CCL' or lab_meth == 'DCCL':
        label_img = connected_components_label(binary_img, connectivity) if lab_meth == 'DCCL' else cv2.connectedComponents(binary_img, connectivity=connectivity)[1]

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
