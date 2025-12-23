import cv2
import bisect as bis
import numpy as np
import scipy.ndimage as ndi
import skimage.filters as filters
import skimage.morphology as morph

from utils import cal_derivative, cal_difference, cal_doh, cal_log, cal_ly, cal_sobel, is_local_topk, is_near_local_max

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


def cal_lcm(img: np.ndarray, method: str, size: int):
    '''
        Calculate local contrast measure.
    '''
    h, w = img.shape
    d = size                                                                                    # patch size
    r = size // 2                                                                               # half of patch size
    
    fimg = img.astype(np.float64)                                                               # change data type to avoid overflow
    padded_img = np.pad(fimg, ((r, r), (r, r)))                                                 # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)
    flatten_patches = np.reshape(patches, (h, w, -1))                                           # flatten patches (h, w, d²)
    sorted_patches = np.sort(flatten_patches, axis=-1)                                          # sorted flatten patches (h, w, d²)
    
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
    kmean_map = np.mean(sorted_patches[..., -kth:], axis=-1)                                    # kth top mean map (h, w)
    tmean_map = np.mean(sorted_patches[..., trim:-trim], axis=-1)                               # trimmed mean map
    mean_map = np.mean(flatten_patches, axis=-1)                                                # mean map (h, w)
    
    shifted_mean_map = np.stack([np.roll(mean_map, shift, axis=(0, 1)) for shift in shitfs])    # shifted mean map(neighbor patches' mean) (8, h, w)
    shifted_mean_map = np.maximum(shifted_mean_map, eps)
    shifted_kmean_map = np.stack([np.roll(kmean_map, shift, axis=(0, 1)) for shift in shitfs]) # shifted kth mean map(neighbor patches' mean) (8, h, w)
    shifted_kmean_map = np.maximum(shifted_kmean_map, eps)
    shifted_tmean_map = np.stack([np.roll(tmean_map, shift, axis=(0, 1)) for shift in shitfs])  # shifted trimmed mean map(neighbor patches' mean) (8, h, w)
    shifted_tmean_map = np.maximum(shifted_tmean_map, eps)

    if method == 'LCM':
        'A Local Contrast Method for Small  Infrared Target Detection http://ieeexplore.ieee.org/document/6479296/'
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                max_map[None, ...]**2 / shifted_mean_map,                                       # (8, h, w)
                np.nan
            ), axis=0
        )                                                                                       # local contrast measure (h, w)
    elif method == 'ILCM':
        'A Robust Infrared Small Target Detection Algorithm  Based on Human Visual System https://ieeexplore.ieee.org/document/6819810/'
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                max_map[None, ...] * mean_map[None, ...] / shifted_mean_map,                    # (8, h, w)
                np.nan
            ), axis=0
        )                                                                                       # improved local contrast measure (h, w)
    elif method == 'NLCM':
        'Effective Infrared Small Target Detection Utilizing a Novel Local Contrast Method http://ieeexplore.ieee.org/document/7725517/'
        kvar_map = np.sum((sorted_patches[..., -kth:] - kmean_map[..., None])**2, axis=-1)      # kth variance map (h, w)
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                kmean_map[None, ...] * kvar_map[None, ...] / shifted_kmean_map,                 # (8, h, w)
                np.nan
            ), axis=0
        )                                                                                       # novel local contrast measure (h, w)
    elif method  == 'RLCM':
        'Infrared Small Target Detection Utilizing the  Multiscale Relative Local Contrast Measure http://ieeexplore.ieee.org/document/8289318/'
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                np.clip(kmean_map[None, ...] / shifted_kmean_map - 1, 0, np.inf) * kmean_map,   # (8, h, w)
                np.nan
            ), axis=0
        )                                                                                       # relative local contrast measure (h, w)
    elif method == 'MLCM':
        'Self-defined Modified Local Contrast Measure'
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                np.clip(tmean_map[None, ...] / shifted_tmean_map - 1, 0, np.inf) * tmean_map,   # (8, h, w)
                np.nan
            ), axis=0
        )     
    elif method == 'SDM':
        shifted_patches = np.stack([np.roll(patches, shift, axis=(0, 1)) for shift in shitfs])  # shifted patches (8, h, w, d, d)
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                np.linalg.norm(patches[None, ...] - shifted_patches, axis=(-2, -1)) / shifted_mean_map,   # (8, h, w)
                np.nan
            ), axis=0
        )                                                                                       # euclidean difference measure (h, w) 
    else:
        measure = np.zeros_like(img, dtype=img.dtype)

    return measure


def cal_gcm(img: np.ndarray, method: str, size: int=5, sigma: float=0.2):
    '''
        Calculate gradient consistency measure.
    '''
    h, w = img.shape
    d = size
    r = size // 2

    padded_img = np.pad(img, ((r, r), (r, r)))                                                  # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)
    flatten_patches = np.reshape(patches, (h, w, -1))                                           # flatten patches (h, w, d²)

    diffs = np.stack([cal_difference(img, dir) for dir in range(8)], axis=0)                    # neighoring pixel difference map (8, h, w)
    if True:
        grad_y = cal_derivative(img, order=(1, 0), sigma=sigma)                                 # gradient y map (h, w)
        grad_x = cal_derivative(img, order=(0, 1), sigma=sigma)                                 # gradient x map (h, w)
    else:
        grad_y = diffs[0]
        grad_x = diffs[2]
    padded_grad_x, padded_grad_y = np.pad(grad_x, ((r, r), (r, r))), np.pad(grad_y, ((r, r), (r, r)))  # padded gradient x and gradient y map (h, w, d, d)

    grad = np.stack([
        np.lib.stride_tricks.sliding_window_view(padded_grad_x, (d, d)), 
        np.lib.stride_tricks.sliding_window_view(padded_grad_y, (d, d))
    ], axis=-1)                                                                                 # gradient map (h, w, d, d, 2)
    gnorm = np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), eps)
    grad = grad / gnorm

    if method == 'PGCM':
        'Self-defined Pixel-wise GCM'
        y, x = np.indices((d, d))                                                               # !careful first row index, then column index
        radial = np.stack([r - x, r - y], axis=-1)                                              # radial vectors (d, d, 2)
        rnorm = np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), eps)                 # radial vectors' norm
        radial = radial / rnorm

        dot_product = np.sum(grad * radial[None, None, ...], axis=-1)                           # dot product (h, w, d, d)
        measure = np.clip(np.sum(dot_product, axis=(-2, -1)) / (d**2 - 1), 0, 1)                # gradient consistency measure (h, w)
        # measure[measure < 0.8] = 0
    elif method == 'BGCM': 
        'Self-defined Block-wise GCM'
        max_indexs = np.argmax(flatten_patches, axis=-1)                                        # the index of maximum values
        y0, x0 = max_indexs // d, max_indexs % d                                                # the local coordinates(row, column) of each maximum values(h, w)
        y, x = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
        radial = np.stack([
            x0[..., None, None] - x[None, None, ...],
            y0[..., None, None] - y[None, None, ...]
        ], axis=-1)                                                                             # radial vectors (h, w, d, d, 2)
        rnorm = np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), eps)
        radial = radial / rnorm

        dot_product = np.sum(grad * radial, axis=-1)                                            # dot product (h, w, d, d)
        measure = np.clip(np.sum(dot_product, axis=(-2, -1)) / (d**2 - 1), 0, 1)                # gradient consistency measure (h, w)
    else:
        measure = np.zeros_like(img)

    return measure


def binarize_image(img: np.ndarray, method: str) -> int:
    '''
        Binarize the image.
    Args:
        img: the image to be processed
        method: the method used to calculate the threshold
    Returns:
        binary_img

    '''
    assert img.dtype == np.uint8

    binary_img = np.zeros_like(img, dtype=np.uint8)
    if method == 'Otsu':
        T = filters.threshold_otsu(img)
        binary_img[img > T] = 1
    elif method.startswith('Liebe'):
        k = float(method[5:])
        mean = np.mean(img)
        std = np.std(img)
        T = mean + k * std
        binary_img[img > T] = 1
    elif method == 'Xiao':
        'Entropic thresholding based on gray-level spatial correlation histogram https://ieeexplore.ieee.org/document/4761626/?arnumber=4761626'
        ## 0. Predefined parameters
        size = 5
        gray_diff = 10
        gray_step = 8
        num_step = 1

        d = size
        r = d // 2
        padded_img = np.pad(img, ((r, r), (r, r)))
        patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))
        
        ## 1. Initialize f(x, y) and g(x, y)
        # f(x, y) is the gray value of the pixel located at the point (x, y) in a digital image.
        # g(x, y) is the number of the pixels of which the gray value is close to it in the corresponding N × N neighborhood,
        f = img        
        g = np.sum(patches - img[:, :, None, None] < gray_diff, axis=(-2, -1))

        ## 2. Construct joint histogram h(k, m) 
        # h(k, m) = prob(f(x, y) == k and g(x, y) == m)
        k_max = 256
        m_max = d**2

        h = np.zeros((k_max, m_max), np.float64)
        np.add.at(h, (f.flatten(), f.flatten()), 1)
        h /= np.sum(h)

        ## 3. Get the optimal threshold for f(x, y) and g(x, y)
        k_vals, m_vals = np.arange(0, k_max, gray_step), np.arange(0, m_max, num_step)
        k, m = np.meshgrid(k_vals, m_vals, indexing='ij')
        T_k_opt, T_m_opt = 0, 0

        ## 4. Do segmentation
        binary_img[(f > T_k_opt) | (g > T_m_opt)] = 1

    return binary_img


def enhance_image(img: np.ndarray, method: str, patch_size: int=3, preserve_dtype: bool=True):
    '''
        Enhance the image.
    Args:
        img: the input image.
        method: the method used to enhance the image.
        patch_size: the size of patch
        preserve_dtype: 
            If True, the enhanced output is clipped to the valid range of the input dtype (e.g., [0, 255] for uint8) and cast back to the original dtype.
            If False, the result is returned as float32 to preserve negative or out-of-range values.
    Return:
        enhanced_img
    '''

    d = patch_size                                                                              # patch size
    r = patch_size // 2                                                                         # half of patch size
    max_val = 255 if img.dtype == np.uint8 else 1.0

    padded_img = np.pad(img, ((r, r), (r, r)))                                                  # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)

    if method in ['LCM', 'ILCM', 'NLCM', 'RLCM', 'MLCM']:
        enhanced_img = cal_lcm(img, method, size=d)
    elif method == 'Top-Hat':
        selem = np.ones((d, d), dtype=bool)                                                     # structural element
        enhanced_img = morph.white_tophat(img, footprint=selem)                                 # enhanced image based on white top-hat
    elif method == 'Max-Median' or method == 'Max-Mean':
        'Max-Mean and Max-Median filters for detection of small-targets http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=905421'
        opt_mask = np.zeros((4, d, d), dtype=bool)                                              # operation mask
        opt_mask[0, :, r] = True                                                                # vertical line
        opt_mask[1, r, :] = True                                                                # horizontal line
        opt_mask[2, np.arange(0, d), np.arange(0, d),] = True                                   # main diagonal
        opt_mask[3, np.arange(0, d)[::-1], np.arange(0, d)] = True                              # anti-diagonal

        if method == 'Max-Median':
            median_map = np.stack([np.median(patches[:, :, mask], axis=-1) for mask in opt_mask], axis=-1)
            enhanced_img = img - np.max(median_map, axis=-1)
        else:
            mean_map = np.stack([np.mean(patches[:, :, mask], axis=-1) for mask in opt_mask], axis=-1)
            enhanced_img = img - np.max(mean_map, axis=-1)
    elif method == 'MSobel':
        enhanced_img = cal_sobel(img, sigma=1)
    elif method in ['PGCM', 'BGCM']:
        enhanced_img = cal_gcm(img, method, size=d, sigma=1)
    elif method == 'IGM':
        lcm = cal_lcm(img, 'MLCM', size=d)
        gcm = cal_gcm(img, 'BGCM', size=d, sigma=1)
        enhanced_img = lcm * gcm
    else: # method == 'None'
        enhanced_img = img

    if preserve_dtype:
        enhanced_img = (enhanced_img / np.maximum(np.max(enhanced_img), eps) * max_val).astype(img.dtype)

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
    #! only use local rank_filter, because the global background threshold might be too high under starry light interference
    #! also, avoid preselection when noise is relatively high
    if True:
        mask = is_local_topk(img1, k=-2, connectivity=connectivity) | is_near_local_max(img1, connectivity=connectivity) # possible seed mask (h, w)
        if d >= 5:                                                              # check the mean gray of inner ring is higher than the outter one
            rr = d // 4
            inner = np.zeros((d, d), dtype=bool)
            inner[r-rr:r+rr+1, r-rr:r+rr+1] = True
            omean = np.maximum(np.mean(patches[..., ~inner], axis=-1), eps)     # outer ring mean map
            inner[r, r] = False                                                 # exclude central pixel
            imean = np.maximum(np.mean(patches[..., inner], axis=-1), eps)      # inner ring mean map 
            mask = mask & ((img1 == max_intensity) | (img1 - 1. * imean >= 0)) & (imean - 1. * omean >= 0)
    else:
        mask = np.ones_like(img1, dtype=bool)
    
    n = np.sum(mask)                                                            # number of possible seeds

    print('Number of seeds after preselection:', n)

    ## 2. Double check with different operators 
    if opt_meth == 'DoH' or opt_meth == 'Ly':
        # https://doi.org/10.16251/j.cnki.1009-2307.2012.01.033
        res = cal_doh(img1, sigma=1) if opt_meth == 'DoH' else cal_ly(img1, sigma=1)[0] # operator results (h, w)
        mask = mask & is_local_topk(res, k=-1, connectivity=connectivity)
    elif opt_meth == 'CGC':
        # combined gradient and curvature
        res1, res2 = cal_gcm(img1, 'PGCM', size=patch_size, sigma=0.2), cal_doh(img1, sigma=1.5)
        res2[mask] *= res1[mask]
        mask = mask & (res1 > 0.3) & is_local_topk(res2, k=-2, connectivity=connectivity)
    else:
        pass

    ## 3. Generate unique label for each seed
    coords = np.argwhere(mask)                                                  # coordinates of preselected seeds (_, 2)
    labels = np.arange(1, len(coords)+1).reshape(-1, 1)                         # initilize labels for each seeds (_, 1)
    n, seeds = merge_seeds(np.hstack([coords, labels]))                         # merge the label of connected seeds (_, 3)

    print('Number of seeds after operator double check:', n)

    ## 4. Gnerate binary image for later growth
    if True: # use local threshold if flag is True
        rr = wind_size // 2

        y, x = coords[:, 0], coords[:, 1]
        ymin, ymax = np.maximum(0, y - rr), np.minimum(h, y + rr + 1)           # top and bottom boundary
        xmin, xmax = np.maximum(0, x - rr), np.minimum(w, x + rr + 1)           # left and right boundary

        binary_img = np.zeros_like(img1, np.uint8)
        for y1, y2, x1, x2 in zip(ymin, ymax, xmin, xmax):    
            binary_img[y1:y2, x1:x2] = binary_img[y1:y2, x1:x2] | binarize_image(img2[y1:y2, x1:x2], thr_meth) #! careful, maybe overlap must use union
    else:
        binary_img = binarize_image(img2, thr_meth)

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

    ## 1. Enhance the input image if needed
    enhanced_img = enhance_image(img, ehc_meth, patch_size=5)
    # enhanced_img = enhance_image_multiscale(img, ehc_meth, patch_sizes=[3, 5, 7, 9])

    ## 2. Binarize the enhanced image, except for RG-based method
    binary_img = binarize_image(enhanced_img, thr_meth)

    ## 3. Label the connected regions in the binary image
    group_coords = []
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
