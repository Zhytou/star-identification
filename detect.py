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

    def union_2(self, x: int, y: int):
        x0, y0 = self.find(x), self.find(y)             # parent of x and y
        if x0 != y0:
            if self.rank[x0] < self.rank[y0]:           # ensure x0 has higher or equal rank
                x0, y0 = y0, x0
            if self.rank[x0] == self.rank[y0]:          # increment rank when merging trees of equal height
                self.rank[x0] += 1
            self.cnt -= 1                               # one less connected component after merging
            self.parent[y0] = x0 
        return x0

    def union_l(self, xs: list | np.ndarray, y: int=-1):
        assert len(xs) >= 1
        x0 = self.find(xs[0]) if y == -1 else self.find(y)
        for x in xs:
            x0 = self.union_2(x0, x)
        return x0

    def add(self, x: int):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.cnt += 1
        return x
    
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
    elif method == 'PCM':
        'Multiscale patch-based contrast measure for small infrared target detection https://linkinghub.elsevier.com/retrieve/pii/S0031320316300358'
        mean_residual_map = mean_map[None, ...] - shifted_mean_map
        positive_mask = mean_residual_map > 0                                                   # only retain position mean residual (bright target)
        mean_residual_map = np.where(shift_mask & positive_mask, mean_residual_map, np.nan)     # apply mask and set invalid/negative to NaN
        measure = np.nanmin(
            np.stack([
                mean_residual_map[0, ...] * mean_residual_map[1, ...],                          # vertical line
                mean_residual_map[2, ...] * mean_residual_map[3, ...],                          # horizontal line
                mean_residual_map[4, ...] * mean_residual_map[5, ...],                          # main diagonal
                mean_residual_map[6, ...] * mean_residual_map[7, ...],                          # anti-diagonal
            ], axis=0),
            initial=np.inf,                                                                     # avoid corner to be np.nan
            axis=0
        )
        measure[measure == np.inf] = eps
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


def binarize_image(img: np.ndarray, method: str):
    '''
        Binarize the grayscale image.
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
        #TODO: optimal threshold calculation based on local entropy
        T_k_opt, T_m_opt = 0, 0

        ## 4. Do segmentation
        binary_img[(f > T_k_opt) | (g > T_m_opt)] = 1

    return binary_img


def binarize_image_local(img: np.ndarray, method: str, coords: np.ndarray, size: int=21):
    '''
        Perform local adaptive binarization around given coordinates.
    '''
    assert img.dtype == np.uint8 and coords.shape[1] == 2

    h, w = img.shape
    d = size
    r = d // 2

    y, x = coords[:, 0], coords[:, 1]
    ymin, ymax = np.maximum(0, y - r), np.minimum(h, y + r + 1)           # top and bottom boundary
    xmin, xmax = np.maximum(0, x - r), np.minimum(w, x + r + 1)           # left and right boundary

    binary_img = np.zeros_like(img, dtype=np.uint8)
    for y1, y2, x1, x2 in zip(ymin, ymax, xmin, xmax):    
        binary_img[y1:y2, x1:x2] = binary_img[y1:y2, x1:x2] | binarize_image(img[y1:y2, x1:x2], method) #! careful, maybe overlap must use union
    return binary_img


def enhance_image(img: np.ndarray, method: str, size: int=3, preserve_dtype: bool=True):
    '''
        Enhance the image.
    '''

    d = size                                                                                    # patch size
    r = size // 2                                                                               # half of patch size
    max_val = 255 if img.dtype == np.uint8 else 1.0

    padded_img = np.pad(img, ((r, r), (r, r)))                                                  # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)

    if method in ['LCM', 'ILCM', 'NLCM', 'RLCM', 'MLCM', 'PCM']:
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
    elif method == 'BE':
        '星敏感器抗杂光背景滤波图像处理方法研究 https://doi.org/10.19328/j.cnki.1006-1630.2016.04.005'

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


def enhance_image_multiscale(img: np.ndarray, method: str, sizes: list[int]):
    '''
        Enhance the image under multiscale.
    '''
    assert method.startswith('MS_')
    enhanced_imgs = np.stack([enhance_image(img, method[3:], size) for size in sizes])

    return np.max(enhanced_imgs, axis=0)


def initialize_seeds(img: np.ndarray, method: str, size: int=5, connectivity: int=4):
    '''
        Initialize seeds for region growth.
    '''

    d = size
    r = size // 2
    padded_img = np.pad(img, ((r, r), (r, r)), mode='constant')                 # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))      # image patches (h, w, d, d)
    max_intensity = 255 if img.dtype == 255 else 1.0                            # max intensity for image data type

    ## 1. Preselect
    #! only use local rank_filter, because the global background threshold might be too high under starry light interference
    #! also, avoid preselection when noise is relatively high
    mask = is_local_topk(img, -2, connectivity) | is_near_local_max(img, connectivity) # possible seed mask (h, w)
    if d >= 5:                                                                  # check the mean gray of inner ring is higher than the outter one
        rr = d // 4
        inner = np.zeros((d, d), dtype=bool)
        inner[r-rr:r+rr+1, r-rr:r+rr+1] = True
        omean = np.maximum(np.mean(patches[..., ~inner], axis=-1), eps)         # outer ring mean map
        inner[r, r] = False                                                     # exclude central pixel
        imean = np.maximum(np.mean(patches[..., inner], axis=-1), eps)          # inner ring mean map 
        mask = mask & ((img == max_intensity) | (img - 1. * imean >= 0)) & (imean - 1. * omean >= 0)
    print('Number of seeds after preselection:', np.sum(mask))

    ## 2. Double check with different operators 
    if method == 'DoH' or method == 'Ly':
        # https://doi.org/10.16251/j.cnki.1009-2307.2012.01.033
        res = cal_doh(img, sigma=1) if method == 'DoH' else cal_ly(img, sigma=1)[0] # operator results (h, w)
        mask = mask & is_local_topk(res, k=-1, connectivity=connectivity)
    elif method == 'CGC':
        # combined gradient and curvature
        res1, res2 = cal_gcm(img, 'PGCM', size=d, sigma=0.2), cal_doh(img, sigma=1.5)
        res2[mask] *= res1[mask]
        mask = mask & (res1 > 0.3) & is_local_topk(res2, k=-2, connectivity=connectivity)
    else:
        mask = np.zeros_like(mask, dtype=bool)
    print('Number of seeds after operator double check:', np.sum(mask))

    ## 3. Generate unique label for each seed, namely merge possible connected seeds
    n = np.sum(mask)                                                            # number of initial unconnected seeds
    label_tab = UnionSet(n)                                                     # label equivalence table
    coords, labels = np.argwhere(mask), np.arange(1, n + 1)                     # coordinates and initial labels
    offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) if connectivity == 4 else np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    # First Pass: construct label equivalence table
    for i in range(n):
        ncoordi = coords[i, None, :] + offsets                                  # neighboring coordinates for i-th seed 
        mask = np.isin(coords[:, 0], ncoordi[:, 0]) & np.isin(coords[:, 1], ncoordi[:, 1])  # connected mask
        if not np.any(mask):
            continue
        clabels = labels[np.nonzero(mask)[0]]                                   # connected labels
        labels[i] = label_tab.union_l(clabels, labels[i])
    
    # Second Pass: allocate new unique labels in [1, label_tab.count()] for each seed
    label_cnt, label_map = 0, {}                                                # unique label counter(for double check) and label mapping(for compression)
    for i in range(n):
        labels[i] = label_tab.find(labels[i])
        if labels[i] in label_map:
            labels[i] = label_map[labels[i]]
        else:
            label_cnt += 1
            label_map[labels[i]] = label_cnt
            labels[i] = label_cnt
    assert label_cnt == label_tab.count(), ''
    print('Number of seeds after merging:', label_cnt)

    return coords, labels


def region_growth_label(img: np.ndarray, coords: np.ndarray, labels: np.ndarray, connectivity: int=4, steps: int=4):
    '''
        Label the image with region growth.
    '''

    h, w = img.shape

    ## 1. Retain only foreground seeds
    valid = img[coords[:, 0], coords[:, 1]] != 0
    coords, labels = coords[valid], labels[valid]

    ## 2. Do region growth on binary image, namely breadth first search
    n = len(np.unique(labels))
    label_img, label_tab = np.zeros_like(img, np.uint32), UnionSet(n)           # label image and label equivalence table
    offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) if connectivity == 4 else np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    while steps > 0 and len(coords) > 0:
        assert np.all(img[coords[:, 0], coords[:, 1]] == 1) and len(coords) == len(labels)
        img[coords[:, 0], coords[:, 1]] = 0                                     # mark the visited
        label_img[coords[:, 0], coords[:, 1]] = labels                          # assign the labels

        # broadcast to get the neighboring seeds
        ncoords, nlabels = np.reshape(coords[:, None, :] + offsets[None, ...], (-1, 2)), np.repeat(labels, connectivity) # neighboring coordinates and labels
        valid = (
            (ncoords[:, 0] >= 0) & (ncoords[:, 0] < h) & 
            (ncoords[:, 1] >= 0) & (ncoords[:, 1] < w)
        )                                                                       # boundary check
        ncoords, nlabels = ncoords[valid], nlabels[valid]
        
        valid = img[ncoords[:, 0], ncoords[:, 1]] != 0                          # foreground and unvisited check
        ncoords, nlabels = ncoords[valid], nlabels[valid]

        #TODO: add overlap check
        # find duplicate coordinates and merge label

        coords, labels = ncoords, nlabels                                       # update current coordinates and labels     
        steps -= 1

    return label_tab.count(), label_img


def connected_components_label(img: np.ndarray, connectivity: int=4):
    '''
        Label the connected regions with two-pass method.
    '''

    h, w = img.shape
    label_img = np.zeros_like(img, dtype=np.uint32)
    label_cnt, label_tab = 0, UnionSet()                    # unique label counter and label equivalence table

    ## 1. First Pass: construct label equivalence table
    offsets = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]) if connectivity == 4 else np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
    coords = np.argwhere(img)                               # coordinates of nonzero pixels
    for coord in coords:
        ncoord = coord[None, :] + offsets                   # coordinates of neighbor pixels
        
        valid = (ncoord[...,0] >= 0) & (ncoord[..., 0] < h) & (ncoord[..., 1] >= 0) & (ncoord[..., 1] < w) # boundary check
        ncoord = ncoord[valid]
        
        valid = img[ncoord[:, 0], ncoord[:, 1]] != 0        # foreground check
        ncoord = ncoord[valid]

        nlabel = label_img[ncoord[:, 0], ncoord[:, 1]]      # labels of neighbor pixels
        if np.all(nlabel == 0):
            label_cnt += 1
            label_img[coord[0], coord[1]] = label_tab.add(label_cnt)
        else:
            nlabel = nlabel[nlabel != 0]
            label_img[coord[0], coord[1]] = label_tab.union_l(nlabel)

    ## 2. Second Pass: allocate new unique labels in [1, label_tab.count()] for each seed
    label_cnt, label_map = 0, {}                            # unique label counter(reset for double check) and label mapping(for compression)
    for coord in coords:
        label = label_tab.find(label_img[coord[0], coord[1]])
        if label in label_map:
            label_img[coord[0], coord[1]] = label_map[label]
        else:
            label_cnt += 1
            label_img[coord[0], coord[1]] = label_cnt
            label_map[label] = label_cnt
    assert label_cnt == label_tab.count(), ''

    return label_cnt, label_img


def run_length_code_label(img: np.ndarray, connectivity: int=4):
    '''
        Label the connected regions with run length code.
    '''
      
    label_cnt, label_tab = 0, UnionSet()    # unique label counter and label equivalence table

    def gen_curr_run(row: int, beg: int, end: int):
        '''
            Generate the current run.
        '''
        nonlocal label_cnt, label_tab
        
        run = {'row': row, 'beg': beg, 'end': end, 'label': -1}
        clabels = []

        ## 1. Use binary search to find the potential connected labels in the previous runs
        idx = bis.bisect_left(prev_runs, run['beg'], key=lambda x: x['end'])
        if idx < len(prev_runs):
            for prev_run in prev_runs[idx:]:
                if prev_run['beg'] > end:
                    break
                if connectivity == 4:
                    overlap = (prev_run['beg'] <= end) and (prev_run['end'] >= beg)
                else:
                    overlap = (prev_run['beg'] <= end + 1) and (prev_run['end'] >= beg - 1)
                if overlap:
                    clabels.append(prev_run['label'])
        
        ## 2. Merge the connected labels
        if len(clabels) == 0:
            label_cnt += 1
            run['label'] = label_tab.add(label_cnt)
        else:
            run['label'] = label_tab.union_l(clabels)
        
        return run

    h, _ = img.shape

    runs, prev_runs= [], []
    ## 1. Iterate on axis 0 of img to generate runs
    for row in range(h):
        if len(prev_runs) > 0 and prev_runs[0]['row'] != row - 1:               # set previous row runs to empty if already skip a row
            prev_runs = []

        curr_runs = []                                                          # current row runs
        col_ranges = find_ranges(img[row])                                      # nonzero column ranges
        curr_runs.extend([gen_curr_run(row, beg, end) for (beg, end) in col_ranges]) # construct row runs
        prev_runs = curr_runs
        runs.extend(curr_runs)
    runs = [dict(run, label=label_tab.find(run['label'])) for run in runs]

    ## 2. Iterate through runs to construct label image
    label_cnt, label_map = 0, {}                                                # unique label counter(reset for double check) and label mapping(for compression)
    label_img = np.zeros_like(img, np.uint32)
    for run in runs:
        if run['label'] in label_map:
            run['label'] = label_map[run['label']]
        else:
            label_cnt += 1
            label_map[run['label']] = label_cnt 
            run['label'] = label_cnt
        label_img[run['row'], run['beg']:run['end']+1] = run['label']
    assert label_cnt == label_tab.count()

    return label_cnt, label_img


def cross_project_label(img: np.ndarray, connectivity: int=4):
    '''
        Label the connected regions with cross projection method.
    '''
    label_cnt = 0                                           # unique label counter
    label_img = np.zeros_like(img, dtype=np.uint32)
    
    vproj = np.sum(img, axis=0)                             # vertical projection
    vranges = find_ranges(vproj)

    for (x1, x2) in vranges:
        hproj = np.sum(img[:, x1 : x2 + 1], axis=1)         # horizontal projection
        hranges = find_ranges(hproj)

        for (y1, y2) in hranges:
            roi = img[y1 : y2 + 1, x1 : x2 + 1]             # the region of interest
            cnt, label_roi = cv2.connectedComponents(roi, connectivity=connectivity)
            label_img[y1 : y2 + 1, x1 : x2 + 1] = label_roi + label_cnt
            label_cnt += cnt
    
    return label_cnt, label_img


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
        return []

    return np.vstack([begs, ends]).transpose()


def group_star(img: np.ndarray, method: list[str], connectivity: int=4, pixel_limit: int=5) -> list[tuple[np.ndarray, np.ndarray]]:
    '''
        Group the potential star in the image.
    '''
    ehc_meth, thr_meth, lab_meth, opt_meth = method    

    ## 1. Enhance the input image
    if ehc_meth.startswith('MS_'):
        enhanced_img = enhance_image_multiscale(img, ehc_meth, sizes=[3, 5, 7])
    else:
        enhanced_img = enhance_image(img, ehc_meth, size=5)

    ## 2. Binarize the enhanced image, except for RGL-based method
    binary_img = binarize_image(enhanced_img, thr_meth)

    ## 3. Label the connected regions in the binary image
    group_coords = []
    if lab_meth == 'RGL':
        'Region Growth Label'
        coords, labels = initialize_seeds(img, opt_meth, 5, connectivity)
        binary_img = binarize_image_local(enhanced_img, thr_meth, coords)
        n, label_img = region_growth_label(binary_img, coords, labels, connectivity)
    elif lab_meth == 'CCL' or lab_meth == 'DCCL':
        'Connected Components Label'
        n, label_img = connected_components_label(binary_img, connectivity) if lab_meth == 'DCCL' else cv2.connectedComponents(binary_img, connectivity)
    elif lab_meth == 'CPL':
        'Cross Projection Label'
        n, label_img = cross_project_label(binary_img, connectivity)
    elif lab_meth == 'RLC':
        'Run Length Code Connected Components Label'
        n, label_img = run_length_code_label(binary_img, connectivity)
    else:
        n, label_img = 0, binary_img

    ## 4. Separate the labelled region
    for i in range(n):
        rows, cols = np.nonzero(label_img == i + 1)
        if len(rows) >= pixel_limit and len(cols) >= pixel_limit:
            group_coords.append((rows, cols))
        else:
            print(rows, cols)

    print('n =', n, 'len_group_coords', len(group_coords))

    return group_coords
