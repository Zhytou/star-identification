import os, cv2
import bisect as bis
import numpy as np
import scipy.ndimage as ndi
from skimage import img_as_ubyte, img_as_float
import skimage.filters as filters
import skimage.morphology as morph

from utils import cal_derivative, cal_doh, cal_ly, is_local_topk, is_near_local_max, gen_gaussian_kernel

eps = 1e-10
DEBUG = False

class UnionSet:
    '''
        Union set for connected components label.
    '''
    def __init__(self, size: int=0, arr: np.ndarray | list=[]):
        self.parent = {}
        self.rank = {}
        self.cnt = max(size, len(arr))
        for i in range(self.cnt):
            xi = arr[i] if len(arr) else i+1
            self.parent[xi] = xi
            self.rank[xi] = 0

    def find(self, x: int):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union_2(self, x: int, y: int):
        '''
            Union two labels x and y.
        '''
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
        '''
            Union a list of labels xs, and if y is not -1 union with it at the same time.
        '''
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


def con_shift_map(map: np.ndarray, size: int):
    '''
        Construct shifted mask and map.
    '''

    h, w = map.shape
    d = size  

    i, j =  np.indices((h, w))
    shift_mask = np.stack(
        [i >= 2 * d, i < h - 2 * d, j >= 2 * d, j < w - 2 * d, 
        (i >= 2 * d) & (j >= 2 * d), (i < h - 2 * d) & (j < w - 2 * d),
        (i < h - 2 * d) & (j >= 2 * d), (i >= 2 * d) & (j < w - 2 * d),],
        axis=0
    )                                                                                           # shift mask indicates valid image patch: True if the entire image patch stays within image bounds after shift
    shitfs = [(d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, -d), (-d, d), (d, -d)]             # shift offsets 
    shifted_map = np.stack([np.roll(map, shift, axis=(0, 1)) for shift in shitfs], axis=0)      # shifted map

    return shift_mask, np.maximum(shifted_map, eps)


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
    
    kth = 3
    trim = 2
    max_map = sorted_patches[..., -1]                                                           # max map (h, w)
    kmean_map = np.mean(sorted_patches[..., -kth:], axis=-1)                                    # kth top mean map (h, w)
    tmean_map = np.mean(sorted_patches[..., trim:-trim], axis=-1)                               # trimmed mean map
    mean_map = np.mean(flatten_patches, axis=-1)                                                # mean map (h, w)
    
    shift_mask = con_shift_map(img, size=d)[0]
    shifted_mean_map = con_shift_map(mean_map, size=d)[1]                                       # shifted mean map(neighbor patches' mean) (8, h, w)
    shifted_kmean_map = con_shift_map(kmean_map, size=d)[1]                                     # shifted kth mean map(neighbor patches' mean) (8, h, w)
    shifted_tmean_map = con_shift_map(tmean_map, size=d)[1]                                     # shifted trimmed mean map(neighbor patches' mean) (8, h, w)

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
        'Infrared Small Target Detection Utilizing the Multiscale Relative Local Contrast Measure http://ieeexplore.ieee.org/document/8289318/'
        measure = np.nanmin(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                np.clip(kmean_map[None, ...] / shifted_kmean_map - 1, 0, np.inf) * kmean_map,   # (8, h, w)
                np.nan
            ), axis=0
        )                                                                                       # relative local contrast measure (h, w)
    elif method == 'DLCM':
        'Difference-based Local Contrast Measure'
        measure = np.nanmax(
            np.where(
                shift_mask,                                                                     # (8, h, w)
                np.abs(mean_map[None, ...] - shifted_mean_map) * img[None, ...],                # (8, h, w)
                np.nan
            ), axis=0
        )
    else:
        measure = np.zeros_like(img, dtype=img.dtype)

    return measure


def cal_pcm(img: np.ndarray, method: str, size: int):
    '''
        Calculate patch-based contrast measure.
    '''

    h, w = img.shape
    d = size                                                                                    # patch size
    r = size // 2                                                                               # half of patch size
    
    fimg = img.astype(np.float64)                                                               # change data type to avoid overflow
    padded_img = np.pad(fimg, ((r, r), (r, r)))                                                 # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)
    flatten_patches = np.reshape(patches, (h, w, -1))
    sorted_pactches = np.sort(flatten_patches, axis=-1)

    mean_map = np.mean(patches, axis=(-1, -2))                                                  # mean map (h, w)
    shift_mask, shifted_mean_map = con_shift_map(mean_map, size=d)                              # shifted mean map(neighbor patches' mean) (8, h, w)
    
    mean_residual_map = mean_map[None, ...] - shifted_mean_map                                  # mean residual map (8, h, w)
    bright_mask = mean_residual_map > 0                                                         # bright target mask
    mean_residual_map = np.where(shift_mask & bright_mask, mean_residual_map, np.nan)           # apply mask and set invalid/negative to NaN

    structural_response = np.nanmin(
        np.stack([
            mean_residual_map[0, ...] * mean_residual_map[1, ...],                          # vertical line
            mean_residual_map[2, ...] * mean_residual_map[3, ...],                          # horizontal line
            mean_residual_map[4, ...] * mean_residual_map[5, ...],                          # main diagonal
            mean_residual_map[6, ...] * mean_residual_map[7, ...],                          # anti-diagonal
        ], axis=0),
        initial=np.inf,                                                                     # avoid four np.nan
        axis=0
    )
    valid = structural_response != np.inf

    if method == 'PCM':
        'Multiscale patch-based contrast measure for small infrared target detection https://linkinghub.elsevier.com/retrieve/pii/S0031320316300358'
        measure = np.where(valid, structural_response, eps)
        # selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        # measure = morph.erosion(measure, footprint=selem)
    elif method == 'IPCM':
        'An Improved Multiscale Patch-Based Contrast Measure for Small Infrared Target Detection https://ieeexplore.ieee.org/document/10065630'
        sigma = 1.5
        measure = np.where(
            valid, 
            img * (1 - np.exp(-(structural_response / sigma)**2)), 
            eps
        )
    else:
        measure = np.zeros_like(img, dtype=img.dtype)
    
    return measure


def cal_gcm(img: np.ndarray, method: str, size: int=5, sigma: float=1):
    '''
        Calculate gradient based/enhanced local contrast measure.
    '''
    h, w = img.shape
    d = size
    r = size // 2

    padded_img = np.pad(img, ((r, r), (r, r)))                                                  # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)
    flatten_patches = np.reshape(patches, (h, w, -1))                                           # flatten patches (h, w, d²)

    if True:
        grad_x = cal_derivative(img, order=(0, 1), sigma=sigma)                                 # gradient x map (h, w)
        grad_y = cal_derivative(img, order=(1, 0), sigma=sigma)                                 # gradient y map (h, w)
    else:
        grad_x = ndi.sobel(img, axis=1)
        grad_y = ndi.sobel(img, axis=0)
    padded_grad_x, padded_grad_y = np.pad(grad_x, ((r, r), (r, r))), np.pad(grad_y, ((r, r), (r, r)))  # padded gradient x and gradient y map (h, w, d, d)

    grad_patches = np.stack([
        np.lib.stride_tricks.sliding_window_view(padded_grad_x, (d, d)), 
        np.lib.stride_tricks.sliding_window_view(padded_grad_y, (d, d))
    ], axis=-1)                                                                                 # gradient patches (h, w, d, d, 2)
    grad_patches = grad_patches / np.maximum(np.linalg.norm(grad_patches, axis=-1, keepdims=True), eps)

    if method == 'GCM':
        'Gradient Consistency Measure'
        y, x = np.indices((d, d))                                                               # !careful first row index, then column index
        radial = np.stack([r - x, r - y], axis=-1)                                              # radial vectors (d, d, 2)
        rnorm = np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), eps)                 # radial vectors' norm
        radial = radial / rnorm

        dot_product = np.sum(grad_patches * radial[None, None, ...], axis=-1)                   # dot product (h, w, d, d)
        measure = np.clip(np.sum(dot_product, axis=(-2, -1)) / (d**2 - 1), 0, 1)
    elif method == 'PGCM': 
        'Patch Based Gradient Consistency Measure'
        max_indexs = np.argmax(flatten_patches, axis=-1)                                        # the index of maximum values
        y0, x0 = max_indexs // d, max_indexs % d                                                # the local coordinates(row, column) of each maximum values(h, w)
        y, x = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
        radial = np.stack([
            x0[..., None, None] - x[None, None, ...],
            y0[..., None, None] - y[None, None, ...]
        ], axis=-1)                                                                             # radial vectors (h, w, d, d, 2)
        rnorm = np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), eps)
        radial = radial / rnorm

        dot_product = np.sum(grad_patches * radial, axis=-1)                                    # dot product (h, w, d, d)
        measure = np.clip(np.sum(dot_product, axis=(-2, -1)) / (d**2 - 1), 0, 1)
    elif method == 'Lu-GCM':
        '基于梯度特征的弱小目标检测 https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2022&filename=JGHW202201021'
        grad = np.where(np.logical_and(grad_x > 0, grad_y > 0), grad_x * grad_y, 0)
        measure = cal_lcm(grad, 'DLCM', size)
    elif method == 'Zhang-GCM':
        '天基星图预处理技术研究 https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFQ&dbname=CJFDLAST2024&filename=JGDJ202412041'
        ## 0. Sobel operator
        grad = np.hypot(grad_x, grad_y)

        ## 1. Morphology close
        kernel = np.zeros((size, size), dtype=bool)
        kernel[:, r] = True
        kernel[r, :] = True
        grad = morph.closing(grad, footprint=kernel)

        # ## 2. Edge suppression via 4-neighbor rule
        T = cal_threshold(img, 'Liebe5')
        count = ndi.convolve(img < T, kernel)
        measure = np.where(count < 2, grad, 0)
    elif method == 'Mine-GCM':
        measure = cal_gcm(img, 'GCM', size, sigma) * cal_doh(img, sigma)
    else:
        measure = np.zeros_like(img)

    return measure


def cal_morph(img: np.ndarray, method: str, size: int=7, margin: int=2):
    '''
        Calculate response under morphology operations.
    '''
    assert (size % 2 == 1 and margin % 2 == 0)

    h, w = img.shape
    d, dd = size, margin
    r, rr = size // 2, margin // 2

    if method == 'Morph':
        selem = np.ones((d, d), dtype=bool)                                                     # structural element
        response = morph.white_tophat(img, footprint=selem)                                     # enhanced image based on white top-hat

    elif method == 'Jiang-Morph':
        'Robust and accurate star segmentation algorithm based on morphology http://opticalengineering.spiedigitallibrary.org/article.aspx?doi=10.1117/1.OE.55.6.063101'
        selem_m = np.ones((d + dd, d + dd), dtype=bool)                                         # structural element margin
        selem_m[rr:-rr, rr:-rr] = 0
        selem_e = np.ones((d, d), dtype=bool)                                                   # structural element
        selem_s = np.ones((2, 2), dtype=bool)                                                   # structural element

        ## 1. Star Detection
        response1 = morph.erosion(morph.dilation(img, selem_m), selem_e)
        ## 2. Noise suppression
        response2 = morph.dilation(morph.erosion(img, selem_s), selem_s)
        ## 3. Combine two responses with modified top-hat
        response = response2 - np.minimum(response1, response2)
    elif method == 'Xu-Morph':
        'Stray Light Elimination Method Based on Recursion Multi-Scale Gray-Scale Morphology for Wide-Field Surveillance https://ieeexplore.ieee.org/document/9333588'
        selem_d = np.pad(morph.disk(radius=r, dtype=bool), ((dd, dd), (dd, dd)))                        # structural element for dilation
        selem_e = morph.disk(radius=r, dtype=bool)                                                      # structural element for erosion
        response = np.minimum(img, morph.erosion(morph.dilation(img, selem_d), selem_e))
    elif method == 'Xi-Morph':
        '基于局部对比度的自适应Top-Hat红外小目标检测 http://www.opticsjournal.net/Articles/OJf9b777891546d03b/FullText'
        selem_d = np.pad(morph.disk(radius=r, dtype=bool), ((dd, dd), (dd, dd)))                        # structural element for dilation
        selem_e = morph.disk(radius=r-1, dtype=bool)                                                    # structural element for erosion

        ## Modified top-hat response
        response = img - np.minimum(img, morph.erosion(morph.dilation(img, selem_d), selem_e))
    else:
        response = np.zeros_like(img)

    return response


def cal_threshold(img: np.ndarray, method: str):
    '''
        Calculate the segmentation threshold.
    '''

    if method == 'Otsu':
        T = filters.threshold_otsu(img)
    elif method.startswith('Liebe'):
        k = float(method[5:])
        mean = np.mean(img)
        std = np.std(img)
        T = mean + k * std
    
    return T


def binarize_image(img: np.ndarray, method: str, size: int=5):
    '''
        Binarize the grayscale image.
    '''
    assert img.dtype == np.uint8

    d = size
    r = d // 2
    padded_img = np.pad(img, ((r, r), (r, r)))
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))
    max_val = 255 if img.dtype == np.uint8 else 1.0

    fimg = img.astype(np.float64)
    bimg = np.zeros_like(img, dtype=np.uint8)
    if method == 'Otsu' or method.startswith('Liebe'):
        T = cal_threshold(img, method)
        bimg[img >= T] = 1
    elif method == 'Zhang':
        '杂光背景下星点提取与星图识别技术的研究 https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201902&filename=1019159575.nh'
        n = -1 # connected region count
        while True:
            fimg = np.maximum(fimg - np.sum(fimg) / fimg.size, 0)
            bimg[fimg > 0] = 1
            nn = cv2.connectedComponents(bimg, connectivity=4)[0]
            if n != -1 and abs(n - nn) < 5:# and n < 30:
                break
            fimg = fimg / fimg.max() * max_val
            n = nn
    elif method == 'Xu':
        'A novel star image thresholding method for effective segmentation and centroid statistics https://linkinghub.elsevier.com/retrieve/pii/S0030402613002490'
        T = cal_threshold(img, 'Liebe3')
        delta = 0.2
        while True:
            mean1, mean2 = np.mean(img[img >= T]), np.mean(img[img < T])
            Tn = (1 - delta) * mean1 + (1 + delta) * mean2
            if abs(Tn - T) < 10:
                break
            T = Tn
        bimg[img > T] = 1
    elif method == 'Xiao':
        'Entropic thresholding based on gray-level spatial correlation histogram https://ieeexplore.ieee.org/document/4761626/?arnumber=4761626'
        ## 0. Predefined parameters
        gray_diff = 10
        gray_step = 8
        num_step = 1
        
        ## 1. Initialize f(x, y) and g(x, y)
        # f(x, y) is the gray value of the pixel located at the point (x, y) in a digital image.
        # g(x, y) is the number of the pixels of which the gray value is close to it in the corresponding N × N neighborhood,
        f_xy = img        
        g_xy = np.sum(patches - img[:, :, None, None] < gray_diff, axis=(-2, -1))

        ## 2. Construct joint histogram h(k, m) 
        # h(k, m) = prob(f(x, y) == k and g(x, y) == m)
        k_max = 256
        m_max = d**2

        # h(x, y) is the histgram function
        h_xy = np.zeros((k_max, m_max), np.float64)
        np.add.at(h_xy, (f_xy.flatten(), g_xy.flatten()), 1)
        h_xy /= np.sum(h_xy)

        ## 3. Get the optimal threshold for f(x, y) and g(x, y)
        k_vals, m_vals = np.arange(0, k_max, gray_step), np.arange(0, m_max, num_step)
        k, m = np.meshgrid(k_vals, m_vals, indexing='ij')
        #TODO: optimal threshold calculation based on local entropy
        T_k_opt, T_m_opt = 0, 0

        ## 4. Do segmentation
        bimg[(f_xy > T_k_opt) | (g_xy > T_m_opt)] = 1

    return bimg


def enhance_image(img: np.ndarray, method: str, size: int | list[int]=3, preserve_dtype: bool=True):
    '''
        Enhance the image.
    '''

    if method.startswith('MS_'):                                                                # enhance image under multiscale
        assert type(size) is list 
        enhanced_imgs = np.stack([enhance_image(img, method[3:], size_i, preserve_dtype) for size_i in size])
        return np.max(enhanced_imgs, axis=0)

    d = size                                                                                    # patch size
    r = size // 2                                                                               # half of patch size
    max_val = 255 if img.dtype == np.uint8 else 1.0

    padded_img = np.pad(img, ((r, r), (r, r)), 'reflect')                                       # padded raw image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))                      # raw patches (h, w, d, d)

    if method.endswith('LCM'):
        enhanced_img = cal_lcm(img, method, size=d)
    elif method.endswith('PCM'):
        enhanced_img = cal_pcm(img, method, size=d)
    elif method.endswith('GCM'):
        enhanced_img = cal_gcm(img, method, size=d, sigma=1)
    elif method.endswith('Morph'):
        enhanced_img = cal_morph(img, method, size=d)
    elif method == 'BEF':
        '星敏感器抗杂光背景滤波图像处理方法研究 https://doi.org/10.19328/j.cnki.1006-1630.2016.04.005'
        kernel = gen_gaussian_kernel(sigma=1, size=d)                                           # default gaussian kernel with a size of d
        kernel[1:-1, 1:-1] = 0                                                                  # zero out the inner region of gaussian kernel to estimate the bacground
        kernel[0, 0], kernel[0, -1], kernel[-1, 0], kernel[-1, -1] = kernel[0, 0] + 1, kernel[0, -1] + 1, kernel[-1, 0] + 1, kernel[-1, -1] + 1 # emphasize corner contributions for background estimation
        kernel /= kernel.sum()                                                                  # normalize the kernel
        enhanced_img = np.clip((img - np.sum(patches * kernel[None, None, ...], axis=(-1, -2))), 0, max_val) # background estimation via convolution and subtract
    elif method == 'Max-Median' or method == 'Max-Mean':
        'Max-Mean and Max-Median filters for detection of small-targets http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=905421'
        opt_mask = np.zeros((4, d, d), dtype=bool)                                              # operation mask
        opt_mask[0, :, r] = True                                                                # vertical line
        opt_mask[1, r, :] = True                                                                # horizontal line
        opt_mask[2, np.arange(0, d), np.arange(0, d),] = True                                   # main diagonal
        opt_mask[3, np.arange(0, d)[::-1], np.arange(0, d)] = True                              # anti-diagonal

        if method == 'Max-Median':
            median_map = np.stack([np.median(patches[:, :, mask], axis=-1) for mask in opt_mask], axis=-1)
            enhanced_img = np.maximum(img - np.max(median_map, axis=-1), eps)
        else:
            mean_map = np.stack([np.mean(patches[:, :, mask], axis=-1) for mask in opt_mask], axis=-1)
            enhanced_img = np.maximum(img - np.max(mean_map, axis=-1), eps)
    else: # method == 'None'
        return img

    if preserve_dtype and enhanced_img.dtype != img.dtype:
        enhanced_img = (enhanced_img / np.max(enhanced_img, initial=eps) * max_val).astype(img.dtype)

    return enhanced_img


def initialize_seeds(img: np.ndarray, method: str, size: int=5, connectivity: int=4):
    '''
        Initialize seeds for region growth.
    '''

    d = size
    r = size // 2
    padded_img = np.pad(img, ((r, r), (r, r)))                                  # padded image
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))      # image patches (h, w, d, d)

    ## 1. Preselect
    #!NOTE: only use local rank_filter, because the global background threshold might be too high under starry light interference
    std = np.std(img)
    mask = is_local_topk(img, -2, connectivity) | is_near_local_max(img, std, connectivity) # possible seed mask (h, w)
    if d >= 5:                                                                  # check the mean gray of inner ring is higher than the outter one
        rr = d // 4
        inner = np.zeros((d, d), dtype=bool)
        inner[r-rr:r+rr+1, r-rr:r+rr+1] = True
        omean = np.maximum(np.mean(patches[..., ~inner], axis=-1), eps)         # outer ring mean map
        inner[r, r] = False                                                     # exclude central pixel
        imean = np.maximum(np.mean(patches[..., inner], axis=-1), eps)          # inner ring mean map 
        mask = mask & (imean - 1. * omean >= 0)
    if DEBUG:
        print('Number of seeds after preselection:', np.sum(mask))

    ## 2. Double check with different operators 
    if method == 'DoH' or method == 'Ly':
        # https://doi.org/10.16251/j.cnki.1009-2307.2012.01.033
        res = cal_doh(img, sigma=1) if method == 'DoH' else cal_ly(img, size, sigma=1)[1] # operator results (h, w)
        mask = mask & is_local_topk(res, k=-1, connectivity=connectivity)
    elif method == 'Cgc':
        # combined gradient and curvature
        res1, res2 = cal_gcm(img, 'GCM', size=d, sigma=1), cal_doh(img, sigma=1)
        res2[mask] *= res1[mask]
        mask = mask & (res1 > 0.3) & is_local_topk(res2, k=-2, connectivity=connectivity)
    else:
        mask = np.zeros_like(mask, dtype=bool)
    if DEBUG:
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
    assert label_cnt == 0 or labels.min() >= 1 and labels.max() <= label_cnt
    if DEBUG:
        print('Number of seeds after merging:', label_cnt)

    return coords, labels


def region_growth_label(img: np.ndarray, coords: np.ndarray, labels: np.ndarray, connectivity: int=4, steps: int=4):
    '''
        Label the image with region growth.
    '''

    assert np.all((img == 0) | (img == 1))

    h, w = img.shape
    ## 1. Retain only foreground seeds
    valid = img[coords[:, 0], coords[:, 1]] == 1
    coords, labels = coords[valid], labels[valid]

    ## 2. Do region growth on binary image, namely breadth first search
    label_img, label_tab = np.zeros_like(img, np.uint32), UnionSet(arr=np.unique(labels)) #!NOTE: label equivalence table must be initialized with `np.unique(labels)` because the labels may no longer form a contiguous sequence after foreground check
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

        # find duplicate coordinates and merge corresponding labels
        ncoords_view = ncoords.view([('', ncoords.dtype)] * 2)                  # structural array for duplicate search
        _, uniuqe_idx, inverse_idx = np.unique(ncoords_view, return_index=True, return_inverse=True)

        ulabels = nlabels[uniuqe_idx]                                           # labels of unique coordinates
        for group_id in range(len(ulabels)):
            mask_group = (inverse_idx == group_id)
            clabels = nlabels[mask_group]                                       # connected labels, namely the labels with same inverse mapping index
            if len(clabels) == 1:
                continue
            label_tab.union_l(clabels)
        
        coords, labels = ncoords, nlabels                                       # update current coordinates and labels     
        steps -= 1

    ## 3. Allocate new unique labels in [1, label_tab.count()] for each seed
    #!NOTE: due to foreground and duplicate check, seed labels may no longer stay within the range of [1, label_tab.count()].
    #!NOTE: thus compact relabeling is applied to restore the valid range.
    if label_img.max() > label_tab.count():
        label_cnt, label_map = 0, {}                                            # unique label counter(for double check) and label mapping(for compression)
        ulabels = np.unique(label_img[label_img != 0])
        for label in ulabels:
            rlabel = label_tab.find(label)                                      # root label for each seed
            if rlabel in label_map:
                label_img[label_img == label] = label_map[rlabel]
            else:
                label_cnt += 1
                label_img[label_img == label] = label_cnt
                label_map[rlabel] = label_cnt
        assert label_cnt == label_tab.count() and label_img.max() <= label_cnt

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

    ## 2. Second Pass: allocate new unique labels in [1, label_tab.count()] for each coord
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
        run['label'] = label_tab.find(run['label'])
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


def find_ranges(nums: np.ndarray, threshold: int=0):
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


def group_star(img: np.ndarray, method: list[str], size: int | list[int]=5, wind: int=19, connectivity: int=4, pixel_limit: int=5, output_dir: str=None):
    '''
        Group the potential star in the image.
    '''
    h, w = img.shape
    ehc_meth, thr_meth, lab_meth, opt_meth = method    

    ## 1. Generate seeds if opt_meth is valid
    if opt_meth != 'None':
        coords, labels = initialize_seeds(img, opt_meth, size=size, connectivity=connectivity)
    else:
        coords, labels = np.array([]), np.array([])
    
    ## 2. Enhance the input image, and then binarize the enhanced image
    enhanced_img = enhance_image(img, ehc_meth, size=size)
    if len(coords) > 0:
        assert coords.shape[1] == 2
        
        y, x = coords[:, 0], coords[:, 1]
        ymin, ymax = np.maximum(0, y - wind // 2), np.minimum(h, y + wind // 2 + 1)             # top and bottom boundary
        xmin, xmax = np.maximum(0, x - wind // 2), np.minimum(w, x + wind // 2 + 1)             # left and right boundary

        binary_img = np.zeros_like(img, dtype=np.uint8)
        for y1, y2, x1, x2 in zip(ymin, ymax, xmin, xmax):
            binary_img[y1:y2, x1:x2] |= binarize_image(enhanced_img[y1:y2, x1:x2], thr_meth, size=5) #!NOTE: use union to avoid overlap
    else:
        binary_img = binarize_image(enhanced_img, thr_meth, size=5)

    ## 3. Label the connected regions in the binary image
    group_coords = []
    if lab_meth == 'RAW':
        mask = binary_img[coords[:, 0], coords[:, 1]] > 0
        coords, labels = coords[mask], labels[mask]
        for ulabel in np.unique(labels):
            group_coords.append((coords[labels==ulabel, 0], coords[labels==ulabel, 1]))
        return group_coords
    if lab_meth == 'RGL':
        'Region Growth Label'
        n, label_img = region_growth_label(binary_img.copy(), coords, labels, connectivity)
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
            pass
            # print(i, rows, cols)

    ## 5. Save intermediate results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'img_d.png'), img)
        cv2.imwrite(os.path.join(output_dir, 'img_e.png'), enhanced_img)
        cv2.imwrite(os.path.join(output_dir, 'img_ee.png'), np.where(enhanced_img == 0, 0, 255))
        cv2.imwrite(os.path.join(output_dir, 'img_b.png'), binary_img * 255)
    return group_coords
