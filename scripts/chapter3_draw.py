import os
import cv2
import numpy as np
import timeit
from math import radians

from simulate import create_star_image, add_gaussian_and_pepper_noise, add_stellar_noise, get_stellar_intensity
from denoise import denoise_image, denoise_with_blf
from detect import group_star, cal_threshold
from extract import get_star_centroids
from utils import find_overlap_and_unique, cal_mse_psnr_ssim, cal_doh


# 非局部均匀滤波测试——Lena图片
if False:
    img = cv2.imread(f'example/lena/lena.png', cv2.IMREAD_GRAYSCALE)

    imgs = {}
    imgs['noised'] = add_gaussian_and_pepper_noise(img, 0.00, 0.001, clipped=True)
    imgs['nlm'] = denoise_with_nlm(imgs['noised'])
    imgs['gaussian'] = filter_image(imgs['noised'], 'gaussian')
    imgs['median'] = filter_image(imgs['noised'], 'median')
    imgs['glp'] = filter_image(imgs['noised'], 'gaussian low pass')

    for key in imgs:
        cv2.imwrite(f'res/chapter3/lena/{key}.png', imgs[key])
        mse, psnr, ssim = cal_mse_psnr_ssim(img, imgs[key])
        print(key, mse, psnr, ssim)


ra, de, roll=radians(29.2104), radians(-12.0386), radians(0)
d=64
h, w=512, 512
x, y=188, 169*2
fov=12
limit_mag=6
background=9
psf=1


# 各种基础方法的降噪效果
if False:
    dir = 'res/chapter3/basis/'
    os.makedirs(dir, exist_ok=True)

    img0, stars = create_star_image(
        ra, de, roll,
        w=w, 
        h=h,
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        sigma_psf=psf,
        background=background
    )
    img1 = add_gaussian_and_pepper_noise(img0, 0.05, 0.005, clipped=True)

    for method in ['ORINGINAL', 'NOISED', 'MEAN', 'MEDIAN', 'GLF', 'WAVELET', 'NLM']:
        if method == 'ORINGINAL':
            img2 = img0
        elif method == 'NOISED':
            img2 = img1
        else:
            img2 = denoise_image(img1, method)
        cv2.imwrite(f'{dir}/{method}.png', img2)
        cv2.imwrite(f'{dir}/{method}_S.png', img2[y-d:y+d, x-d:x+d])

        mse, psnr, ssim = cal_mse_psnr_ssim(img0, img2)
        print(method, mse, psnr, ssim)


# 改进双边滤波测试
if False:    
    dir = 'res/chapter3/blf'
    os.makedirs(dir, exist_ok=True)

    img0, stars = create_star_image(
        ra, de, roll,
        w=w, 
        h=h,
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        sigma_psf=psf,
        background=background
    )
    img1 = add_gaussian_and_pepper_noise(img0, 0.05, 0.001, clipped=False)

    for method in ['NLM', 'BLF', 'MBLF']:
        if method == 'NLM':
            img2 = denoise_image(img1, 'NLM')
        else:    
            img2 = denoise_image(denoise_image(img1, 'NLM'), method)
        
        mse, psnr, ssim = cal_mse_psnr_ssim(img0, img2)
        print(method, mse, psnr, ssim)
        cv2.imwrite(f'{dir}/{method}.png', img2)


# 第三章末尾测试参数
ra, de, roll=radians(29.2104), radians(-12.0386), radians(0) # 可能每个测试拍摄视角不同
h, w=512, 512
fov=12
limit_mag=6
background=9
psf=1
roi=3
save=True
d=128

# 星图降噪效果测试（质量指标比较）
if False:
    img0, stars = create_star_image(
        ra, de, roll, 
        w=w, 
        h=h,
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        sigma_psf=psf,
        roi=roi,
        background=background,
        dtype=np.float32
    )

    dir = f'res/chapter3/denoise'
    for (g, p) in [
        (0.03, 0.003),
        (0.06, 0.006), 
        (0.09, 0.009), 
    ]:
        print(
            'DENOISE TEST'
            '\nSigma of gaussian noise:', g, 
            '\nProbability of pepper noise', p, 
            '\n--------------------------------'
        )

        if save:
            os.makedirs(f'{dir}/{g}_{p}', exist_ok=True)
            os.makedirs(f'{dir}/{g}_{p}/scale', exist_ok=True)

        img1 = add_gaussian_and_pepper_noise(img0, g, p, clipped=True)
        for method in [
            # 'ORINGINAL',
            'NOISED', 
            # 'MEDIAN', 'MEAN', 'GAUSSIAN,
            'BLF', 'NLM', 'AMF', 'WAVELET',
            # 'NLM_BLF',
            # 'EMF',
            # 'CWM', 
            # 'CMG',
            # 'CNB', 
        ]:
            if method == 'ORINGINAL':
                img2 = img0
            elif method == 'NOISED':
                img2 = img1
            else:
                img2 = denoise_image(img1, method)

            # 计算降噪前后图像质量指标
            mse, psnr, ssim = cal_mse_psnr_ssim(img0, img2)
            print(
                method,
                '\nPSNR:', psnr, 'SSIM:', ssim,
                '\n',
            )

            if save:
                img3 = img2 if img2.dtype == np.uint8 else 255*img2
                cv2.imwrite(f'{dir}/{g}_{p}/{method}.png', img3)
                cv2.imwrite(f'{dir}/{g}_{p}/scale/{method}.png', img3[y-d:y+d, x-d:x+d])


def cal_centroid_error(coords1: np.ndarray, coords2: np.ndarray):
    '''
        Calculate the centroid method error.
    '''
    coords1, coords2, _, _ = find_overlap_and_unique(coords1, coords2)
    # print(np.hstack([coords1, coords2]))

    assert len(coords1) == len(coords2), 'Error in find_overlap_and_unique!'
    error = np.mean(np.linalg.norm(coords1 - coords2, axis=1))
    n = len(coords1)

    return n, error


# 图像降噪对质心计算的影响
if False:
    num_test = 10

    img0, stars = create_star_image(
        ra, de, roll, 
        w=w, 
        h=h,
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        sigma_psf=psf,
        background=background
    )
    tot = len(stars)
    real_coords = stars[:, 1:3]

    # 打印视场内恒星信息
    print(np.histogram(stars[:, -1], range=(0, 6), bins=6))

    # 比较各个降噪方法后误差大小
    # 其中NONE为无预处理时
    for method in [
        'NONE', 
        'GAUSSIAN', 
        'MEAN', 
        # 'MEDIAN', 
        # 'BLF'
    ]:
        
        cnts, errs = [], []
        for _ in range(num_test):
            img1 = add_gaussian_and_pepper_noise(img0, sigma_g=0.1, prob_p=0.005)
            esti_coords = np.array(get_star_centroids(
                img1,
                method,
                'Liebe3',
                'CCL',
                'MCoG',
                pixel_limit=5,
                num_esti=3 
            ))

            # 正确提取质心数量和质心误差
            cnt, err = cal_centroid_error(real_coords, esti_coords)
            cnts.append(cnt)
            errs.append(err)

        print(
            'Method:', method,  
            '\nTotal number of stars for the test:', tot,
            '\nNumber of correct extracted stars for each test image:', cnts,
            '\nError for each test image:', errs,
            '\nAveragae error:', np.mean(errs),
            '\n--------------------------------'
        )


def label_image(img: np.ndarray, coords: np.ndarray, color: tuple=(0, 255, 0),  radius: int=5):
    '''
        Label image with colored circles.
    '''
    for coord in coords:
        row, col = int(coord[0]), int(coord[1])
        cv2.circle(img, (col, row), radius, color, 1)
    return img


def find_miss_idxs(coords: np.ndarray, miss_coords: np.ndarray, atol: float=1e-8, rtol: float=1e-5):
    '''
        Find the indexs of missing coordinates.
    '''
    # broadcast comparison
    mask = np.isclose(
        coords[:, None, :],             # shape (n, 1, 2)
        miss_coords[None, :, :],        # shape (1, m, 2)
        atol=atol,                      # absolute error
        rtol=rtol                       # relative error
    )                                   # shape (n, m, 2)
    
    # a match exists if both x and y are close
    mask = np.all(mask, axis=-1)    # shape (n, m)
    matched = np.any(mask, axis=0)  # shape (m, )
    
    # return the indexs of miss_coords in the coods
    idxs = np.full(len(miss_coords), -1)
    idxs[matched] = np.argmax(mask[:, matched], axis=0)

    return idxs


# DoH算子效果
if False:
    img = np.array([
        [6, 10, 0, 9, 7, 19, 7],
        [0, 6, 3, 36, 0, 0, 0],
        [1, 8, 91, 141, 81, 2, 0],
        [6, 34, 158, 255, 156, 27, 13],
        [11, 16, 94, 147, 95, 12, 5],
        [0, 0, 6, 38, 0, 3, 0],
        [7, 0, 12, 6, 7, 8, 0]
    ])

    x, y = np.arange(1, 6), np.arange(1, 6)
    xx, yy = np.meshgrid(x, y)
    print(img)
    print(cal_doh(img, xx, yy, 1))


    img = np.array([
        [1, 22, 11, 0, 7, 8, 14],
        [25, 31, 134, 225, 130, 28, 5],
        [1, 133, 254, 252, 237, 124, 3],
        [5, 216, 247, 255, 255, 213, 1],
        [7, 129, 248, 252, 255, 134, 7],
        [5, 30, 147, 220, 142, 22, 16],
        [17, 6, 5, 23, 12, 4, 2]
    ])
    x, y = np.arange(1, 6), np.arange(1, 6)
    xx, yy = np.meshgrid(x, y)
    print(img)
    print(cal_doh(img, xx, yy, 1))

    img = cv2.GaussianBlur(img, (5, 5), 1.5)
    x, y = np.arange(1, 6), np.arange(1, 6)
    xx, yy = np.meshgrid(x, y)
    print(img)
    print(cal_doh(img, xx, yy, 1))

    img = np.array([
        [7, 0, 7, 10, 2, 0, 0],
        [3, 7, 0, 15, 11, 5, 16],
        [1, 0, 10, 5, 8, 9, 16],
        [23, 4, 12, 255, 1, 23, 23],
        [14, 14, 20, 13, 7, 0, 2],
        [0, 0, 7, 8, 0, 2, 6],
        [4, 6, 12, 0, 16, 3, 2]
    ])
    x, y = np.arange(1, 6), np.arange(1, 6)
    xx, yy = np.meshgrid(x, y)
    print(img)
    print(cal_doh(img, xx, yy, 1).astype(int))


# 选择一处恒星数量多、星等差异大的视场，从而说明检测算法针对不同星等的恒星均能有限检测
ra, de, roll=radians(25.0588), radians(-21.7205), radians(0)
limit_mag=5.9
fov=20
psf=1
roi=3
save=True


# 星点检测作图
if True:
    dir = 'res/chapter3/detect'

    img0, stars = create_star_image(
        ra, de, roll,
        h=h,
        w=w, 
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        background=np.nan,
        sigma_psf=psf,
        roi=roi
    )
    real_coords = stars[:, 1:3]

    for den_meth, seg_meth, pixel_num in [
        # CCL-Based
        ('None', ['LCM',  'Liebe3', 'CCL', 'None'], 3),
        # ('None', ['ILCM',  'Liebe3', 'CCL', 'None'], 3),
        # ('None', ['NLCM',  'Liebe3', 'CCL', 'None'], 3),
        # ('None', ['GCM',  'Liebe3', 'CCL', 'None'], 3),

        # RG-Based
        # ('None', ['None', 'Liebe3', 'RG',  'DOH'], 3),
        ('None', ['LCM',  'Liebe3', 'RG',  'DOH'], 3),
        # ('None', ['SDM',  'Liebe3', 'RG',  'DOH'], 3),
        # ('None', ['GCM',  'Liebe3', 'RG', 'DOH'], 3),
    ]:
        for (g, p, s, y, x, lum, roi) in [
            # Constant stellar background
            # (0.00, 0.000, 'Constant', 0, 0, 0, 0),
            (0.05, 0.000, 'Constant', 0, 0, 7, 0),
            (0.05, 0.000, 'Constant', 0, 0, 8, 0),
            (0.05, 0.000, 'Constant', 0, 0, 9, 0),

            # Gasussian stellar background
            # (0.00, 0.000, 'Gaussian', h//2, w//4, 5.5, 128), 
            # (0.00, 0.005, 'Gaussian', h//2, w//4, 5, 128), 
            # (0.05, 0.000, 'Gaussian', h//2, w//4, 5.5, 128),
            # (0.05, 0.005, 'Gaussian', h//2, w//4, 5, 128), 

            # Linear stellar background
            # (0.00, 0.000, 'Linear_X', 0, 0, 5.3, 128), 
        ]:
            if save:
                os.makedirs(f'{dir}/{g}_{p}_{s}_{x}_{y}_{lum}_{roi}', exist_ok=True)

            img1 = add_stellar_noise(img0, method=s, position=(y, x), background=lum, sigma=roi)
            img2 = add_gaussian_and_pepper_noise(img1, sigma_g=g, prob_p=p)
            esti_coords = np.array(get_star_centroids(img2, den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num))

            # coords1: correct match
            # coords2: miss match
            # coords3: false match
            _, coords1, coords2, coords3 = find_overlap_and_unique(real_coords, esti_coords, 4)       
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            img2 = label_image(img2, coords1, (0, 255, 0))
            img2 = label_image(img2, coords2, (255, 0, 0))
            img2 = label_image(img2, coords3, (0, 0, 255))

            miss_idxs = find_miss_idxs(real_coords, coords2)
            print(
                'Deviation of Gaussian Noise:', g,
                'Probability of Salt-Pepper Noise:', p,
                '\nSegmentation Method:', seg_meth,
                '\nNumber of Miss Stars:', len(coords2),
                '\nNumber of False Stars:', len(coords3)
                # '\nMiss:\n', 
                # stars[miss_idxs, :3].astype(int),
                # '\nFalse:\n', 
                # coords3.astype(int)
            )

            if save:
                seg_meth_full = '_'.join(seg_meth)
                cv2.imwrite(f'{dir}/{g}_{p}_{s}_{x}_{y}_{lum}_{roi}/{den_meth}_{seg_meth_full}.png', img2)

    if save:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        img0 = label_image(img0, real_coords)
        cv2.imwrite(f'res/chapter3/detect/clean.png', img0)


# 星点检测数量对比
if False:
    num_test = 2

    img0, stars = create_star_image(
        ra, de, roll,
        h=h,
        w=w, 
        fovx=fov, 
        fovy=fov, 
        sigma_psf=psf,
        limit_mag=limit_mag, 
        background=background,
        roi=roi
    )
    real_coords = stars[:, 1:3]
    mags = stars[:, -1]

    real_coord = real_coords[3]

    # 打印测试相关信息
    print(
        'Detect Test',
        '\n-----------------------------',
        '\nNumber of test:', num_test,
        '\nRA:', ra, 'DE:', de,
        '\nMag info:', np.sort(mags), #np.histogram(mags, range=(0, limit_mag), bins=int(limit_mag)),
        '\nBackgroud intensity:', get_stellar_intensity(background),
        '\n-----------------------------',
    )

    for den_meth, thr_meth, seg_meth, pixel_num in [
        ('GAUSSIAN', 'Otsu', 'DCCL', 12),
        ('NONE', 'Liebe3', 'RG_LY', 5),
        ('NONE', 'Liebe3', 'RG_SOBEL', 5),
        ('CNB', 'Liebe3', 'RG_DOH', 3),
    ]:
        
        avg_cnts = []
        avg_err = []
        for (g, p) in [
            # (0.01, 0.001), 
            # (0.02, 0.002), 
            # (0.03, 0.003), 
            # (0.04, 0.004), 
            # (0.05, 0.005),
            # (0.06, 0.006), 
            # (0.07, 0.007), 
            # (0.08, 0.008),
            # (0.09, 0.009),

            # (0.02, 0.002), 
            # (0.05, 0.005),
            # (0.08, 0.008)
        ]:
            
            cnts = []
            err = []
            for _ in range(num_test):
                img1 = add_gaussian_and_pepper_noise(img0, sigma_g=g, prob_p=p)

                esti_coords = np.array(get_star_centroids(
                    img1, 
                    den_meth=den_meth, 
                    thr_meth=thr_meth, 
                    seg_meth=seg_meth, 
                    cen_meth='CoG',
                    pixel_limit=pixel_num,
                    connectivity=8
                ))

                # coords1: correct match
                # coords2: miss match
                # coords3: false match
                _, coords1, coords2, coords3 = find_overlap_and_unique(real_coords, esti_coords, eps=2)

                cnts.append((len(coords1), len(coords2), len(coords3)))
            
                # find the closest esti_coord
                if len(coords1) == 0:
                    err.append(np.inf)
                else:
                    dis = np.linalg.norm(coords1 - real_coord, axis=1)
                    err.append(np.min(dis))

            avg_cnts.append(np.mean(cnts, axis=0))
            avg_err.append(np.mean(err))

        avg_cnts, avg_err = np.array(avg_cnts), np.array(avg_err)
        print(
            'Method:', den_meth, seg_meth, thr_meth, pixel_num,
            # '\nTotal number of stars in test image:', len(real_coords),
            # '\nDetect result:', avg_cnts,
            '\nAverage correct rate:', avg_cnts[:, 0] / len(real_coords) * 100.0,
            '\nAverage miss count:', avg_cnts[:, 1],
            '\nAverage false count:', avg_cnts[:, 2],
            f'\nAverage centroid error for {real_coord[0], real_coord[1]}:', avg_err,
            '\n---------------------------------'
        )


# 星点检测耗时测试
if False:
    # random ra & de test
    num_test = 5
    
    # generate random right ascension[0, 360] and declination[-90, 90]
    ras = np.random.uniform(0, 2*np.pi, num_test)
    des = np.arcsin(np.random.uniform(-1, 1, num_test))

    # time test result
    res = {
        'RG_DOH': [],
        'CCL': [],
        'DCCL': [],
        'RLC': [],
        'CPL': []
    }

    # generate the star image
    for i in range(num_test):
        img1, _ = create_star_image(
            ra, de, roll, 
            fovx=fov, 
            fovy=fov, 
            h=h,
            w=w,
            limit_mag=limit_mag, 
            sigma_psf=psf,
            background=background,
            sigma_g=0.05, # default noise is important to time
            prob_p=0.001,
        )

        # denoise
        img2 = denoise_image(img1, 'NLM')

        # threshold
        T = cal_threshold(img2, 'Liebe3')

        for method in res:
            res[method].append(timeit.timeit(lambda: group_star(img2, method, T, connectivity=4, pixel_limit=5), number=3))
            # res[method].append(timeit.timeit(lambda: get_star_centroids(img1, 'MEDIAN', 'Liebe3', method, 'CoG', pixel_limit=5), number=3))
        
    for method in res:
        print(
            'Method:', method, 
            'Mean:', round(np.mean(res[method]), 4), 
            'Min', round(np.min(res[method]), 4), 
            'Max', round(np.max(res[method]), 4)
        )
