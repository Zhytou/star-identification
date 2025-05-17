import os
import cv2
import numpy as np
from astropy.io import fits
import timeit
import cProfile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from math import radians

from simulate import create_star_image
from denoise import denoise_image
from detect import group_star, cal_threshold
from extract import get_star_centroids
from utils import find_overlap_and_unique, cal_mse_psnr_ssim


# 非局部均匀吕布 Lena图片测试
if False:
    img = cv2.imread(f'example/lena/lena.png', cv2.IMREAD_GRAYSCALE)

    imgs = {}
    imgs['noised'] = add_gaussian_and_pepper_noise(img, 0.00, 0.001)
    imgs['nlm'] = denoise_with_nlm(imgs['noised'])
    imgs['gaussian'] = filter_image(imgs['noised'], 'gaussian')
    imgs['median'] = filter_image(imgs['noised'], 'median')
    imgs['glp'] = filter_image(imgs['noised'], 'gaussian low pass')

    for key in imgs:
        cv2.imwrite(f'res/chapter3/lena/{key}.png', imgs[key])
        mse, psnr, ssim = cal_mse_psnr_ssim(img, imgs[key])
        print(key, mse, psnr, ssim)


ra, de, roll = radians(29.2104), radians(-12.0386), radians(0)
d = 64
h, w = 512, 512
x, y = 188, 169
fov = 12
limit_mag = 6
background = 9
psf = 1

# 改进双边滤波测试
if False:
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
    img1, _ = create_star_image(
        ra, de, roll, 
        w=w, 
        h=h,
        fovx=fov, 
        fovy=fov,
        sigma_g=0.05,
        prob_p=0.001,
        limit_mag=limit_mag, 
        sigma_psf=psf,
        background=background
    )
    
    img2 = denoise_with_nlm(img1)
    img3 = cv2.bilateralFilter(img2, 9, 30, 4)
    img4 = denoise_with_blf_new(img2, 3)

    cv2.imwrite('res/chapter3/nlm/clean.png', img0)
    cv2.imwrite('res/chapter3/nlm/noised.png', img1)
    cv2.imwrite('res/chapter3/nlm/nlm.png', img2)
    cv2.imwrite('res/chapter3/nlm/nlm_cvblf.png', img3)
    cv2.imwrite('res/chapter3/nlm/nlm_sdblf.png', img4)

    cv2.imwrite('res/chapter3/nlm/clean_scale.png', img0[y-d:y+d, x-d:x+d])
    cv2.imwrite('res/chapter3/nlm/noised_scale.png', img1[y-d:y+d, x-d:x+d])
    cv2.imwrite('res/chapter3/nlm/nlm_scale.png', img2[y-d:y+d, x-d:x+d])
    cv2.imwrite('res/chapter3/nlm/nlm_cvblf_scale.png', img3[y-d:y+d, x-d:x+d])
    cv2.imwrite('res/chapter3/nlm/nlm_sdblf_scale.png', img4[y-d:y+d, x-d:x+d])

    print('nlm', cal_mse_psnr_ssim(img0, img2))
    print('nlm+cvblf', cal_mse_psnr_ssim(img0, img3))
    print('nlm+sdblf', cal_mse_psnr_ssim(img0, img4))


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
        background=background
    )

    dir = f'res/chapter3/denoise'
    for (g, p) in [(0.05, 0.005)]:
        os.makedirs(f'{dir}/{g}_{p}', exist_ok=True)
        
        img1, _ =  create_star_image(
            ra, de, roll, 
            h=h,
            w=w,
            fovx=fov, 
            fovy=fov, 
            sigma_g=g,
            prob_p=p,
            limit_mag=limit_mag, 
            background=background
        )
        cv2.imwrite(f'res/chapter3/denoise/{g}_{p}/ORIGINAL.png', img1)

        for method in ['NLM_BLF', 'GAUSSIAN', 'MEAN', 'MEDIAN', 'BLF']:
            img2 = denoise_image(img1, method)
            cv2.imwrite(f'res/chapter3/denoise/{g}_{p}/{method}.png', img2)
            
            # 计算降噪前后图像质量指标
            mse, psnr, ssim = cal_mse_psnr_ssim(img0, img2)
            print(
                # 'Sigma of gaussian noise:', g, 
                # 'Probability of pepper noise', p, 
                'Method:', method, 
                'MSE:', mse, 
                'PSNR:', psnr, 
                'SSIM:', ssim,
                '\n--------------------------------'
            )


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


ra, de, roll = radians(229.2104), radians(-34.0386), radians(0)


# 图像降噪对质心计算的影响
if False:
    img0, stars = create_star_image(
        ra, de, roll, 
        w=w, 
        h=h,
        fovx=fov, 
        fovy=fov, 
        sigma_g=0.1,
        prob_p=0.01,
        limit_mag=limit_mag, 
        sigma_psf=psf,
        background=background
    )
    real_coords = stars[:, 1:3]

    for method in ['NONE', 'NLM_BLF', 'GAUSSIAN', 'MEAN', 'MEDIAN', 'BLF']:
        # 比较各个降噪方法后误差大小
        # 其中NONE为无预处理时
        esti_coords = np.array(get_star_centroids(
            img0,
            method,
            'Liebe3',
            'CCL',
            'MCoG',
            pixel_limit=5,
            num_esti=3 
        ))

        # correct count and error
        cnt, err = cal_centroid_error(real_coords, esti_coords)

        print(
            'Method:', method,  
            '\nNumber of correct extracted stars:', cnt,
            '\nTotal number of stars:', len(stars),
            '\nError:', err,
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


# 星点检测测试
if True:
    dir = 'res/chapter3/detect'
    os.makedirs('res/chapter3/detect', exist_ok=True)
    
    img0, stars = create_star_image(
        ra, de, roll,
        h=h,
        w=w, 
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        background=background
    )
    real_coords = stars[:, 1:3]

    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img0 = label_image(img0, real_coords)
    cv2.imwrite(f'res/chapter3/detect/clean.png', img0)

    for (g, p) in [(0.01, 0.001), (0.03, 0.003), (0.05, 0.005), (0.07, 0.007), (0.1, 0.01)]:
        img1, _ =create_star_image(
            ra, de, roll, 
            h=h,
            w=w,
            fovx=fov, 
            fovy=fov, 
            sigma_g=g,
            prob_p=p,
            limit_mag=limit_mag, 
            background=background
        )
        esti_coords = np.array(get_star_centroids(
            img1, 
            'NLM_BLF', 
            'Liebe5', 
            'RG', 
            'MCoG',
            pixel_limit=3
        ))

        # coords1: correct match
        # coords2: miss match
        # coords3: false match
        _, coords1, coords2, coords3 = find_overlap_and_unique(real_coords, esti_coords)

        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img1 = label_image(img1, coords1, (0, 255, 0))
        img1 = label_image(img1, coords2, (255, 0, 0))
        img1 = label_image(img1, coords3, (0, 0, 255))

        cv2.imwrite(f'res/chapter3/detect/{g}_{p}.png', img1)


# 星点检测耗时测试
if False:
    # random ra & de test
    num_test = 50
    
    # generate random right ascension[0, 360] and declination[-90, 90]
    ras = np.random.uniform(0, 2*np.pi, num_test)
    des = np.arcsin(np.random.uniform(-1, 1, num_test))

    # time test result
    res = {
        'RG': [],
        'CCL': [],
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
        img2 = denoise_image(img1, 'NLM_BLF')

        # threshold
        T = cal_threshold(img2, 'Liebe3')

        for method in res:
            res[method].append(timeit.timeit(lambda: group_star(img2, method, T0=T, connectivity=4, pixel_limit=5), number=3))
            # res[method].append(timeit.timeit(lambda: get_star_centroids(img1, 'MEDIAN', 'Liebe3', method, 'CoG', pixel_limit=5), number=3))
        
    for method in res:
        print(
            'Method:', method, 
            'Mean:', round(np.mean(res[method]), 4), 
            'Min', round(np.min(res[method]), 4), 
            'Max', round(np.max(res[method]), 4)
        )
