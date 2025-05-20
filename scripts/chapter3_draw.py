import os
import cv2
import numpy as np
import timeit
from math import radians

from simulate import create_star_image, add_gaussian_and_pepper_noise
from denoise import denoise_image
from detect import group_star, cal_threshold
from extract import get_star_centroids
from utils import find_overlap_and_unique, cal_mse_psnr_ssim


# 非局部均匀滤波测试——Lena图片
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


# 第三章末尾测试参数
ra, de, roll = radians(29.2104), radians(-12.0386), radians(0) # 可能每个测试拍摄视角不同
h, w = 512, 512
fov = 12
limit_mag = 6
background = 9
psf = 1


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
                method,
                '\nSigma of gaussian noise:', g, 
                '\nProbability of pepper noise', p, 
                '\nMSE:', mse, 
                '\nPSNR:', psnr, 
                '\nSSIM:', ssim,
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
        'NLM_BLF', 
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


# 选择一处恒星数量多、星等差异大的视场，从而说明检测算法针对不同星等的恒星均能有限检测
ra, de, roll = radians(55.0588), radians(49.7205), radians(0)
background = 7.5


# 星点检测作图
if False:
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

    for (g, p) in [(0.01, 0.001), (0.03, 0.003), (0.05, 0.005), (0.07, 0.007), (0.1, 0.01)]:
        img1 = add_gaussian_and_pepper_noise(img0, sigma_g=g, prob_p=p)
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

    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img0 = label_image(img0, real_coords)
    cv2.imwrite(f'res/chapter3/detect/clean.png', img0)


# 星点检测数量对比
if True:
    num_test = 10

    img0, stars = create_star_image(
        ra, de, roll,
        h=h,
        w=w, 
        fovx=fov, 
        fovy=fov, 
        sigma_psf=psf,
        limit_mag=limit_mag, 
        background=background
    )
    real_coords = stars[:, 1:3]
    mags = stars[:, -1]

    # 打印测试相关信息
    print(
        'Detect Test',
        '\n-----------------------------',
        '\nNumber of test:', num_test,
        '\nRA:', ra, 'DE:', de,
        '\nMag info:', np.sort(mags), #np.histogram(mags, range=(0, limit_mag), bins=int(limit_mag)),
        '\n-----------------------------',
    )

    for den_meth, seg_meth in [('MEDIAN', 'RG'), ('MEDIAN', 'DCCL')]:
        for (g, p) in [
            # (0.01, 0.001), 
            # (0.03, 0.003), 
            # (0.05, 0.005), 
            # (0.07, 0.007), 
            (0.1, 0.01)
        ]:
            cnts = []

            for _ in range(num_test):
                img1 = add_gaussian_and_pepper_noise(img0, sigma_g=g, prob_p=p)

                T = cal_threshold(img1, 'Liebe3')
                esti_coords = np.array(get_star_centroids(
                    img1, 
                    den_meth=den_meth, 
                    thr_meth='Liebe3', 
                    seg_meth=seg_meth, 
                    cen_meth='CoG',
                    pixel_limit=3,
                    T2=T,
                ))

                # coords1: correct match
                # coords2: miss match
                # coords3: false match
                _, coords1, coords2, coords3 = find_overlap_and_unique(real_coords, esti_coords, eps=1)

                cnts.append((len(coords1), len(coords2), len(coords3)))
            
            agv_cnts = np.mean(cnts, axis=0)
            print(
                'Denoise and segmentation method:', den_meth, seg_meth,
                '\nGuassian noise:', g, p,
                '\nTotal number of stars in test image:', len(real_coords),
                '\nDetect result:', cnts,
                '\nAverage correct count:', agv_cnts[0],
                '\nAverage miss count:', agv_cnts[1],
                '\nAverage false count:', agv_cnts[2],
                '\n---------------------------------'
            )


# 星点检测耗时测试
if False:
    # random ra & de test
    num_test = 1000
    
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
