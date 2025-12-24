import os
import cv2
import numpy as np
import timeit
from math import radians

from simulate import create_star_image, add_gaussian_and_pepper_noise, add_stellar_noise
from denoise import denoise_image
from detect import group_star, cal_gcm
from extract import get_star_centroids
from utils import find_overlap_and_unique, cal_mse_psnr_ssim, cal_doh, draw_gray_3d


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
        # (0.06, 0.006), 
        # (0.09, 0.009), 
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
            # 'Oringinal',
            'Noised', 
            'Bilateral', 
            'NLM', 'AMF', 
            'Wavelet',
            # 'NLM_BLF',
            'EMF',
            'CWM', 
            'CMG',
            'CNB', 
        ]:
            if method == 'Oringinal':
                img2 = img0
            elif method == 'Noised':
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


def label_detect_result(img: np.ndarray, real_coords: np.ndarray, esti_coords: np.ndarray, legend_text: np.ndarray=None):
    '''
        Label the detect result with different shapes and colors.
    '''

    def draw_shape(img: np.ndarray, center: tuple[float, float], shape: str, color: tuple[int, int, int]=(0, 255, 0), size: int=4, thickness: int=1):
        y, x = int(center[0]), int(center[1])

        if shape == 'triangle' or shape == 'Triangle':
            pts = np.array([
                [x, y - size],
                [x - size, y + size],
                [x + size, y + size]
            ], dtype=np.int32)
            # cv2.fillPoly(img, [pts], color)
            if thickness >= 0:
                cv2.polylines(img, [pts], True, color, max(1, thickness))
        elif shape == 'cross' or shape == 'Cross':
            cv2.line(img, (x - size, y), (x + size, y), color, thickness)
            cv2.line(img, (x, y - size), (x, y + size), color, thickness)
        elif shape == 'rectangle' or shape == 'Rectangle':
            cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, thickness)
        else: # shape == 'circle' or shape == 'Circle'
            cv2.circle(img, (x, y), size, color, thickness)
            # cv2.circle(img, (x, y), size//2, (0, 0, 0), 1)

    def draw_legend(img: np.ndarray, entries: dict, top_left: tuple[int, int], legend_size: tuple[int, int]=(60, 100), gap: int=15):
        h, w = img.shape[:2]
        legend_h, legend_w = legend_size

        start_y, start_x = top_left
        start_y, start_x = max(legend_h + gap, min(h - gap - legend_h, start_y)), max(0, min(w - gap - legend_w, start_x))

        cv2.rectangle(img, (start_x, start_y), (start_x + legend_w, start_y + legend_h), (255, 255, 255), thickness=-1)

        current_y = start_y
        for label in entries:
            shape, color = entries[label]

            margin_y, margin_x = (legend_h - (len(entries) - 1 ) * gap) // 2, legend_w // 8
            legend_shape_y, legend_shape_x = current_y + margin_y, start_x + margin_x
            legend_shape_size = 8
            draw_shape(img, (legend_shape_y, legend_shape_x), shape, color=color, size=legend_shape_size)  #! careful, draw_shape use (row, column), while cv2 rawing functions expect (column, row)
            if legend_text is None:
                (_, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                total_text_height = text_height + baseline
                text_baseline_y = current_y + gap // 2 + (total_text_height / 2 - baseline)
                cv2.putText(img, label, (start_x + 2 * gap, int(text_baseline_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                paste_y, paste_x = start_y + margin_y, int(legend_shape_x + legend_shape_size * 1.2)
                text = cv2.resize(legend_text, (legend_w - 2 * margin_x, legend_h - 2 * margin_y))
                text_h, text_w = text.shape[:2]

                img[paste_y:paste_y+text_h, paste_x:paste_x+text_w] = text
            current_y += gap

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _, matched_coords, missed_coords, false_coords = find_overlap_and_unique(real_coords, esti_coords, 1)       
    entries = {
        'Correct Star': ('Cross', (0, 0, 255)),
        'Matched Detection': ('Triangle', (0, 255, 255)),
        # 'Missed Star': ('', (0, 0, 255)),
        'False Alarm': ('Circle', (255, 0, 0))
    }

    for coord in matched_coords:
        shape, color = entries['Correct Star']
        draw_shape(img, coord, shape, color)
    for coord in matched_coords:
        shape, color = entries['Matched Detection']
        draw_shape(img, coord, shape, color)
    # for coord in matched_coords:
    #     shape, color = entries['Missed Star']
    #     draw_shape(img, coord, shape, color)
    # for coord in matched_coords:
    #     shape, color = entries['False Alarm']
    #     draw_shape(img, coord, shape, color)

    draw_legend(img, entries, (h-15, w-40), (50, 120))

    # miss_idxs = find_miss_idxs(real_coords, missed_coords)
    print(
        'Total Number of Stars:', len(real_coords),
        '\nNumber of Matched Stars:', len(matched_coords),
        '\nNumber of Miss Stars:', len(missed_coords),
        '\nNumber of False Stars:', len(false_coords),
        # '\nMiss:\n', 
        # real_coords[miss_idxs].astype(int),
        # '\nFalse:\n', 
        # coords3.astype(int)
    )

    return img


# 选择一处恒星数量多、星等差异大的视场，从而说明检测算法针对不同星等的恒星均能有限检测
ra, de, roll=radians(25.0588), radians(-21.7205), radians(0)
limit_mag=6
fov=12
psf=3
roi=4
background=7
save=True


# 含噪声图的3D分布
if False:
    img, stars = create_star_image(
        ra, de, roll,
        h=h,
        w=w, 
        fovx=fov, 
        fovy=fov, 
        limit_mag=limit_mag, 
        background=background,
        sigma_psf=psf,
        roi=roi,
    )

    img = add_stellar_noise(img, method='Gaussian', position=(h//2, w//4), luminosity=4.5, sigma_x=64, sigma_y=96)
    # img = add_gaussian_and_pepper_noise(img, sigma_g=0.002, prob_p=0.0)

    draw_gray_3d(img)


# DoH算子 及 改进后的DoH算子运算结果
if False:
    imgs = np.array([
        # star 5 mv
        [[6, 10, 0, 9, 7, 19, 7],
        [0, 6, 3, 36, 0, 0, 0],
        [1, 8, 91, 141, 81, 2, 0],
        [6, 34, 158, 255, 156, 27, 13],
        [11, 16, 94, 147, 95, 12, 5],
        [0, 0, 6, 38, 0, 3, 0],
        [7, 0, 12, 6, 7, 8, 0]],
        # star 3 mv
        [[1, 22, 11, 0, 7, 8, 14],
        [25, 31, 134, 225, 130, 28, 5],
        [1, 133, 254, 252, 237, 124, 3],
        [5, 216, 247, 255, 255, 213, 1],
        [7, 129, 248, 252, 255, 134, 7],
        [5, 30, 147, 220, 142, 22, 16],
        [17, 6, 5, 23, 12, 4, 2]],
        # pepper-salt noise
        [[7, 0, 7, 10, 2, 0, 0],
        [3, 7, 0, 15, 11, 5, 16],
        [1, 0, 10, 5, 8, 9, 16],
        [23, 4, 12, 255, 1, 23, 23],
        [14, 14, 20, 13, 7, 0, 2],
        [0, 0, 7, 8, 0, 2, 6],
        [4, 6, 12, 0, 16, 3, 2]]
    ])

    K = 7       # size
    R = K // 2  # half of size
    P = 5   # pad
    for i, img in enumerate(imgs):
        img = np.pad(img, ((P, P), (P, P)))
        # draw_gray_3d(img, 'bar3d')
        os.makedirs('res/chapter3/doh/', exist_ok=True)
        cv2.imwrite(f'res/chapter3/doh/original_{i}.png', img)

        doh = cal_doh(img, sigma=1.5)
        doh = np.clip(doh, 0, 255)
        cv2.imwrite(f'res/chapter3/doh/doh_{i}.png', doh)

        gcm = cal_gcm(img, 'PGCM')
        mdoh = doh
        mdoh[P+R, P+R] = doh[P+R, P+R] * gcm[P+R, P+R]
        if i == 0:
            print(np.round(doh[P:-P, P:-P], 2))
            print(np.round(mdoh[P:-P, P:-P], 2))
            print(np.round(gcm[P:-P, P:-P], 2))
        

fov=20
psf=1
roi=2


# 星点检测作图
if False:
    dir = 'res/chapter3/detect'

    img0, stars = create_star_image(
        ra, de, roll, h=h, w=w, 
        fovx=fov, fovy=fov, 
        limit_mag=limit_mag, 
        background=6.5,
        sigma_psf=psf,
        roi=roi
    )
    real_coords = stars[:, 1:3]

    for (g, p, s, y, x, lum, roi) in [
        ## Constant stellar background
        (0.00, 0.000, 'Constant', 0, 0, 7.5, 0),

        # Gasussian stellar background
        (0.00, 0.000, 'Gaussian', h//2, -w//4, 5.5, 128), 
        (0.00, 0.000, 'Gaussian', h//3, w//5*3, 5, 64), 

        ## Linear stellar background
        # (0.00, 0.000, 'Linear_X', 0, 0, 5.3, 128), 
        (0.00, 0.000, 'Linear_Y', 0, 0, 5.3, 128), 
    ]:
        img1 = add_stellar_noise(img0, method=s, position=(y, x), luminosity=lum, sigma_x=roi)
        img2 = add_gaussian_and_pepper_noise(img1, sigma_g=g, prob_p=p)
    
        os.makedirs(f'{dir}/{g}_{p}_{s}_{x}_{y}_{lum}_{roi}', exist_ok=True)
        cv2.imwrite(f'{dir}/{g}_{p}_{s}_{x}_{y}_{lum}_{roi}/Clean.png', img2)
        for den_meth, seg_meth, pixel_num in [
            # CCL-Based
            # ('None', ['None',  'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['LCM',  'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['ILCM',  'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['NLCM',  'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['RLCM',  'Liebe3', 'CCL', 'None'], 1),
            # ('None', ['Top-Hat', 'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['Max-Median', 'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['IGM', 'Liebe3', 'CCL', 'None'], 3),

            # RG-Based
            # ('None', ['None', 'Liebe3', 'RG',  'None'], 3),
            ('None', ['None', 'Liebe3', 'RG',  'CGC'], 3),
        ]:
            print(
                'Deviation of Gaussian Noise:', g,
                '\tProbability of Salt-Pepper Noise:', p,
                '\tLuminosity of Background:', lum,
                '\nSegmentation Method:', seg_meth,
            )

            esti_coords = np.array(get_star_centroids(img2.copy(), den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num))

            if os.path.exists('res/chapter3/detect/legend_text.png'):
                legend_text = cv2.imread('res/chapter3/detect/legend_text.png', cv2.IMREAD_COLOR)
            else:
                legend_text = None
            img3 = label_detect_result(img2, real_coords, esti_coords, legend_text)

            seg_meth_full = '_'.join(seg_meth)
            cv2.imwrite(f'{dir}/{g}_{p}_{s}_{x}_{y}_{lum}_{roi}/{seg_meth_full}.png', img3)


# 实拍红外小目标图像检测效果
if False:
    # dataset_name = 'sirst' 
    dataset_name = 'irstd-1k'
    os.makedirs(f'res/chapter3/realshot/{dataset_name}', exist_ok=True)

    for name in os.listdir(f'realshot/{dataset_name}/images'):
        img1 = cv2.imread(f'realshot/{dataset_name}/images/{name}', cv2.IMREAD_GRAYSCALE)
        if name == 'Misc_10.png':
            break
        for den_meth, seg_meth, pixel_num in [
            # ('None', ['LCM',  'Liebe3', 'CCL', 'None'], 3),
            ('None', ['Top-Hat', 'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['Max-Median', 'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['IGM', 'Liebe3', 'CCL', 'None'], 3),
        ]:
            esti_coords = np.array(get_star_centroids(img1.copy(), den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num))

            img2 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2 = label_image(img2, esti_coords, (0, 255, 0))
        
            seg_meth_full = '_'.join(seg_meth)
            os.makedirs(f'res/chapter3/realshot/{dataset_name}/{seg_meth_full}', exist_ok=True)
            cv2.imwrite(f'res/chapter3/realshot/{dataset_name}/{seg_meth_full}/{name}', img2)


# 实拍星图检测实验（该星图由实拍杂光背景和仿真星点合成得到）
if False:
    dataset_name = 'stellar-bg'
    for name in os.listdir(f'realshot/{dataset_name}'):
        img0 = cv2.imread(f'realshot/{dataset_name}/{name}', cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, (w, h)) # 注意resize形状先列后行

        img1, stars = create_star_image(
            ra, de, roll,
            h=h,
            w=w, 
            fovx=fov, 
            fovy=fov, 
            limit_mag=limit_mag, 
            background=img0,
            sigma_psf=psf,
            roi=roi,
        )
        real_coords = stars[:, 1:3]

        img2 = add_gaussian_and_pepper_noise(img1, sigma_g=0, prob_p=0)

        for den_meth, seg_meth, pixel_num in [
            ('None', ['IGM', 'Liebe3', 'CCL', 'None'], 3),
            ('None', ['MLCM', 'Liebe3', 'RG',  'CGC'], 3),
        ]:
            esti_coords = np.array(get_star_centroids(img2.copy(), den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num))

            # coords1: correct match
            # coords2: miss match
            # coords3: false match
            _, coords1, coords2, coords3 = find_overlap_and_unique(real_coords, esti_coords, 4)       
            img3 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            img3 = label_image(img3, coords1, (0, 255, 0))
            img3 = label_image(img3, coords2, (255, 0, 0))
            img3 = label_image(img3, coords3, (0, 0, 255))

            print(
                'Segmentation Method:', seg_meth,
                '\nTotal Number of Stars:', len(real_coords),
                '\nNumber of Miss Stars:', len(coords2),
                '\nNumber of False Stars:', len(coords3),
            )

            seg_meth_full = '_'.join(seg_meth)
            os.makedirs(f'res/chapter3/realshot/{dataset_name}/{seg_meth_full}', exist_ok=True)
            cv2.imwrite(f'res/chapter3/realshot/{dataset_name}/{seg_meth_full}/{name}', img3)
            

# 星点检测数量对比
if True:
    num_test = 20

    img0, stars = create_star_image(
        ra, de, roll,
        h=h,
        w=w, 
        fovx=fov, 
        fovy=fov, 
        sigma_psf=psf,
        limit_mag=limit_mag, 
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
        '\n-----------------------------',
    )

    for den_meth, seg_meth, pixel_num in [
        # ('None', ['None',  'Otsu', 'CCL', 'None'], 3),
        # ('None', ['Top-Hat',  'Liebe3', 'CCL', 'None'], 3),
        # ('Median', ['MSobel', 'Liebe5', 'CCL', 'None'], 3),
        ('Median', ['MS_PCM',  'Liebe3', 'CCL', 'None'], 3),
        # ('None', ['None',  'Liebe3', 'RG',  'Ly'], 3),
        # ('None', ['None',  'Liebe3', 'RG',  'CGC'], 3),
    ]:
        avg_cnts = []
        avg_err = []
        for (g, p, s, y, x, lum, roi) in [
            # Constant stellar background
            # (0.03, 0.003, 'Constant', 0, 0, 7, 0),
            # (0.06, 0.006, 'Constant', 0, 0, 8, 0),
            # (0.09, 0.009, 'Constant', 0, 0, 7, 0),

            # Gasussian stellar background
            # (0.00, 0.000, 'Gaussian', h//2, w//4, 5.5, 128), 
            (0.03, 0.003, 'Gaussian', h//2, w//4, 5, 128), 
            # (0.06, 0.006, 'Gaussian', h//2, w//4, 5, 128),
            # (0.09, 0.009, 'Gaussian', h//2, w//4, 5, 128)
        ]:
    
            cnts = []
            err = []
            for _ in range(num_test):
                img1 = add_stellar_noise(img0, method=s, position=(y, x), luminosity=lum, sigma_x=roi)
                img2 = add_gaussian_and_pepper_noise(img1, sigma_g=g, prob_p=p)
                esti_coords = np.array(get_star_centroids(img2, den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num))

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
            'Method:', den_meth, seg_meth, pixel_num,
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

        for method in res:
            res[method].append(timeit.timeit(lambda: group_star(img2, method, connectivity=4, pixel_limit=5), number=3))
            # res[method].append(timeit.timeit(lambda: get_star_centroids(img1, 'MEDIAN', 'Liebe3', method, 'CoG', pixel_limit=5), number=3))
        
    for method in res:
        print(
            'Method:', method, 
            'Mean:', round(np.mean(res[method]), 4), 
            'Min', round(np.min(res[method]), 4), 
            'Max', round(np.max(res[method]), 4)
        )
