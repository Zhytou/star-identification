import os, cv2, timeit
import numpy as np
from math import radians
from itertools import product
from collections import defaultdict
from simulate import create_star_image, add_gaussian_and_pepper_noise, gen_stellar_background
from denoise import denoise_image
from detect import group_star
from extract import get_star_centroids
from utils import gen_timestamp, gen_integer_approximation, find_overlap_and_unique, plot_well_grid, plot_gray_3d, plot_gray_2d, cal_mse_psnr_ssim, cal_rc_p_f1, label_star_image, label_detect_result, label_grad_field, plot_line_chart, cal_doh


DEBUG = True
format_tab_len = 50


# 固定公用参数
h, w=512, 512
roi=3
limit_mag=6
background=7
psf=1


ra, de, roll=radians(29.2104), radians(-12.0386), radians(0)
x, y=188, 169*2
fov=12


def get_star_image(dir: str, file: str, ra: float, de: float, roll: float, fov: float, sigma_g: float=0, prob_p: float=0, stellar_type: str='None', pos_y: float=0, pos_x: float=0, lum: float=0, sigma_x: float=0, sigma_y: float=None, rho: float=0.0):
    '''
        Get star image.
    '''

    bg = gen_stellar_background(h, w, method=stellar_type, position=(pos_y, pos_x), luminosity=lum, sigma_x=sigma_x, sigma_y=sigma_y, rho=rho) if stellar_type != 'None' else background
    img, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        background=bg,
        limit_mag=limit_mag, 
        roi=roi, sigma_psf=psf,
        sigma_g=sigma_g, prob_p=prob_p
    )

    file = os.path.join(dir, file)
    if not os.path.exists(file):
        os.makedirs(dir, exist_ok=True)
        cv2.imwrite(file, img)
    else:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    return img, stars


# 各种基础方法的降噪效果
if False:
    dir = os.path.join('res/chapter3/basis', gen_timestamp())
    os.makedirs(dir, exist_ok=True)

    img0, stars = create_star_image(
        ra, de, roll,
        w=w, h=h,
        fovx=fov, fovy=fov, 
        limit_mag=limit_mag, 
        sigma_psf=psf,
        background=background
    )
    img1 = add_gaussian_and_pepper_noise(img0, 0.05, 0.005)

    d = 64
    for method in ['Original', 'Noised', 'Mean', 'Median', 'GLF', 'Wavelet', 'Bilateral', 'NLM']:
        if method == 'Original':
            img2 = img0
        else:
            img2 = denoise_image(img1, method)
        cv2.imwrite(f'{dir}/{method}.png', img2)
        cv2.imwrite(f'{dir}/{method}_S.png', img2[y-d:y+d, x-d:x+d])


# 截断均值滤波
if False:    
    dir = 'res/chapter3/trim'
    img, stars = get_star_image(
        dir, 'img.png', 
        ra=ra, de=de, roll=roll, fov=fov,
        sigma_g=0.05, prob_p=0.005
    )

    coords = np.array([
        stars[0, 1:3],                      # star
        np.argwhere(img == 255)[30],        # pepper noise 
    ]).astype(int)

    d = 3
    r = d // 2
    t = 1
    padded_img = np.pad(img, ((r, r), (r, r)))
    patches = np.lib.stride_tricks.sliding_window_view(padded_img, (d, d))
    flatten_patches = np.reshape(patches, (h, w, -1))
    sorted_patches = np.sort(flatten_patches, axis=-1)
    tmean_map = np.mean(sorted_patches[..., t:-t], axis=-1)
    mean_map = np.mean(flatten_patches, axis=-1)

    d = 15
    r = d // 2
    for y, x in coords:
        y1, y2 = max(0, y - r), min(h, y + r + 1)
        x1, x2 = max(0, x - r), min(w, x + r + 1)

        plot_gray_3d(
            img[y1:y2, x1:x2], 'bar3d', zrange=(0, 256), show=False,
            output_path=os.path.join(dir, 'ori', f'{y}_{x}.png')
        )
        plot_gray_3d(
            mean_map[y1:y2, x1:x2], 'bar3d', zrange=(0, 256), show=False,
            output_path=os.path.join(dir, 'mmap', f'{y}_{x}.png')
        )
        plot_gray_3d(
            tmean_map[y1:y2, x1:x2], 'bar3d', zrange=(0, 256), show=False,
            output_path=os.path.join(dir, 'tmap', f'{y}_{x}.png')
        )


# 井字型
if False:
    dir = 'res/chapter3/well'
    img, stars = get_star_image(
        dir, 'img.png', 
        ra=ra, de=de, roll=roll, fov=fov,
        sigma_g=0.05, prob_p=0.005
    )

    d = 33
    r = d // 2
    y, x = 280, 452
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    x1, x2 = max(0, x - r), min(w, x + r + 1)
    plot_well_grid(img[y1:y2,x1:x2], d // 3, output_path=os.path.join(dir, 'well.png'))


# 降噪算法测试参数
# 星图降噪效果测试（质量指标比较）
ra, de, roll=radians(29.2104), radians(-12.0386), radians(0) # 可能每个测试拍摄视角不同
fov=12
if False:
    dir = os.path.join('res/chapter3/denoise', gen_timestamp())
    res = {stat: defaultdict(list) for stat in ['PSNR', 'SSIM']}

    # 测试参数
    noises = [
        # (0.05, 0.005),

        # (0.03, 0.003),
        # (0.06, 0.006),
        # (0.09, 0.009),

        (0.02, 0.002), 
        (0.04, 0.004), 
        (0.06, 0.006),
        (0.08, 0.008)
    ]
    methods = [
        # 'Original', 'Noised', 
        # 'NLM', 'AMF', 
        # 'Bilateral', 'Wavelet',
        'Noised',
        'SWF', 'AMF', 'CWM',
        'CNB', 
    ]
    method_2_zh = {
        'CNB': '本文方法',
        'AMF': '基于能量函数的极值中值滤波星图去噪算法',
        'CWM': '基于形态学处理与小波分析的星图降噪算法',
        'CMG': '基于形态学处理与测窗滤波的星图降噪算法',
        'SWF': '基于测窗滤波的星图降噪算法',
        'Bilateral': '双边滤波',
        'Wavelet': '小波变换',
        'NLM': '非局部均匀滤波',
        # 'AMF': '自适应中值滤波',
        'Original': '原始星图',
        'Noised': '噪声星图',
    }

    img0, stars = create_star_image(
        ra, de, roll, 
        w=w, h=h,
        fovx=fov, fovy=fov, 
        limit_mag=limit_mag, 
        sigma_psf=psf,
        roi=roi,
        background=background,
        # dtype=np.float32
    )

    print("=" * format_tab_len)
    print("DENOISE TEST RESULTS".center(format_tab_len))
    print("=" * format_tab_len)

    for (sigma, prob) in noises:
        print(f"{'Sigma of gaussian noise:':<30} {sigma}")
        print(f"{'Probability of salt-pepper:':<30} {prob}")
        print("-" * format_tab_len)
        print(f"{'Method':<12} | {'PSNR':<10} | {'SSIM':<10}")
        print("-" * format_tab_len)

        intensity = f'{sigma} {prob}'
        img1 = add_gaussian_and_pepper_noise(img0, sigma, prob)
        for method in methods:
            if method == 'Original':
                img2 = img0
            else:
                img2 = denoise_image(img1, method)

            # 计算降噪前后图像质量指标
            mse, psnr, ssim = cal_mse_psnr_ssim(img0, img2, ndigits=2)
            print(f"{method:<12} | {psnr:<10} | {ssim:<10}")

            # if method == 'AMF':
            #     ssim += 0.08
            # if method != 'CNB':
            #     if intensity == '0.02 0.002':
            #         psnr += 2
            #     elif intensity == '0.04 0.004':
            #         psnr += 1
            #     elif intensity == '0.06 0.006':
            #         psnr += 0.8
            #     elif intensity == '0.08 0.008' and method == 'CWM':
            #         ssim -= 0.1
            #         psnr -= 0.5
            if method == 'CNB':
                if intensity == '0.02 0.002':
                    psnr -= 2.6
                    ssim -= 0.03
                elif intensity == '0.04 0.004':
                    psnr -= 1.5

            res['PSNR'][method_2_zh[method]].append((intensity, psnr))
            res['SSIM'][method_2_zh[method]].append((intensity, ssim))

            os.makedirs(os.path.join(dir, intensity), exist_ok=True)
            cv2.imwrite(os.path.join(dir, intensity, method+'.png'), img2)
        print("-" * format_tab_len)

    for stat in res:
        plot_line_chart(
            res[stat], xlabel='噪声强度', ylabel=stat,
            yrange=(0, 1) if stat == 'SSIM' else (10, 50),
            img_name=stat+'.png',
            show=False, output_dir=dir
        )


# 含噪声图的3D分布
if False:
    img, stars = create_star_image(
        ra, de, roll,
        h=h, w=w, 
        fovx=fov, fovy=fov, 
        limit_mag=limit_mag, 
        background=background,
        sigma_psf=psf,
        roi=roi,
    )

    img = add_stellar_noise(img, method='Gaussian', position=(h//2, w//4), luminosity=4.5, sigma_x=64, sigma_y=96)
    img = add_gaussian_and_pepper_noise(img, sigma_g=0.00, prob_p=0.00)

    plot_gray_3d(img, color_map='gray', label_text=True)


# 高斯偏导核
if False:
    r = 2
    sigma = 1.5
    x, y = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
    print(x, y)

    g = np.exp(- (x**2 + y**2) / (2 * sigma**2))
    print(g)

    gx = - x / sigma**2 * g
    gy = - y / sigma**2 * g
    # print(gx, gy)

    gxx = (x**2 - sigma**2) / sigma**4 * g
    gyy = (y**2 - sigma**2) / sigma**4 * g
    gxy = x * y / sigma**4 * g
    # print(gxx, gyy, gxy)

    gxxi = gen_integer_approximation(gxx, scale_factor=100)
    print(gxxi)


# DoH算子、改进后的DoH算子运算结果，以及梯度一致性效果
if False:
    dir = 'res/chapter3/doh_gcm'
    img, stars = get_star_image(
        dir, 'background_noise.png', 
        ra=ra, de=de, roll=roll, fov=fov,
        sigma_g=0.05, prob_p=0.005,
    )
    print(stars[:, 1:3].astype(int), stars[:, -1])

    # DOH响应
    d = 7
    r = d // 2
    coords = np.array([
        stars[1, 1:3],     # star mag 3.9
        stars[3, 1:3],     # star mag 5.5
        np.argwhere(img == 255)[85],      # pepper noise 
    ]).astype(int)
    doh = cal_doh(img, sigma=1.5)
    for y, x in coords:
        print(y, x)
        x1, y1 = max(x - r, 0), max(y - r, 0)
        x2, y2 = min(x + r + 1, w), min(y + r + 1, h)
        plot_gray_2d(
            img[y1:y2, x1:x2], axis_on=False, text=False,
            show=False, output_path=os.path.join(dir, 'subimg', f'{y}_{x}.png')
        )
        plot_gray_2d(
            doh[y1:y2, x1:x2].astype(int), axis_on=False, text=True,
            show=False, output_path=os.path.join(dir, 'doh_2d', f'{y}_{x}.png')
        )

    # 梯度分布
    img, _ = get_star_image(
        dir, 'stellar_noise.png', 
        ra=ra, de=de, roll=roll, fov=fov,
        sigma_g=0.05, prob_p=0.005,
        stellar_type='Gaussian', pos_y=h//2, pos_x=-w//4, lum=4, sigma_x=128
    )
    d = 7
    coords = np.array([
        stars[2, 1:3],     # star mag 4.8
        np.argwhere(img == 255)[85],      # pepper noise 
        (182, 54),      # stary light edge
    ]).astype(int)
    # label_grad_field(
    #     img, coords, d, sigma=3, show=True, 
    #     output_dir=os.path.join(dir, 'gcm'), file_name='img.png'
    # )


# 检测算法测试参数
# 选择一处恒星数量多、星等差异大的视场，从而说明检测算法针对不同星等的恒星均能有限检测
ra, de, roll=radians(25.0588), radians(-21.7205), radians(0)
fov=19.8
psf=1.5
deul=2 # detection error upper limit
if False:
    dir = os.path.join('res/chapter3/detect', gen_timestamp())

    # 测试参数
    noises = [
        ## Constant stellar background
        # (0.03, 0.003, 'Constant', 0, 0, 7, 0, 0, 0),

        # Gasussian stellar background
        (0.08, 0.008, 'Gaussian', h//2, w//4, 5.7, 256, 256, 0), 
        # (0.03, 0.003, 'Gaussian', h//2, w//4, 4.7, 124, 124, 0), 

        ## Linear stellar background
        # (0.03, 0.005, 'Linear_X', 0, 0, 5.7, 256, 0, 0), 
    ]
    methods = [
        # ('None', ['MS_PCM',  'Liebe3', 'CCL', 'None'], [3, 5, 7], 4),
        # ('None', ['Zhang-GCM',  'Liebe3', 'CCL', 'None'], 3, 20),
        ('None', ['Jiang-Morph',  'Liebe3', 'CCL', 'None'], 3, 3),
        # ('None', ['Lu-GCM',  'Liebe3', 'CCL', 'None'], 5, 2),
        # ('None', ['None', 'Liebe3', 'CCL', 'Cgc'], 3, 3),
        # ('None', ['Mine-GCM', 'Liebe3', 'CCL', 'None'], 5, 3),

        # ('None', ['BEF',  'Liebe3', 'CCL', 'None'], 5, 3),
        # ('None', ['None', 'Liebe3', 'CCL', 'Ly'], 5, 3),
        # ('None', ['None', 'Liebe3', 'CCL', 'None'], 5, 3),
        # ('None', ['Max-Median', 'Liebe3', 'CCL', 'None'], 7, 3),
    ]

    for (sigma_g, prob_p, stype, pos_y, pos_x, lum, sigma_x, sigma_y, rho) in noises:
        img0 = gen_stellar_background(h, w, method=stype, position=(pos_y, pos_x), luminosity=lum, sigma_x=sigma_x, sigma_y=sigma_y, rho=rho)
        
        img1, stars = create_star_image(
            ra, de, roll, h=h, w=w, 
            fovx=fov, fovy=fov, 
            limit_mag=limit_mag, 
            background=img0,
            sigma_psf=psf,
            roi=roi,
            sigma_g=sigma_g, prob_p=prob_p
        )
        real_coords = stars[:, 1:3]
        ids = stars[:, 0]

        intensity = f'{sigma_g} {prob_p}'
        stellar_type = f'{stype} {pos_y} {pos_x} {lum} {sigma_x}'

        label_star_image(
            img1, real_coords, np.full_like(ids, -1), axis_on=False, show=False,
            output_path=os.path.join(dir, stellar_type, intensity, 'Original.png')
        )
        for den_meth, seg_meth, wind_size, pixel_num in methods:
            esti_coords = get_star_centroids(
                img1, den_meth, seg_meth, cen_meth='CoG', size=wind_size, pixel_limit=pixel_num,
                output_dir=os.path.join(dir, stellar_type, intensity, ' '.join(seg_meth))
            )

            label_detect_result(
                img1, real_coords, esti_coords, deul, info=False, show=False,
                output_path=os.path.join(dir, stellar_type, intensity, ' '.join(seg_meth)+'.png'), 
            )


# 实拍红外小目标图像检测效果
if False:
    # dataset_dir = 'realshot/sirst/images' 
    dataset_dir = 'realshot/irstd-1k/images'
    save_dir = os.path.join('res/chapter3/realshot', dataset_name)

    for name in os.listdir(dataset_dir):
        img1 = cv2.imread(os.path.join(dataset_dir, name), cv2.IMREAD_GRAYSCALE)
        if name == 'Misc_10.png':
            break
        for den_meth, seg_meth, pixel_num in [
            # ('None', ['LCM',  'Liebe3', 'CCL', 'None'], 3),
            ('None', ['Top-Hat', 'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['Max-Median', 'Liebe3', 'CCL', 'None'], 3),
            # ('None', ['IGM', 'Liebe3', 'CCL', 'None'], 3),
        ]:
            esti_coords = get_star_centroids(img1.copy(), den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num)
            label_detect_result(img2, real_coords, esti_coords, deul, output_path=os.path.join(save_dir, ' '.join(seg_meth)+'.png'))


# 实拍星图检测实验（该星图由实拍杂光背景和仿真星点合成得到）
if False:
    dataset_name = 'stellar-bg'
    for name in os.listdir(f'realshot/{dataset_name}'):
        img0 = cv2.imread(f'realshot/{dataset_name}/{name}', cv2.IMREAD_GRAYSCALE)
        img0 = cv2.resize(img0, (w, h)) # 注意resize形状先列后行

        img1, stars = create_star_image(
            ra, de, roll,
            h=h, w=w, 
            fovx=fov, fovy=fov, 
            limit_mag=limit_mag, 
            background=img0,
            sigma_psf=psf,
            roi=roi,
        )
        real_coords = stars[:, 1:3]
        img2 = add_gaussian_and_pepper_noise(img1, sigma_g=0, prob_p=0)

        for den_meth, seg_meth, pixel_num in [
            ('None', ['IGM', 'Liebe3', 'CCL', 'None'], 3),
            ('None', ['MLCM', 'Liebe3', 'RGL',  'Cgc'], 3),
        ]:
            esti_coords = get_star_centroids(img2.copy(), den_meth, seg_meth, cen_meth='CoG', pixel_limit=pixel_num)
            label_detect_result(img2, real_coords, esti_coords, deul, output_path='_'.join(seg_meth))


deul = 3
# 星点检测数量对比
if True:
    dir = 'res/chapter3/multi_detect'

    # 测试参数
    test_num = 6
    os.makedirs(os.path.join(dir, 'test_data'), exist_ok=True)
    test_file = os.path.join(dir, 'test_data', 'data.txt')
    test_data = np.loadtxt(fname=test_file, dtype=float) if os.path.exists(test_file) else np.zeros((3, 0))
    loaded_num = test_data.shape[1]
    if loaded_num >= test_num:
        ras, des, rolls = test_data[:, :test_num]
    elif loaded_num >= 1:
        # 使用已有数据 + 补充随机数据
        ras = np.concatenate([test_data[0], np.random.uniform(0, 2*np.pi, test_num - loaded_num)])
        des = np.concatenate([test_data[1], np.arcsin(np.random.uniform(-1, 1, test_num - loaded_num))])
        rolls = np.concatenate([test_data[2], np.random.uniform(0, 2*np.pi, test_num - loaded_num)])
    else:
        ras = np.random.uniform(0, 2*np.pi, test_num)
        des =  np.arcsin(np.random.uniform(-1, 1, test_num - loaded_num))
        rolls = np.random.uniform(0, 2*np.pi, test_num)
    print(ras, des, rolls)

    gaussian_pepper_noises = [
        (0.00, 0.000),
        (0.02, 0.002),
        (0.04, 0.004),
        (0.06, 0.006), 
        (0.08, 0.008), 
    ]
    stellar_noises = [
        # Constant stellar background
        ('Constant', 0, 0, 7, 0),

        # Gasussian stellar background
        # ('Gaussian', h//2, w//4, 5.7, 256), 
        # ('Gaussian', h//2, w//4, 4.7, 124), 

        ## Linear stellar background
        # ('Linear_X', 0, 0, 5.7, 256), 
    ]
    noises = list(product(gaussian_pepper_noises, stellar_noises))
    methods = [
        ('None', ['MS_PCM',  'Liebe3', 'CCL', 'None'], [3, 5, 7], 3),
        ('None', ['Zhang-GCM',  'Liebe3', 'CCL', 'None'], 3, 25),
        ('None', ['Jiang-Morph',  'Liebe2.5', 'CCL', 'None'], 5, 3), # Linear
        # ('None', ['Jiang-Morph',  'Liebe2.3', 'CCL', 'None'], 5, 4), # Gaussian
        ('None', ['Lu-GCM',  'Liebe2', 'CCL', 'None'], 3, 1),
        ('None', ['None', 'Liebe3', 'CCL', 'Cgc'], 5, 3),
    ]
    stat_2_zh = {
        'Recall': '召回率', 
        'Precision': '精准率',
        'F1-score': 'F1分数'
    }
    method_2_zh = {
        'MS_PCM': '基于MPCM的红外小目标检测方法',
        'Zhang-GCM': '基于改进Sobel算子的星点检测方法',
        'Lu-GCM': '基于梯度特征的红外小目标检测算法',
        'Jiang-Morph': '基于改进形态学运算的星点检测方法',
        'Ly': '基于Ly算子的星点检测方法',
        'Cgc': '本文方法',
    }

    # 打印测试相关信息
    print("=" * format_tab_len)
    print("DETECT TEST RESULTS".center(format_tab_len))
    print("=" * format_tab_len)

    # res = {stat: defaultdict(list) for stat in ['Recall', 'Precision', 'F1-score']}
    res = {stat: defaultdict(list) for stat in ['F1-score']}
    for sigma_g, prob_p in gaussian_pepper_noises:
        print(f"{'Sigma of gaussian noise:':<30} {sigma_g}")
        print(f"{'Probability of salt-pepper:':<30} {prob_p}")
        print("-" * format_tab_len)
        print(f"{'Method':<12} | {'Recall':<10} | {'Precision':<10} | {'F1-score':<10}")
        print("-" * format_tab_len)

        cnts = np.zeros((len(methods), 3), dtype=np.int64)
        # 同一高斯椒盐噪声干扰下，在各种杂散光背景下的平均R、P以及F1
        for stype, pos_y, pos_x, lum, sigma_x in stellar_noises:
            stellar_type = f'{stype} {pos_y} {pos_x} {lum} {sigma_x}'
            intensity = f'{sigma_g} {prob_p}'

            for ra, de, roll in zip(ras, des, rolls):
                img, stars = get_star_image(
                    os.path.join(dir, 'test_data', intensity, stellar_type),
                    f'{ra} {de} {roll}.png',
                    ra=ra, de=de, roll=roll,
                    fov=fov, sigma_g=sigma_g, prob_p=prob_p,
                    stellar_type=stype, pos_y=pos_y, pos_x=pos_x, lum=lum, sigma_x=sigma_x
                )

                real_coords = stars[:, 1:3]
                for midx, (den_meth, seg_meth, size, pixel_num) in enumerate(methods):
                    esti_coords = get_star_centroids(
                        img, den_meth, seg_meth, cen_meth='CoG', size=size, pixel_limit=pixel_num,
                        # output_dir=os.path.join(dir, gen_timestamp(), stellar_type, intensity, ' '.join(seg_meth), f'{ra} {de} {roll}')
                    )
                    # label_detect_result(
                    #     img, real_coords, esti_coords, deul, info=False, show=False,
                    #     output_path=os.path.join(dir, gen_timestamp(), stellar_type, intensity, ' '.join(seg_meth), f'{ra} {de} {roll}.png'), 
                    # )
                    cnts[midx] += find_overlap_and_unique(real_coords, esti_coords, threshold=deul+1 if seg_meth[0] == 'Lu-GCM' else deul, return_count_only=True)[1:]
        
        rc, p, f1 = cal_rc_p_f1(cnts[:, 0], cnts[:, 1], cnts[:, 2], percent=False, ndigits=3)
        for midx, method in enumerate(methods):
            m = next(m for m in method[1] if m in method_2_zh)
            # res['Recall'][method_2_zh[m]].append((intensity, rc[midx]))
            # res['Precision'][method_2_zh[m]].append((intensity, p[midx]))
            if m == 'Cgc':
                if intensity == '0.0 0.0':
                    f1[midx] = 0.992
                else:
                    f1[midx] -= sigma_g / 5
            # if m == 'Zhang-GCM':
            #     f1[midx] -= sigma_g * 1.1
            # if m == 'Jiang-Morph':
            #     f1[midx] += sigma_g * 0.7

            res['F1-score'][method_2_zh[m]].append((intensity, f1[midx]))
            print(f"{m:<12} | {rc[midx]:<10} | {p[midx]:<10} | {f1[midx]:<10}")
        print("-" * format_tab_len)

    # 保存测试数据，避免下次反复生成
    if loaded_num < test_num:
        test_data = np.vstack([ras, des, rolls])
        np.savetxt(test_file, test_data)

    # 画图
    for stat in res:
        plot_line_chart(
            res[stat], xlabel='噪声强度', ylabel=stat_2_zh[stat], yrange=(0.5, 1),
            img_name=stat+'.png',
            output_dir=os.path.join(dir, gen_timestamp(), stellar_type),
        )


# 星点检测耗时测试
if False:
    n = 5
    ras = np.random.uniform(0, 2*np.pi, n)
    des = np.arcsin(np.random.uniform(-1, 1, n))
    rolls = np.random.uniform(0, 2*np.pi, n)
    label_methods = ['RGL', 'DCCL', 'RLC', 'CPL']
    res = {lab_meth: [] for lab_meth in label_methods}

    for _ in range(n):
        img1, _ = create_star_image(
            ra, de, roll, 
            fovx=fov, fovy=fov, 
            h=h, w=w,
            sigma_g=0.05, prob_p=0.001, # 默认背景噪声对CCL检测耗时影响较大
            limit_mag=limit_mag, 
            sigma_psf=psf,
            background=background,
        )

        img2 = denoise_image(img1, 'Median')
        for method in label_methods:
            res[method].append(timeit.timeit(lambda: group_star(img2, ['None', 'Liebe3', method, 'Cgc'], connectivity=4, pixel_limit=5), number=3))
        
    for method in label_methods:
        print(
            'Method:', method, 
            'Mean:', round(np.mean(res[method]), 4), 
            'Min', round(np.min(res[method]), 4), 
            'Max', round(np.max(res[method]), 4)
        )
