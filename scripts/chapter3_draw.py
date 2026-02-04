import os, cv2, timeit
import numpy as np
import matplotlib.pyplot as plt
from math import radians
from collections import defaultdict
from simulate import draw_star, create_star_image, add_gaussian_and_pepper_noise, add_stellar_noise
from denoise import denoise_image
from detect import group_star, initialize_seeds, enhance_image, binarize_image, cal_gcm
from extract import get_star_centroids
from utils import gen_timestamp, gen_integer_approximation, find_overlap_and_unique, cal_doh, plot_well_grid, plot_gray_3d, cal_mse_psnr_ssim, cal_rc_p_f1, label_star_image, label_detect_result, label_grad_field, plot_line_chart


DEBUG = True

roi=2
limit_mag=6
background=7
psf=1


ra, de, roll=radians(29.2104), radians(-12.0386), radians(0)
h, w=512, 512
x, y=188, 169*2
fov=12


def get_star_image(file: str, sigma_g: float=0, prob_p: float=0, stellar_type: str='None', pos_y: float=0, pos_x: float=0, lum: float=0, sigma_x: float=0, sigma_y: float=None):
    '''
        Get star image.
    '''

    img, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        background=background,
        limit_mag=limit_mag, 
        roi=roi, sigma_psf=psf,
    )
    if not os.path.exists(file):
        img = add_stellar_noise(img, method=stellar_type, position=(pos_y, pos_x), luminosity=lum, sigma_x=sigma_x, sigma_y=sigma_y)
        img = add_gaussian_and_pepper_noise(img, sigma_g=sigma_g, prob_p=prob_p)
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
    img, stars = get_star_image(os.path.join(dir, 'img.png'), sigma_g=0.05, prob_p=0.005)

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


# 降噪算法测试参数
# 星图降噪效果测试（质量指标比较）
ra, de, roll=radians(29.2104), radians(-12.0386), radians(0) # 可能每个测试拍摄视角不同
h, w=512, 512
fov=12
if False:
    dir = os.path.join('res/chapter3/denoise', gen_timestamp())
    res = {stat: defaultdict(list) for stat in ['PSNR', 'SSIM']}

    # 测试参数
    noises = [
        (0.03, 0.003), 
        (0.06, 0.006), 
        (0.09, 0.009)
    ]
    methods = [
        'Original', 'Noised', 
        # 'NLM', 'AMF', 
        'Bilateral', 'Wavelet',
        'EMF', 'CWM', 'CMG',
        'CNB', 
    ]

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

    for (sigma, prob) in noises:
        print(
            'DENOISE TEST'
            '\n--------------------------------'
            '\nSigma of gaussian noise:', sigma, 
            '\nProbability of pepper noise', prob, 
            '\n--------------------------------'
        )
        intensity = f'{sigma} {prob}'
        img1 = add_gaussian_and_pepper_noise(img0, sigma, prob)
        for method in methods:
            if method == 'Noised':
                img2 = img1
            elif method == 'Original':
                img2 = img0
            else:
                img2 = denoise_image(img1, method)

            # 计算降噪前后图像质量指标
            mse, psnr, ssim = cal_mse_psnr_ssim(img0, img2, ndigits=2)
            print(method, ' PSNR:', psnr, ' SSIM:', ssim,'\n',)

            res['PSNR'][method].append((intensity, psnr))
            res['SSIM'][method].append((intensity, ssim))
            if save:
                img3 = img2 if img2.dtype == np.uint8 else 255*img2
                cdir = os.path.join(dir, intensity)
                os.makedirs(cdir, exist_ok=True)
                cv2.imwrite(os.path.join(cdir, method+'.png'), img3)

    # for stat in res:
    #     plot_line_chart(
    #         res[stat], xlabel='噪声强度', ylabel=stat,
    #         yrange=(0, 1) if stat == 'SSIM' else (0, 50),
    #         img_name=stat+'.png',
    #         show=False, output_dir=dir
    #     )


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
    img, _ = get_star_image(
        os.path.join(dir, 'img.png'), sigma_g=0.05, prob_p=0.005,
        stellar_type='Gaussian', pos_y=h//2, pos_x=-w//4, lum=4, sigma_x=128
    )

    # 井字型
    if True:
        d = 33
        r = d // 2
        y, x = 280, 452
        y1, y2 = max(0, y - r), min(h, y + r + 1)
        x1, x2 = max(0, x - r), min(w, x + r + 1)
        plot_well_grid(img[y1:y2,x1:x2], d // 3, output_path=os.path.join(dir, 'well.png'))

    # DoH算子效果
    if True:
        coords = np.array([
            (280, 452),     # star mag 4.2
            (274, 242),     # star mag 5.7
            (65, 239),      # pepper noise 
        ])
        doh = cal_doh(img, sigma=1.5)
        gcm = cal_gcm(img, 'GCM')

        d = 7
        r = d // 2
        for y, x in coords:
            y1, y2 = max(0, y - r), min(h, y + r + 1)
            x1, x2 = max(0, x - r), min(w, x + r + 1)
            cv2.imwrite(os.path.join(dir, 'ori', f'{y}_{x}.png'), img[y1:y2, x1:x2])
            plot_gray_3d(
                doh[y1:y2, x1:x2], method='plot_surface', color_map='RdBu_r', 
                output_path=os.path.join(dir, 'doh', f'{y}_{x}.png')
            )

    # 梯度分布
    if False:
        d = 9
        coords = np.array([
            (280, 452),     # star mag 4.2
            (65, 239),      # pepper noise 
            (182, 84),      # stary light edge
        ])
        label_grad_field(
            img, coords, d, sigma=1, show=True, 
            output_dir=os.path.join(dir, 'gcm'), file_name='img.png'
        )


# 检测算法测试参数
# 选择一处恒星数量多、星等差异大的视场，从而说明检测算法针对不同星等的恒星均能有限检测
ra, de, roll=radians(25.0588), radians(-21.7205), radians(0)
fov=19.8
psf=1
roi=2
deul=2 # detection error upper limit
if False:
    dir = os.path.join('res/chapter3/detect', gen_timestamp())

    img0, stars = create_star_image(
        ra, de, roll, h=h, w=w, 
        fovx=fov, fovy=fov, 
        limit_mag=limit_mag, 
        background=background,
        sigma_psf=psf,
        roi=roi
    )
    real_coords = stars[:, 1:3]
    ids = stars[:, 0]

    # 测试参数
    noises = [
        ## Constant stellar background
        (0.03, 0.003, 'Constant', 0, 0, 7, 0),

        # Gasussian stellar background
        (0.03, 0.003, 'Gaussian', h//2, -w//4, 4.6, 128), 
        (0.03, 0.003, 'Gaussian', h//3, w//5*3, 5, 64), 

        # ## Linear stellar background
        (0.03, 0.003, 'Linear_X', 0, 0, 5.7, 128), 
    ]
    methods = [
        ('None', ['PCM',  'Liebe3', 'CCL', 'None'], 5, 4),
        ('None', ['Zhang-GCM',  'Liebe3', 'CCL', 'None'], 3, 25),
        ('None', ['Jiang-Morph',  'Liebe3', 'CCL', 'None'], 5, 3),
        ('None', ['BEF',  'Liebe3', 'CCL', 'None'], 5, 3),
        ('None', ['None', 'Liebe3', 'RGL', 'Cgc'], 5, 3),

        # ('None', ['None', 'Liebe3', 'RGL', 'Ly'], 5, 3),
        # ('None', ['Lu-GCM',  'Liebe3', 'CCL', 'None'], 5, 3),
        # ('None', ['Max-Median', 'Liebe3', 'CCL', 'None'], 7, 3),
    ]

    for (sigma_g, prob_p, stype, pos_y, pos_x, lum, sigma_x) in noises:
        img1 = add_stellar_noise(img0, method=stype, position=(pos_y, pos_x), luminosity=lum, sigma_x=sigma_x)
        img2 = add_gaussian_and_pepper_noise(img1, sigma_g=sigma_g, prob_p=prob_p)

        intensity = f'{sigma_g} {prob_p}'
        stellar_type = f'{stype} {pos_y} {pos_x} {lum} {sigma_x}'

        # label_star_image(img2, real_coords, np.full_like(ids, -1), axis_on=False, output_path=os.path.join(dir, stellar_type, intensity, 'Original.png'))
        for den_meth, seg_meth, wind_size, pixel_num in methods:
            esti_coords = get_star_centroids(
                img2, den_meth, seg_meth, cen_meth='CoG', size=wind_size, pixel_limit=pixel_num,
                output_dir=os.path.join(dir, stellar_type, intensity, ' '.join(seg_meth))
            )

            label_detect_result(
                img2, real_coords, esti_coords, deul, info=False, show=False,
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


# 星点检测数量对比
if False:
    dir = os.path.join('res/chapter3/detect', gen_timestamp())

    # 测试参数
    n = 5
    flag = True # 是否使用统一测试图片
    ras = np.full(n, ra) if flag else np.random.uniform(0, 2*np.pi, n)
    des = np.full(n, de) if flag else np.arcsin(np.random.uniform(-1, 1, n))
    rolls = np.full(n, 0) if flag else np.random.uniform(0, 2*np.pi, n)
    stellar_noises = [
        # Constant stellar background
        # ('Constant', 0, 0, 7, 0),

        # Gasussian stellar background
        ('Gaussian', h//2, -w//4, 5.5, 128),
    ]
    gaussian_pepper_noises = [
        (0.01, 0.001),
        (0.02, 0.002),
        (0.03, 0.003),
        (0.04, 0.004),
        (0.05, 0.005), 
    ]
    methods = [
        ('None', ['PCM',  'Liebe3', 'CCL', 'None'], 5, 4),
        ('None', ['Zhang-GCM',  'Liebe3', 'CCL', 'None'], 3, 25),
        ('None', ['Jiang-Morph',  'Liebe3', 'CCL', 'None'], 5, 3),
        ('None', ['BEF',  'Liebe3', 'CCL', 'None'], 5, 3),
        ('None', ['None', 'Liebe3', 'RGL', 'Cgc'], 5, 3),
        # ('None', ['None', 'Liebe3', 'RGL', 'Ly'], 5, 3),
    ]
    abbr_2_name = {
        'Recall': '召回率', 
        'Precision': '精准率',
        'F1-score': 'F1分数'
    }

    # 打印测试相关信息
    print(
        'Detect Test',
        '\n-----------------------------',
        '\nNumber of test:', n,
        '\nRas:', ras, '\nDes:', des,
    )
    for stype, pos_y, pos_x, lum, sigma_x in stellar_noises:
        res = {stat: defaultdict(list) for stat in ['Recall', 'Precision', 'F1-score']}
        cnts = np.zeros((len(methods), 3), dtype=np.int64)

        stellar_type = f'{stype} {pos_y} {pos_x} {lum} {sigma_x}'
        for sigma_g, prob_p in gaussian_pepper_noises:
            intensity = f'{sigma_g} {prob_p}'

            for ra, de, roll in zip(ras, des, rolls):
                img0, stars = create_star_image(
                    ra, de, roll, h=h, w=w, 
                    fovx=fov, fovy=fov, sigma_psf=psf,
                    limit_mag=limit_mag, 
                    background=background,
                    roi=roi
                )
                img1 = add_stellar_noise(img0, method=stype, position=(pos_y, pos_x), luminosity=lum, sigma_x=sigma_x)
                img2 = add_gaussian_and_pepper_noise(img1, sigma_g=sigma_g, prob_p=prob_p)

                real_coords = stars[:, 1:3]
                for midx, (den_meth, seg_meth, size, pixel_num) in enumerate(methods):
                    esti_coords = get_star_centroids(img2, den_meth, seg_meth, cen_meth='CoG', size=size, pixel_limit=pixel_num)
                    cnts[midx] += find_overlap_and_unique(real_coords, esti_coords, threshold=deul, return_count_only=True)[1:]
            
            rc, p, f1 = cal_rc_p_f1(cnts[:, 0], cnts[:, 1], cnts[:, 2], percent=False, ndigits=3)
            for midx, (den_meth, seg_meth, size, pixel_num) in enumerate(methods):
                res['Recall'][' '.join(seg_meth)].append((intensity, rc[midx]))
                res['Precision'][' '.join(seg_meth)].append((intensity, rc[midx]))
                res['F1-score'][' '.join(seg_meth)].append((intensity, rc[midx]))

                if midx == 0:
                    print(
                        'Method:', den_meth, seg_meth, pixel_num,
                        '\n---------------------------------'
                    )
                else:
                    print(
                        'Detect result:', cnts[midx], 
                        '\nRecall:', rc[midx], ' Precsion:', p[midx], ' F1-score:', f1[midx],
                    )

        for stat in res:
            plot_line_chart(
                res[stat], xlabel='噪声强度', ylabel=abbr_2_name[stat],
                img_name=stat+'.png',
                output_dir=os.path.join(dir, stellar_type),
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
