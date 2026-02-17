import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import radians, tan

from simulate import create_star_image, draw_star, add_stellar_noise, add_gaussian_and_pepper_noise
from utils import cal_angdist, plot_gray_3d, label_star_image


# 灰度分布模型
if False:
    mag = 3
    psf = 0.7
    img = draw_star(np.zeros((7, 7), dtype=np.uint8), (3.5, 3.5), mag, sigma=psf)
    plot_gray_3d(img, 'bar3d')


ra, de, roll = radians(249.2104), radians(-12.0386), radians(-13.3845)
h = w = 512
fov = 10
limit_mag = 6
f = h/tan(radians(fov/2))


# 无噪声仿真图
if False:
    img, stars = create_star_image(ra, de, roll, h=h, w=w, limit_mag=limit_mag, fovx=fov, fovy=fov, rot_meth=1)
    ids = stars[:, 0].astype(int)
    coords = stars[:, 1:3].astype(int)
    label_star_image(img, coords, ids)


# 角距验证
if False:
    img0, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        limit_mag=limit_mag, 
    )

    coords = stars[:, 1:3]
    img1 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    for row, col in coords:
        row, col = int(row), int(col)
        img1 = cv2.circle(img1, (col, row), 5, (255, 0, 0), 1)
    cv2.imwrite('res/chapter2/sim/coord.png', img1)

    # print stars info
    for star in stars:
        print(
            int(star[0]),       # id
            round(star[1], 2),  # row
            round(star[2], 2),  # col
            round(star[3], 2),  # ra
            round(star[4], 2),  # de
        )
    n = len(stars)
    ras, des = stars[:, 3], stars[:, 4]

    # view vectors
    vvs = np.full((n, 3), f)
    vvs[:, 0] = coords[:, 1]-w/2
    vvs[:, 1] = coords[:, 0]-h/2

    # reference vectors
    rvs = np.zeros((n, 3))
    rvs[:, 0] = np.cos(ras) * np.cos(des)
    rvs[:, 1] = np.sin(ras) * np.cos(des)
    rvs[:, 2] = np.sin(des)

    # angular distances
    vagds, ragds = cal_angdist(vvs), cal_angdist(rvs)

    # print validation results
    for i in range(n):
        for j in range(i+1, n):
            print(i, j, vagds[i, j], ragds[i, j])


# 噪声仿真测试
os.makedirs('res/chapter2/sim', exist_ok=True)


def extract_rect(img, top_left: tuple[int, int], bot_right: tuple[int, int], line_color: tuple[int, int, int]=(0, 255, 255), line_width: int=2):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    x1, y1 = top_left
    x2, y2 = bot_right
    crop_img = img[y1+line_width:y2-line_width, x1+line_width:x2-line_width]
    cv2.rectangle(img, top_left, bot_right, line_color, line_width)

    return img, crop_img


# 成像噪声/杂散光干扰
if True:
    img0, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        limit_mag=limit_mag, 
        background=7,
    )
    cv2.imwrite('res/chapter2/sim/original.png', img0)
    
    row, col = 100, 200
    d = 32
    top_left = (col-d-1, row-d-1)
    bot_right = (col+d+1, row+d+1)

    img1 = add_gaussian_and_pepper_noise(img0, sigma_g=0.05, prob_p=0.001)
    img1, crop_img1 = extract_rect(img1, top_left, bot_right)    
    cv2.imwrite('res/chapter2/sim/noise.png', img1)
    cv2.imwrite('res/chapter2/sim/noise_scale.png', crop_img1)

    img2 = add_stellar_noise(img0, method='Gaussian', position=(h//2, w//3), luminosity=5.3, sigma_x=64)
    img2, crop_img2 = extract_rect(img2, top_left, bot_right)
    cv2.imwrite('res/chapter2/sim/gaussian.png', img2)
    cv2.imwrite('res/chapter2/sim/gaussian_scale.png', crop_img2)

    img3 = add_stellar_noise(img0, method='Linear_X', position=(h//2, 0), luminosity=5.4, sigma_x=64)
    img3, crop_img3 = extract_rect(img3, top_left, bot_right)
    cv2.imwrite('res/chapter2/sim/linear.png', img3)
    cv2.imwrite('res/chapter2/sim/linear_scale.png', crop_img3)
