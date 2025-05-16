import os
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import radians, tan, degrees

from simulate import create_star_image


if False:
    z = draw_star((3.5, 3.5), 5, np.zeros((7, 7), dtype=np.uint8), 0.7)

    x, y = np.meshgrid(np.arange(7), np.arange(7))
    x = x.flatten() + 0.5  # 偏移到像素中心
    y = y.flatten() + 0.5  # 偏移到像素中心
    # z = np.zeros_like(x)
    dx = dy = 0.8  # 柱子宽度
    dz = z.flatten()  # 高度为归一化灰度值

    # 创建三维画布
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制柱状图（颜色与高度同步）
    bars = ax.bar3d(x, y, 0, dx, dy, dz, 
                    color='gray',
                    # edgecolor='k',         # 黑色边框
                    linewidth=0.3,         # 边框粗细
                    alpha=0.9)            # 透明度

    plt.show()


h, w = 512, 512
fov = 10
limit_mag = 6
ra, de, roll = radians(249.2104), radians(-12.0386), radians(-13.3845)
f = 58e-3
mtot = 2*tan(radians(fov/2))*f
xpixel = w/mtot
ypixel = h/mtot

if False:
    # stars= id, row, col, ra, de, mag
    img0, stars = create_star_image(ra, de, roll, h=h, w=w, fov=fov, limit_mag=limit_mag, f=f)

    coords = stars[:, 1:3]
    img1 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    for row, col in coords:
        row, col = int(row), int(col)
        img1 = cv2.circle(img1, (col, row), 5, (255, 0, 0), 1)
    cv2.imwrite('res/chapter2/sim/coord.png', img1)

    ids = stars[:, 0].astype(np.int64)
    ras, des = stars[:, 3], stars[:, 4]
    Xs = np.cos(ras) * np.cos(des)
    Ys = np.sin(ras) * np.cos(des)
    Zs = np.sin(des)

    print(ids, coords, np.degrees(ras), np.degrees(des))

    num_star = len(stars)
    for i in range(num_star):
        idi = ids[i]
        # celestial cartesian coordinate vector
        Xi, Yi, Zi = Xs[i], Ys[i], Zs[i]
        Vci = np.array([Xi, Yi, Zi]).transpose()
        # screen(image) coordinate vector
        rowi, coli = coords[i]
        xi, yi = (coli-w/2)/xpixel, (rowi-h/2)/ypixel
        Vsi = np.array([xi, yi, f]).transpose()
        
        for j in range(i+1, num_star):
            idj = ids[j]
            # celestial cartesian coordinate vector
            Xj, Yj, Zj = Xs[j], Ys[j], Zs[j]
            Vcj = np.array([Xj, Yj, Zj])
            # screen(image) coordinate vector
            rowj, colj = coords[j]
            xj, yj = (colj-w/2)/xpixel, (rowj-h/2)/ypixel
            Vsj = np.array([xj, yj, f])
           
            dc = np.arccos(np.dot(Vci, Vcj) / (np.linalg.norm(Vci) * np.linalg.norm(Vcj)))
            ds = np.arccos(np.dot(Vsi, Vsj) / (np.linalg.norm(Vsi) * np.linalg.norm(Vsj)))

            print(idi, idj, dc, ds)


if True:
    os.makedirs('res/chapter2/sim', exist_ok=True)

    # backgroud noise
    img, stars = create_star_image(ra, de, roll, sigma_g=0.1, prob_p=0.001, fov=fov, limit_mag=limit_mag, h=h, w=w, f=f)
    cv2.imwrite('res/chapter2/sim/noise.png', img)

    real_ids = stars[:, 0].astype(np.int64)
    real_rows, real_cols = stars[:, 1:3].astype(np.int64).transpose()

    # positional noise
    img, _ = create_star_image(ra, de, roll, fov=fov, limit_mag=limit_mag, h=h, w=w, f=f, sigma_pos=10)
    cv2.imwrite('res/chapter2/sim/pos.png', img)

    # magnititude noise
    # img, stars = create_star_image(ra, de, roll, fov=fov, limit_mag=limit_mag, h=h, w=w, f=f, sigma_mag=0.3)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # ids = stars[:, 0].astype(np.int64)
    # rows, cols = stars[:, 1:3].astype(np.int64).transpose()
    # for idi, row, col in zip(ids, rows, cols):
    #     if idi not in real_ids:
    #         img = cv2.circle(img, (col, row), 5, (0, 0, 255), 1)
    # for idi, row, col in zip(real_ids, real_rows, real_cols):
    #     if idi not in ids:
    #         img = cv2.circle(img, (col, row), 5, (255, 255, 0), 1)
    # cv2.imwrite('res/chapter2/sim/mag.png', img)

    # false star noise
    # img, stars = create_star_image(ra, de, roll, fov=fov, limit_mag=limit_mag, h=h, w=w, f=f, num_fs=1)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # ids = stars[:, 0].astype(np.int64)
    # rows, cols = stars[:, 1:3].astype(np.int64).transpose()
    # for idi, row, col in zip(ids, rows, cols):
    #     if idi == -1:
    #         img = cv2.circle(img, (col, row), 5, (0, 0, 255), 1)
    # cv2.imwrite('res/chapter2/sim/fs.png', img)

    # img, _ = create_star_image(ra, de, roll, fov=fov, limit_mag=limit_mag, h=h, w=w, f=f, num_ms=1)
    # cv2.imwrite('res/chapter2/sim/ms.png', img)