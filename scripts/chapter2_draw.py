import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import radians, tan

from simulate import create_star_image, draw_star
from utils import get_angdist, find_overlap_and_unique


# 灰度分布模型
if True:
    x, y = np.meshgrid(np.arange(7), np.arange(7))
    x, y = x.flatten() + 0.5, y.flatten() + 0.5
    dx = dy = 0.8
    
    mag = 3
    psf = 0.7
    z = draw_star((3.5, 3.5), mag, np.zeros((7, 7), dtype=np.uint8), sigma=psf)
    dz = z.flatten()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    bars = ax.bar3d(
        x, y, 0, dx, dy, dz, 
        color='gray',
        linewidth=0.3,
        alpha=0.9
    )           

    plt.show()


ra, de, roll = radians(249.2104), radians(-12.0386), radians(-13.3845)
h = w = 512
fov = 10
limit_mag = 6
f = h/tan(radians(fov/2))

# 角距验证
if True:
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
    vagds, ragds = get_angdist(vvs), get_angdist(rvs)

    # print validation results
    for i in range(n):
        for j in range(i+1, n):
            print(i, j, vagds[i, j], ragds[i, j])


def label_image(img: np.ndarray, coords: np.ndarray, color: tuple=(0, 255, 0),  radius: int=5):
    '''
        Label image with colored circles.
    '''
    for coord in coords:
        row, col = int(coord[0]), int(coord[1])
        cv2.circle(img, (col, row), radius, color, 1)
    return img


# 噪声仿真测试
if True:
    os.makedirs('res/chapter2/sim', exist_ok=True)

    # backgroud noise
    img, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        limit_mag=limit_mag, 
        sigma_g=0.05, prob_p=0.001
    )
    cv2.imwrite('res/chapter2/sim/noise.png', img)

    ids = stars[:, 0].astype(np.int64)
    coords = stars[:, 1:3]

    # positional noise
    img, _ = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        limit_mag=limit_mag, 
        sigma_pos=10
    )
    cv2.imwrite('res/chapter2/sim/pos.png', img)

    # magnititude noise
    img, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        limit_mag=limit_mag, 
        sigma_mag=0.3
    )
    _, _, coords1, coords2 = find_overlap_and_unique(coords, stars[:, 1:3])
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = label_image(img, coords1, (0, 0, 255)) # miss
    img = label_image(img, coords2, (255, 0, 0)) # false
    cv2.imwrite('res/chapter2/sim/mag.png', img)

    # false star noise
    img, stars = create_star_image(
        ra, de, roll, 
        h=h, w=w, 
        fovy=fov, fovx=fov, 
        limit_mag=limit_mag, 
        num_fs=2
    )
    mask = stars[:, 0] == -1
    coords = stars[mask, 1:3]

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = label_image(img, coords, (255, 0, 0)) # false
    cv2.imwrite('res/chapter2/sim/fs.png', img)
