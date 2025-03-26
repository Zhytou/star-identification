import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from simulate import draw_star, create_star_image
from extract import cal_center_of_gravity, get_star_centroids


def draw_err_3d(num_step: int, sigmas: list[float], mag: float=5.5):
    '''
        Draw the 3D error image.
    '''
    center = 4, 4

    err = np.zeros((num_step, len(sigmas)))
    for i in range(num_step):
        for j, sigma in enumerate(sigmas):
            wind = draw_star((center[0]+i/num_step, center[1]), mag, np.zeros((9, 9), dtype=np.uint8), sigma)

            rows, cols = np.where(wind > 0)
            esti_coords = cal_center_of_gravity(wind, rows, cols, 'CoG', False)
            if sigma == 0.1:
                print(wind)
                print(rows, cols)
                print(esti_coords)
            err[i][j] = center[0]+i/num_step - esti_coords[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-0.5, 0.5, num_step)
    y = np.array(sigmas)
    X, Y = np.meshgrid(x, y, indexing='ij')
    surf = ax.plot_wireframe(X, Y, err, cmap='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('sigma')
    ax.set_zlabel('systematic error')
    fig.colorbar(surf)
    plt.show()


draw_err_3d(100, [0.3, 0.4, 0.5, 0.6], 3.5)

def func(x, a, b):
    return a*x + b

def func2(x, a, b, c):
    return a*x**2 + b*x + c

def func3(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def func4(x, a, b, c):
    return a*np.sin(b*x+c)

# p0 = [1, 1, 0]
# popt, pcov = curve_fit(func4, esti_xs, errs, p0=p0)

# a, b, c = popt
# print(a, b, c)

if False:
    # final extract test
    # times of estimation using centroid algorithm
    num_esti_times = 3
    # random ra & de test
    num_test = 5
    # generate random right ascension[0, 360] and declination[-90, 90]
    ras = np.random.uniform(0, 2*np.pi, num_test)
    des = np.arcsin(np.random.uniform(-1, 1, num_test))
    # centroid position error
    # without/with compensation
    errs = [0.0, 0.0]

    # generate the star image
    for i in range(num_test):
        img, star_info = create_star_image(ras[i], des[i], 0, sigma_g=0.05, prob_p=0.0001)
        real_coords = np.array([x[1] for x in star_info])

        for compensated in [0, 1]:
            # estimate the centroids of the stars in the image
            arr_esti_coords = []
            for _ in range(num_esti_times):
                esti_coords = np.array(get_star_centroids(img, 'Liebe', 'CoG', compensated, wind_size=-1))
                arr_esti_coords.append(esti_coords)
            avg_esti_coords = np.mean(arr_esti_coords, axis=0)
            
            # calculate the distance between the real and estimated centroids
            dif_real_2_etsi = np.sum((real_coords[:, None] - avg_esti_coords)**2, axis=-1)
            errs[compensated] += np.sum(np.min(dif_real_2_etsi, axis=-1)) / len(real_coords)
     
    print(errs)
