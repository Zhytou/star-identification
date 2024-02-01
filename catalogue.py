import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, sqrt, cos
from simulate import FOVx, FOVy


def draw_star_distribution(catalogue: pd.DataFrame):
    ras, des = catalogue['RA'], catalogue['DE']
    plt.scatter(ras, des, s=1)
    plt.show()


def draw_propability_versus_star_num_within_FOV(catalogue: pd.DataFrame, num_vector: int=10000):
    # calculate the half of FOV diagonal distance
    R = sqrt((radians(FOVx)**2)+(radians(FOVy)**2))/2
    
    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.randint(-180, 180, num_vector)
    des = np.random.uniform(-90, 90, num_vector)
    
    # record the result: star_num -> sample_num
    table = {}
    for ra, de in zip(ras, des):
        # get the range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        # get the num of stars within FOV
        stars_in_ra_range = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2)]
        stars_in_de_range = catalogue[(de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)]
        stars_in_de_range = stars_in_de_range[['Star ID']].copy()
        stars_within_FOV = pd.merge(stars_in_ra_range, stars_in_de_range, on="Star ID")
        star_num = len(stars_within_FOV)
        print(star_num)
        # table[star_num] += 1
    print(table)
    # plt.


if __name__ == '__main__':
    df = pd.read_csv("catalogues/Below_6.0_SAO.csv", usecols=["Star ID", "RA", "DE", "Magnitude"])
    # before uniform
    
    draw_propability_versus_star_num_within_FOV(df, 1000)
    # after uniform