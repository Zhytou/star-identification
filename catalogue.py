import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.axes._axes as axes
from math import radians, sqrt, cos, pi

from simulate import FOV


def draw_star_distribution(catalogue: pd.DataFrame, ax: axes.Axes, title: str):
    '''
        Draw the distribution of stars in the celestial sphere.
    Args:
        catalogue: the star catalogue
        ax: the axes to draw
        title: the title of the plot
    '''
    ras, des = np.degrees(catalogue['RA']), np.degrees(catalogue['DE'])
    ax.scatter(ras, des, s=1)
    ax.set_title(title)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('RA')
    ax.set_ylabel('DE')


def draw_probability_versus_star_num_within_FOV(catalogue: pd.DataFrame, ax: axes.Axes, title: str, num_vector: int=10000):
    '''
        Draw the probability distribution of the number of stars within FOV.
    Args:
        catalogue: the original catalogue
        ax: the axes to draw
        title: the title of the plot
        num_vector: the number of vectors to be generated
    '''
    # calculate the half of FOV diagonal distance
    R = sqrt((radians(FOV)**2)+(radians(FOV)**2))/2
    
    # generate random right ascension[-pi, pi] and declination[-pi/2, pi/2], method from http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09
    ras = np.random.uniform(-pi, pi, num_vector)
    des = np.arcsin(np.random.uniform(-1, 1, num_vector))
    
    # record the result: star_num -> sample_num
    table = {}
    for ra, de in zip(ras, des):
        # get the rough range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        # print(de1, de2, ra1, ra2)
        # get the num of stars within FOV
        stars_within_FOV = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2) & (de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)]
        star_num = len(stars_within_FOV)
        table[star_num] = table.get(star_num, 0) + 1
    
    # sort table items by key
    num_stars = sorted(table.keys())
    probability = [(table[num_star]*100.0)/num_vector for num_star in num_stars]
    print(num_stars, probability)

    # calculate average number of star in FOV
    average_num = sum([num_star*table[num_star] for num_star in num_stars])/num_vector
    print(average_num)

    ax.plot(num_stars, probability)
    ax.vlines(average_num, 0, 100, linestyles='dashed', colors='red')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Number of stars within FOV')
    ax.set_ylabel('Probability%')


def filter_catalogue(catalogue: pd.DataFrame, limit_mv: float=6.0, num_vector: int=100000, num_star_per_region_limit: int=10, angular_distance_limit: float=0.5) -> pd.DataFrame:
    '''
        Filter navigation stars using method from http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09.
    Args:
        catalogue: the original catalogue
        limit_mv: the magnitude limit
        num_vector: the number of vectors to be generated
        num_star_per_region_limit: the max number of stars in each region(if the number of stars in a region is more than this limit, choose the top num_star_per_region_limit brightest one)
        angular_distance_limit: the angular distance limit in degrees(if two stars' angular distance is less than this limit, choose the brighter one)
    Returns:    
        the filtered catalogue
    '''

    # calculate the half of FOV diagonal distance
    R = sqrt((radians(FOV)**2)+(radians(FOV)**2))/2

    # eliminate the stars with magnitude > limit_mv
    catalogue = catalogue[catalogue['Magnitude'] <= limit_mv].reset_index(drop=True)

    # convert to celestial rectangular coordinate system
    positions = pd.DataFrame()
    positions['X'] = np.cos(catalogue['RA'])*np.cos(catalogue['DE'])
    positions['Y'] = np.sin(catalogue['RA'])*np.cos(catalogue['DE'])
    positions['Z'] = np.sin(catalogue['DE'])

    # calculate the angular distance between each pair of stars
    positions = positions.to_numpy()
    norms = np.linalg.norm(positions, axis=1)
    inner_products = positions @ positions.T
    # do rounding because some cos_theta are slightly greater than 1 or less than -1 as a result of precision problem
    cos_theta = np.round(inner_products/(norms*norms[:, None]), 6)
    angular_distance = np.arccos(cos_theta)

    # get small angular distance star pairs
    idx1, idx2 = np.nonzero(angular_distance < radians(angular_distance_limit))
    # avoid idx1[i] == idx2[i]
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    star_pairs = set()
    darker_idx = set()
    for i1, i2 in zip(idx1, idx2):
        # avoid duplicate star pairs because of the symmetry of the angular distance matrix
        if (i1, i2) in star_pairs or (i2, i1) in star_pairs:
            continue
        if i1 in darker_idx or i2 in darker_idx:
            continue
        star_pairs.add((i1, i2))
        mv1, mv2 = catalogue.loc[i1, 'Magnitude'], catalogue.loc[i2, 'Magnitude']
        if mv1 < mv2:
            darker_idx.add(i1)
        else:
            darker_idx.add(i2)
    darker_idx = list(darker_idx)
    # eliminate the darker stars from small angular distance star pairs
    catalogue = catalogue[~catalogue.index.isin(darker_idx)]

    # generate uniform distributed random vectors, which seperate the celestial sphere into multiple regions
    ras = np.random.uniform(-pi, pi, num_vector)
    des = np.arcsin(np.random.uniform(-1, 1, num_vector))

    # filtered star ids
    filtered_stars_id = set()
    for ra, de in zip(ras, des):
        # calculate the range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = de - R, de + R
        stars_within_FOV = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2) & (de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)]
        
        # screen the top star_num_per_region_limit brightest stars in each region
        if len(stars_within_FOV) > num_star_per_region_limit:
            stars_within_FOV = stars_within_FOV.nsmallest(num_star_per_region_limit, 'Magnitude')
        filtered_stars_id.update(stars_within_FOV['Star ID'])

    return catalogue[catalogue['Star ID'].isin(filtered_stars_id)]


if __name__ == '__main__':
    file = 'catalogue/below_6.0_SAO.csv'
    limit_mv = 5.6
    filtered_file = f'catalogue/filtered_below_{limit_mv}_SAO.csv'

    df = pd.read_csv(file, usecols=["Star ID", "RA", "DE", "Magnitude"])
    if os.path.exists(filtered_file):
        filtered_df = pd.read_csv(filtered_file)
    else:
        filtered_df = filter_catalogue(df, limit_mv).reset_index(drop=True)
        filtered_df.to_csv(filtered_file)

    fig1, axs1 = plt.subplots(2)
    draw_star_distribution(df, axs1[0], "Original")
    draw_star_distribution(filtered_df, axs1[1], "Filtered")

    fig2, axs2 = plt.subplots(2)
    draw_probability_versus_star_num_within_FOV(df, axs2[0], "Original")
    draw_probability_versus_star_num_within_FOV(filtered_df, axs2[1], "Filtered")

    plt.show()