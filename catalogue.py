import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.axes._axes as axes
from math import radians, sqrt, cos
from simulate import FOVx, FOVy


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
    R = sqrt((radians(FOVx)**2)+(radians(FOVy)**2))/2
    
    # generate random right ascension[-180, 180] and declination[-90, 90]
    ras = np.random.uniform(-180, 180, num_vector)
    des = np.random.uniform(-90, 90, num_vector)
    
    # record the result: star_num -> sample_num
    table = {}
    for ra, de in zip(ras, des):
        ra, de = radians(ra), radians(de)
        # get the rough range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        # get the num of stars within FOV
        stars_in_ra_range = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2)]
        stars_in_de_range = catalogue[(de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)]
        stars_in_de_range = stars_in_de_range[['Star ID']].copy()
        stars_within_FOV = pd.merge(stars_in_ra_range, stars_in_de_range, on="Star ID")
        star_num = len(stars_within_FOV)
        table[star_num] = table.get(star_num, 0) + 1
    
    # sort table items by key
    num_stars = sorted(table.keys())
    probability = [(table[num_star]*100.0)/num_vector for num_star in num_stars]
    print(num_stars, probability)
    ax.plot(num_stars, probability)
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Number of stars within FOV')
    ax.set_ylabel('Probability%')


def filter_catalogue(catalogue: pd.DataFrame, limit_mv: float=6.0, num_vector: int=10000, num_star_per_region_limit: int=12, angular_distance_limit: float=1) -> pd.DataFrame:
    '''
    Filter navigation stars using method from http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09.
    Args:
        catalogue: the original catalogue
        limit_mv: the magnitude limit
        num_vector: the number of vectors to be generated
        num_star_per_region_limit: the max number of stars in each region(if the number of stars in a region is more than this limit, choose the top num_star_per_region_limit brightest one)
        angular_distance_limit: the angular distance limit(if two stars' angular distance is less than this limit, choose the brighter one)
    Returns:    
        the filtered catalogue
    '''
    # calculate the half of FOV diagonal distance
    R = sqrt((radians(FOVx)**2)+(radians(FOVy)**2))/2

    # eliminate the stars with magnitude > limit_mv
    catalogue = catalogue[catalogue['Magnitude'] <= limit_mv]

    # generate uniform distributed random vectors, which seperate the celestial sphere into multiple regions
    ras = np.random.uniform(-180, 180, num_vector)
    des = np.random.uniform(-90, 90, num_vector)

    # filtered star ids
    filtered_stars_id = set()
    for ra, de in zip(ras, des):
        ra, de = radians(ra), radians(de)
        # calculate the range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = de - R, de + R

        # if the region include the pole, adjust the range
        if de1 < radians(-90) or de2 > radians(90):
            de1 = max(de1, radians(-90))
            de2 = min(de2, radians(90))
        stars_in_ra_range = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2)]
        stars_in_de_range = catalogue[(de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)]
        stars_in_de_range = stars_in_de_range[['Star ID']].copy()
        stars_within_FOV = pd.merge(stars_in_ra_range, stars_in_de_range, on="Star ID")
        
        # convert to celestial rectangular coordinate system
        stars_within_FOV['X'] = np.cos(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
        stars_within_FOV['Y'] = np.sin(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
        stars_within_FOV['Z'] = np.sin(stars_within_FOV['DE'])

        # eliminate the darker stars from small angular distance star pairs
        stars_pos = stars_within_FOV[['X', 'Y', 'Z']].values
        eliminated_stars_id = set()
        for i, a in enumerate(stars_pos):
            for j, b in enumerate(stars_pos):
                if i == j:
                    continue
                # calculate the angular distance: theta = arccos(a*b/|a||b|)
                angular_distance = np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                if angular_distance < angular_distance_limit:
                    if stars_within_FOV.loc[i, 'Magnitude'] > stars_within_FOV.loc[j, 'Magnitude']:
                        eliminated_stars_id.add(j)
                    else:
                        eliminated_stars_id.add(i)
        stars_within_FOV = stars_within_FOV[~stars_within_FOV['Star ID'].isin(eliminated_stars_id)]

        # screen the top star_num_per_region_limit brightest stars in each region
        if len(stars_within_FOV) > num_star_per_region_limit:
            stars_within_FOV = stars_within_FOV.nsmallest(num_star_per_region_limit, 'Magnitude')

        filtered_stars_id.update(stars_within_FOV['Star ID'])

    return catalogue[catalogue['Star ID'].isin(filtered_stars_id)]


if __name__ == '__main__':
    file = 'catalogues/Below_6.0_SAO.csv'
    limit_mv = 5.7
    filtered_file = f'catalogues/Filtered_Below_{limit_mv}_SAO.csv'

    df = pd.read_csv(file, usecols=["Star ID", "RA", "DE", "Magnitude"])
    if os.path.exists(filtered_file):
        filtered_df = pd.read_csv(filtered_file)
    else:
        filtered_df = filter_catalogue(df, 5.7).reset_index(drop=True)
        filtered_df.to_csv(filtered_file)

    fig1, axs1 = plt.subplots(2)
    draw_star_distribution(df, axs1[0], "Original")
    draw_star_distribution(filtered_df, axs1[1], "Filtered")

    fig2, axs2 = plt.subplots(2)
    draw_probability_versus_star_num_within_FOV(df, axs2[0], "Original")
    draw_probability_versus_star_num_within_FOV(filtered_df, axs2[1], "Filtered")

    plt.show()