import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.axes._axes as axes
from math import radians, sqrt, tan, sin, cos, pi

from simulate import f, FOV


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


def get_rotation_matrix(ra: float, de: float, roll: float) -> np.ndarray:
    """
        Get the rotation matrix from star sensor coordinates to celestial coordinates. Note that M is an orthogonal matrix, which means the transpose of M represents the transformation matrix from celestial coordinates to star sensor coordinates.
    Args:
        ra: right ascension in radians
        de: declination in radians
        roll: roll angle of star sensor in radians
    Returns:
        M: rotation matrix
    """
    a1 = sin(ra)*cos(roll) - cos(ra)*sin(de)*sin(roll)
    a2 = -sin(ra)*sin(roll) - cos(ra)*sin(de)*cos(roll)
    a3 = -cos(ra)*cos(de)
    b1 = -cos(ra)*cos(roll) - sin(ra)*sin(de)*sin(roll)
    b2 = cos(ra)*sin(roll) - sin(ra)*sin(de)*cos(roll)
    b3 = -sin(ra)*cos(de)
    c1 = cos(ra)*sin(roll)
    c2 = cos(de)*cos(roll)
    c3 = -sin(de)
    M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
        
    return M


def draw_probability_versus_star_num_within_FOV(catalogue: pd.DataFrame, ax: axes.Axes, title: str, num_vec: int=100000):
    '''
        Draw the probability distribution of the number of stars within FOV.
    Args:
        catalogue: the original catalogue
        ax: the axes to draw
        title: the title of the plot
        num_vec: the number of vectors to be generated
    '''
    # calculate the half of FOV diagonal distance
    R = sqrt((radians(FOV)**2)+(radians(FOV)**2))/2
    
    # generate random right ascension[-pi, pi] and declination[-pi/2, pi/2], method from http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09
    ras = np.random.uniform(-pi, pi, num_vec)
    des = np.arcsin(np.random.uniform(-1, 1, num_vec))
    
    # record the result: star_num -> sample_num
    table = {}
    for ra, de in zip(ras, des):
        # get the rough range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        # get the num of stars within FOV
        stars_within_FOV = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2) & (de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)].copy()
        
        # convert to celestial rectangular coordinate system
        stars_within_FOV['X1'] = np.cos(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
        stars_within_FOV['Y1'] = np.sin(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
        stars_within_FOV['Z1'] = np.sin(stars_within_FOV['DE'])
        
        # convert to star sensor coordinate system
        M = get_rotation_matrix(ra, de, 0)
        stars_within_FOV[['X2', 'Y2', 'Z2']] = stars_within_FOV[['X1', 'Y1', 'Z1']].dot(M)
        
        # convert to image coordinate system
        stars_within_FOV['X3'] = f*(stars_within_FOV['X2']/stars_within_FOV['Z2'])
        stars_within_FOV['Y3'] = f*(stars_within_FOV['Y2']/stars_within_FOV['Z2'])
        
        # exclude stars beyond range
        l = tan(radians(FOV/2))*f
        stars_within_FOV = stars_within_FOV[stars_within_FOV['X3'].between(-l, l) & stars_within_FOV['Y3'].between(-l, l)]

        star_num = len(stars_within_FOV)
        table[star_num] = table.get(star_num, 0) + 1
    
    # sort table items by key
    num_stars = sorted(table.keys())
    probability = [(table[num_star]*100.0)/num_vec for num_star in num_stars]
    print(num_stars, probability)

    # calculate average number of star in FOV
    avg = sum([num_star*table[num_star] for num_star in num_stars])/num_vec

    # calculate the standard deviation
    std = sqrt(sum([((num_star - avg)**2)*table[num_star] for num_star in num_stars])/num_vec)
    print('avg: ', avg, ' std: ', std)

    ax.plot(num_stars, probability)
    ax.vlines(avg, 0, 100, linestyles='dashed', colors='red')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.set_xlabel('Number of stars within FOV')
    ax.set_ylabel('Probability%')


def filter_catalogue(catalogue: pd.DataFrame, num_limit: int, mv_limit: float=6.0, agd_limit: float=0.5, num_sector: int=4, num_vec: int=100) -> pd.DataFrame:
    '''
        Filter navigation stars.
        Referred [1](http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09)
                 [2](https://www.cnki.com.cn/Article/CJFDTotal-HWYJ201501059.htm)
    Args:
        catalogue: the original catalogue
        num_limit: minimum number of stars in each circular area
        mv_limit: the magnitude limit
        agd_limit: the angular distance limit in degrees(if two stars' angular distance is less than this limit, choose the brighter one)
    Returns:    
        the filtered catalogue
    '''

    # calculate the half of FOV diagonal distance
    R = sqrt((radians(FOV)**2)+(radians(FOV)**2))/2

    # eliminate the stars with magnitude > mv_limit
    catalogue = catalogue[catalogue['Magnitude'] <= mv_limit].reset_index(drop=True)

    # convert to celestial rectangular coordinate system
    positions = pd.DataFrame()
    positions['X1'] = np.cos(catalogue['RA'])*np.cos(catalogue['DE'])
    positions['Y1'] = np.sin(catalogue['RA'])*np.cos(catalogue['DE'])
    positions['Z1'] = np.sin(catalogue['DE'])

    # calculate the angular distance between each pair of stars
    positions = positions.to_numpy()
    norms = np.linalg.norm(positions, axis=1)
    inner_products = positions @ positions.T
    # do rounding because some cos_theta are slightly greater than 1 or less than -1 as a result of precision problem
    cos_theta = np.round(inner_products/(norms*norms[:, None]), 6)
    angular_distance = np.arccos(cos_theta)

    # get small angular distance star pairs
    idx1, idx2 = np.nonzero(angular_distance < radians(agd_limit))
    # avoid idx1[i] == idx2[i]
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    star_pairs = set()
    darker_idxs = set()
    for i1, i2 in zip(idx1, idx2):
        # avoid duplicate star pairs because of the symmetry of the angular distance matrix
        if (i1, i2) in star_pairs or (i2, i1) in star_pairs:
            continue
        if i1 in darker_idxs or i2 in darker_idxs:
            continue
        star_pairs.add((i1, i2))
        mv1, mv2 = catalogue.loc[i1, 'Magnitude'], catalogue.loc[i2, 'Magnitude']
        if mv1 < mv2:
            darker_idxs.add(i1)
        else:
            darker_idxs.add(i2)
    darker_idxs = list(darker_idxs)
    # eliminate the darker stars from small angular distance star pairs
    catalogue = catalogue[~catalogue.index.isin(darker_idxs)]

    ras = np.arange(-pi, pi, 2*pi/num_vec)
    des = np.arcsin(np.arange(-1, 1, 2/num_vec))
    for ra in ras:
        for de in des:
            # calculate the range of ra & de
            ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
            de1, de2 = de - R, de + R
            stars_within_FOV = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2) & (de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)].copy()
        
            if len(stars_within_FOV) < num_limit:
                continue
            
            # convert to celestial rectangular coordinate system
            stars_within_FOV['X1'] = np.cos(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
            stars_within_FOV['Y1'] = np.sin(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
            stars_within_FOV['Z1'] = np.sin(stars_within_FOV['DE'])
            
            # convert to star sensor coordinate system
            M = get_rotation_matrix(ra, de, 0)
            stars_within_FOV[['X2', 'Y2', 'Z2']] = stars_within_FOV[['X1', 'Y1', 'Z1']].dot(M)
            
            # convert to image coordinate system
            stars_within_FOV['X3'] = f*(stars_within_FOV['X2']/stars_within_FOV['Z2'])
            stars_within_FOV['Y3'] = f*(stars_within_FOV['Y2']/stars_within_FOV['Z2'])
            
            # exclude stars beyond range
            l = tan(radians(FOV/2))*f
            stars_within_FOV = stars_within_FOV[stars_within_FOV['X3'].between(-l, l) & stars_within_FOV['Y3'].between(-l, l)]

            if len(stars_within_FOV) < num_limit:
                continue

            # claculate angle of each star
            stars_within_FOV['Angle'] = np.arctan2(stars_within_FOV['Y3'], stars_within_FOV['X3'])
            for i in range(num_sector):
                ag1, ag2 = i*2*pi/num_sector-pi, (i+1)*2*pi/num_sector-pi
                stars_within_sector = stars_within_FOV[stars_within_FOV['Angle'].between(ag1, ag2)]
                if len(stars_within_sector) < num_limit//num_sector+1:
                    continue
                # screen the top th1 brightest stars in each sector region
                tot = len(stars_within_sector)
                stars_within_sector = stars_within_sector.nlargest(tot-num_limit//num_sector-1, 'Magnitude')
                catalogue = catalogue[~catalogue.index.isin(stars_within_sector.index)]

    return catalogue


if __name__ == '__main__':
    file = 'catalogue/SAO6.0.csv'
    num_limit = 30
    mv_limit = 5.6
    filtered_file = f'catalogue/SAO{mv_limit}_{FOV}_{num_limit}.csv'

    df = pd.read_csv(file, usecols=["Star ID", "RA", "DE", "Magnitude"])
    if os.path.exists(filtered_file):
        filtered_df = pd.read_csv(filtered_file)
    else:
        filtered_df = filter_catalogue(df, num_limit, mv_limit).reset_index(drop=True)
        filtered_df.to_csv(filtered_file)

    fig1, axs1 = plt.subplots(2)
    draw_star_distribution(df, axs1[0], "Original")
    draw_star_distribution(filtered_df, axs1[1], "Filtered")

    fig2, axs2 = plt.subplots(2)
    draw_probability_versus_star_num_within_FOV(df, axs2[0], "Original")
    draw_probability_versus_star_num_within_FOV(filtered_df, axs2[1], "Filtered")

    plt.show()