import os, struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.axes._axes as axes
from math import radians, sqrt, tan, sin, cos, pi


# Chinese font setting
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_distri(cata: pd.DataFrame, title: str = '', ax: axes.Axes = None):
    '''
        Draw the distribution of stars in the celestial sphere.
    Args:
        cata: the star catalogue
        title: the title of the plot
        ax: the axes to draw
    '''
    ras, des = np.degrees(cata['Ra']), np.degrees(cata['De'])
    if ax == None:
        plt.scatter(ras, des, s=1)
        plt.title(title)
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.xlabel('赤经(°)')
        plt.ylabel('赤纬(°)')
    else:
        ax.scatter(ras, des, s=1)
        ax.set_title(title)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('赤经(°)')
        ax.set_ylabel('赤纬(°)')    
    plt.show()


def draw_prob(cata: pd.DataFrame, ax: axes.Axes = None, title: str = '', fov: int=20, num_vec: int=10000):
    '''
        Draw the probability distribution of the number of stars within fov.
    Args:
        cata: the original catalogue
        ax: the axes to draw
        title: the title of the plot
        fov: the field of view in degrees
        num_vec: the number of vectors to be generated
    '''
    
    # generate random right ascension[-pi, pi] and declination[-pi/2, pi/2], method from http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09
    ras = np.random.uniform(0, 2*pi, num_vec)
    des = np.arcsin(np.random.uniform(-1, 1, num_vec))
    vecs = np.array([np.cos(ras)*np.cos(des), np.sin(ras)*np.cos(des), np.sin(des)]).transpose()

    # calculate the celestial cartesian coordinates of stars
    cata['X'] = np.cos(cata['Ra'])*np.cos(cata['De'])
    cata['Y'] = np.sin(cata['Ra'])*np.cos(cata['De'])
    cata['Z'] = np.sin(cata['De'])

    # record the result: star_num -> sample_num
    table = {}
    for vec in vecs:
        angle = cata[['X', 'Y', 'Z']].dot(vec)
        stars_within_fov = cata[angle >= cos(radians(fov/2))]

        star_num = len(stars_within_fov)
        table[star_num] = table.get(star_num, 0) + 1
    
    # sort table items by key
    num_stars = sorted(table.keys())
    probability = [(table[num_star]*100.0)/num_vec for num_star in num_stars]
    # print(num_stars, probability)

    # calculate average number of star in fov
    avg = sum([num_star*table[num_star] for num_star in num_stars])/num_vec

    # calculate the standard deviation
    std = sqrt(sum([((num_star - avg)**2)*table[num_star] for num_star in num_stars])/num_vec)
    print('avg: ', avg, ' std: ', std)

    if ax == None:
        plt.bar(num_stars, probability)
        plt.title(title)
        plt.xlim(0, 100)
        plt.ylim(0, 20)
        plt.xlabel('视场内恒星数量')
        plt.ylabel('概率(%)')
        plt.grid(True)
    else:
        ax.bar(num_stars, probability)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        ax.set_xticks(np.arange(0, 100, 5))
        ax.set_xlabel('视场内恒星数量')
        ax.set_ylabel('概率(%)')
        ax.grid(True)
    
    plt.show()


def parse_tdc_sao(file_path: str):
    '''
        Parse SAO star catalogue's raw data from Harvard University Telescope Data Center http://tdc-www.harvard.edu/catalogs/sao.html. 
        
        The first 28 bytes of the file is its header, containing the following information:
            Integer*4 STAR0=0	Subtract from star number to get sequence number
            Integer*4 STAR1=1	First star number in file
            Integer*4 STARN=258996	Number of stars in file
            Integer*4 STNUM=1	0 if no star i.d. numbers are present
			                    1 if star i.d. numbers are in catalog file
			                    2 if star i.d. numbers are  in file
            Logical*4 MPROP=t	True if proper motion is included
			                    False if no proper motion is included
            Integer*4 NMAG=1	Number of magnitudes present
            Integer*4 NBENT=32	Number of bytes per star entry

        Each entry in the raw data contains 32 bytes with the following information:
        
            Real*4 XNO		Catalog number of star
            Real*8 SRA0		B1950 Right Ascension (radians)
            Real*8 SDEC0		B1950 Declination (radians)
            Character*2 IS	Spectral type (2 characters)
            Integer*2 MAG		V Magnitude * 100
            Real*4 XRPM		R.A. proper motion (radians per year)
            Real*4 XDPM		Dec. proper motion (radians per year)

    Args:
        path: the path to the raw data file

    Returns:
        dataframe of the parsed data
    '''
    file_name = file_path.split('/')[-1].split('.')[0]
    endian = file_name.split('_')[-1]
    
    flag = ''
    if endian == 'bigendian':
        flag = '>'
    elif endian == 'smallendian':
        flag = '<'
    else:
        print('wrong file', file_name, endian)
        return

    with open(file_path, 'rb') as file:
        # parse header
        header_bytes = file.read(28)
        star0, star1, starn, stnum, mprop, nmag, nbent = struct.unpack(f'{flag}i i i i i i i', header_bytes)
        print('header', star0, star1, starn, stnum, mprop, nmag, nbent)

        # parse entries
        entries = []
        while True:
            entry_bytes = file.read(nbent)
            # no star id in entry(no xno)
            if stnum == 0:
                xno = len(entries)
                sra0, sdec0, _, mag, _, _ = struct.unpack(f'{flag}d d h H f f', entry_bytes)
            else:
                xno, sra0, sdec0, _, mag, _, _ = struct.unpack(f'{flag}f d d h H f f', entry_bytes)
            
            print(xno, sra0, sdec0, mag)
            entries.append([xno, sra0, sdec0, mag])
            # if len(entries) > 10:
            #     break
        
    df = pd.DataFrame(entries, columns=["Star ID", "RA", "DE", "Magnitude"])
    print(df)
    return df
            

def parse_heasarc_sao(file_path: str, file_storage_path: str):
    '''
        Parse SAO star catalogue's raw data from HEASARC https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/sao.html.
    
    Args:
        file_path: the path to the raw data file
        file_storage_path: the path to store the parsed data
    Returns:
        dataframe of the parsed data
    '''

    # if os.path.exists(file_storage_path):
    #     return pd.read_csv(file_storage_path, usecols=["Star ID", "RA", "DE", "Magnitude"])

    with open(file_path, 'r') as file:
        data_list = []

        lines = file.readlines()
        for line in lines:            
            cata_id = int(line[0:6])
            ra = float(line[183:193])
            de = float(line[193:204])
            # photographic magnitude
            pmag = float(line[76:80])
            # visual magnitude
            vmag = float(line[80:84])

            # print(len(line), cata_id, ra, de, mag)
            data_list.append([cata_id, ra, de, vmag])
    
    # convert to dataframe
    df = pd.DataFrame(data_list, columns=["Star ID", "Ra", "De", "Magnitude"])

    # calculate the celestial cartesian coordinates of stars
    # df['X'] = np.cos(df['Ra'])*np.cos(df['De'])
    # df['Y'] = np.sin(df['Ra'])*np.cos(df['De'])
    # df['Z'] = np.sin(df['De'])
    df.to_csv(file_storage_path)

    return df
        

def filter_catalogue(cata: pd.DataFrame, num_limit: int, mag_limit: float=6.0, agd_limit: float=0.5, fov: int=20, num_vec: int=100, uniform: bool = True) -> pd.DataFrame:
    '''
        Filter catalogue to get navigation stars.
        Referred [1](http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09)
                 [2](https://www.cnki.com.cn/Article/CJFDTotal-HWYJ201501059.htm)
    Args:
        cata: the original catalogue
        num_limit: the maximum number of guide stars in each fov
        mag_limit: the magnitude limit
        agd_limit: the angular distance limit in degrees(if two stars' angular distance is less than this limit, choose the brighter one)
        num_sec: the number of sectors
        fov: the field of view in degrees
        num_vec: the number of vectors to be generated
        uniform: whether to homogenize the stars in the celestial sphere
    Returns:    
        the filtered catalogue(guide star catalogue)
    '''

    num_dark_excl = len(cata)

    # eliminate the stars with magnitude > mag_limit
    cata = cata[cata['Magnitude'] <= mag_limit].reset_index(drop=True)

    num_dark_excl -= len(cata)
    print('the number of dark stars excluded: ', num_dark_excl)

    # calculate the angular distance between each pair of stars
    cata['X'] = np.cos(cata['Ra'])*np.cos(cata['De'])
    cata['Y'] = np.sin(cata['Ra'])*np.cos(cata['De'])
    cata['Z'] = np.sin(cata['De'])

    # get small angular distance star pairs
    pos = cata[['X', 'Y', 'Z']].to_numpy()
    agd = np.dot(pos, pos.T)
    idxs1, idxs2 = np.nonzero(agd > np.cos(radians(agd_limit)))

    # get unique star pairs
    mask = idxs1 < idxs2
    idxs1, idxs2 = idxs1[mask], idxs2[mask]

    # get the darker star index by condition
    mags1, mags2 = cata.loc[idxs1, 'Magnitude'].to_numpy(), cata.loc[idxs2, 'Magnitude'].to_numpy()
    darker_idxs = np.where(mags1 > mags2, idxs1, idxs2)
    print()
    # eliminate the darker stars from small angular distance star pairs
    cata = cata[~cata.index.isin(darker_idxs)]

    print('the number of darker star pairs excluded: ', len(darker_idxs))

    if not uniform:
        # remove X,Y,Z columns
        cata = cata[['Star ID', 'Ra', 'De', 'Magnitude']]
        return cata

    num_unif_excl = len(cata)
    # generate random uniform spherical vectors
    ras = np.arange(0, 2*pi, 2*pi/num_vec)
    des = np.arcsin(np.arange(-1, 1, 2/num_vec))
    ra_grid, de_grid = np.meshgrid(ras, des)
    ras, des = ra_grid.flatten(), de_grid.flatten()

    sensor_x = np.cos(ras)*np.cos(des)
    sensor_y = np.sin(ras)*np.cos(des)
    sensor_z = np.sin(des)
    sensors = np.array([sensor_x, sensor_y, sensor_z]).transpose()

    for ra, de, sensor in zip(ras, des, sensors):
        # fov restriction
        if True:
            # fov restriction
            angle = cata[['X', 'Y', 'Z']].dot(sensor)
            stars_within_fov = cata[angle >= cos(radians(fov/2))].copy()
        else:
            # calculate the half of fov diagonal distance
            R = sqrt((radians(fov)**2)+(radians(fov)**2))/2
            # get the rough range of ra & de
            ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
            de1, de2 = (de - R), (de + R)
            # get the num of stars within fov
            stars_within_fov = cata[(ra1 <= cata['Ra']) & (cata['Ra'] <= ra2) & (de1 <= cata['De']) & (cata['De'] <= de2)].copy()
        
        if len(stars_within_fov) <= num_limit:
            continue
        
        tot = len(stars_within_fov)
        stars_within_fov = stars_within_fov.nlargest(tot-num_limit, 'Magnitude')
        # remain top max_limit brightest stars
        cata = cata[~cata.index.isin(stars_within_fov.index)]

        # claculate angle of each star
        # stars_within_fov['Angle'] = np.arctan2(stars_within_fov['Y'], stars_within_fov['X'])
        # for i in range(num_sec):
        #     ag1, ag2 = i*2*pi/num_sec-pi, (i+1)*2*pi/num_sec-pi
        #     stars_within_sector = stars_within_fov[stars_within_fov['Angle'].between(ag1, ag2, inclusive='left')]
        #     if len(stars_within_sector) < max_limit//num_sec+1:
        #         continue
        #     # screen the top th1 brightest stars in each sector region
        #     tot = len(stars_within_sector)
        #     stars_within_sector = stars_within_sector.nlargest(tot-max_limit//num_sec-1, 'Magnitude')
        #     cata = cata[~cata.index.isin(stars_within_sector.index)]

    num_unif_excl -= len(cata)
    print('the number of uniform stars excluded: ', num_unif_excl)

    # remove X,Y,Z columns
    cata = cata[['Star ID', 'Ra', 'De', 'Magnitude']]

    return cata


def remove_guide_star(gcata: pd.DataFrame, cata: pd.DataFrame, num_limit: int=3, mag_limit: float=6.0, fov: int=20):
    '''
        Remove the guide star with less than num_limit stars.
    '''
    # eliminate the stars with magnitude > mag_limit
    cata = cata[cata['Magnitude'] <= mag_limit].reset_index(drop=True)

    # get the celestial cartesian coordinates of stars
    cata['X'] = np.cos(cata['Ra'])*np.cos(cata['De'])
    cata['Y'] = np.sin(cata['Ra'])*np.cos(cata['De'])
    cata['Z'] = np.sin(cata['De'])

    # get the celestial cartesian coordinates of guide stars
    gcata['X'] = np.cos(gcata['Ra'])*np.cos(gcata['De'])
    gcata['Y'] = np.sin(gcata['Ra'])*np.cos(gcata['De'])
    gcata['Z'] = np.sin(gcata['De'])

    # count the number of stars in each fov
    gpos, pos = gcata[['X', 'Y', 'Z']].to_numpy(), cata[['X', 'Y', 'Z']].to_numpy()
    agd = np.dot(gpos, pos.T)
    num_stars_in_fov = np.sum(agd > np.cos(radians(fov / 2)), axis=1)

    # get the indices of stars with less than num_limit stars
    idxs = np.where(num_stars_in_fov < num_limit)[0]

    # remove the stars with less than num_limit stars
    gcata = gcata[~gcata.index.isin(idxs)]

    print('the number of guide stars excluded: ', len(idxs))

    # remove X,Y,Z columns
    gcata = gcata[['Star ID', 'Ra', 'De', 'Magnitude']]

    return gcata


if __name__ == '__main__':
    # filter parameters
    fov = 12
    max_limit = 15
    min_limit = 8
    mag_limit = 6.0
    agd_limit = 0.03

    raw_file = 'raw_catalogue/sao_j2000.dat'
    sao_file = 'catalogue/sao.csv'
    uni_file = f'catalogue/sao{mag_limit}_d{agd_limit}_{fov}_{max_limit}.csv'
    rni_file = f'catalogue/sao{mag_limit}_d{agd_limit}_{fov}_{max_limit}_r{min_limit}.csv'

    df = parse_heasarc_sao(raw_file, sao_file)
    df = df[df['Magnitude'] < 7.0]
    draw_distri(df)
    f_df = filter_catalogue(df, max_limit, mag_limit, agd_limit, fov=fov, uniform=False).reset_index(drop=True)
    draw_distri(f_df)
    # draw_prob(f_df, fov=fov, num_vec=3000)
    
    if os.path.exists(uni_file):
        uf_df = pd.read_csv(uni_file)
    else:
        uf_df = filter_catalogue(df, max_limit, mag_limit, agd_limit, fov=fov, num_vec=360).reset_index(drop=True)
        uf_df.to_csv(uni_file)
    
    draw_distri(uf_df)
    draw_prob(uf_df, fov=fov, num_vec=3000)