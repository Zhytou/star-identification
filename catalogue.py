import os, struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.axes._axes as axes
from math import radians, sqrt, tan, sin, cos, pi


def draw_star_distribution(catalogue: pd.DataFrame, title: str = '', ax: axes.Axes = None):
    '''
        Draw the distribution of stars in the celestial sphere.
    Args:
        catalogue: the star catalogue
        title: the title of the plot
        ax: the axes to draw
    '''
    ras, des = np.degrees(catalogue['Ra']), np.degrees(catalogue['De'])
    if ax == None:
        plt.scatter(ras, des, s=1)
        plt.title(title)
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        plt.xlabel('RA')
        plt.ylabel('DE')
    else:
        ax.scatter(ras, des, s=1)
        ax.set_title(title)
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('RA')
        ax.set_ylabel('DE')    
    plt.show()


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


def draw_probability_versus_star_num_within_fov(catalogue: pd.DataFrame, ax: axes.Axes = None, title: str = '', fov: int=20, f: float=58e-3, num_vec: int=100000):
    '''
        Draw the probability distribution of the number of stars within fov.
    Args:
        catalogue: the original catalogue
        ax: the axes to draw
        title: the title of the plot
        num_vec: the number of vectors to be generated
    '''
    # calculate the half of fov diagonal distance
    R = sqrt((radians(fov)**2)+(radians(fov)**2))/2
    
    # generate random right ascension[-pi, pi] and declination[-pi/2, pi/2], method from http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09
    ras = np.random.uniform(0, 2*pi, num_vec)
    des = np.arcsin(np.random.uniform(-1, 1, num_vec))
    
    # record the result: star_num -> sample_num
    table = {}
    for ra, de in zip(ras, des):
        # get the rough range of ra & de
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        # get the num of stars within fov
        stars_within_fov = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2) & (de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)].copy()
        
        # convert to celestial rectangular coordinate system
        stars_within_fov['X1'] = np.cos(stars_within_fov['RA'])*np.cos(stars_within_fov['DE'])
        stars_within_fov['Y1'] = np.sin(stars_within_fov['RA'])*np.cos(stars_within_fov['DE'])
        stars_within_fov['Z1'] = np.sin(stars_within_fov['DE'])
        
        # convert to star sensor coordinate system
        M = get_rotation_matrix(ra, de, 0)
        stars_within_fov[['X2', 'Y2', 'Z2']] = stars_within_fov[['X1', 'Y1', 'Z1']].dot(M)
        
        # convert to image coordinate system
        stars_within_fov['X3'] = f*(stars_within_fov['X2']/stars_within_fov['Z2'])
        stars_within_fov['Y3'] = f*(stars_within_fov['Y2']/stars_within_fov['Z2'])
        
        # exclude stars beyond range
        l = tan(radians(fov/2))*f
        stars_within_fov = stars_within_fov[stars_within_fov['X3'].between(-l, l) & stars_within_fov['Y3'].between(-l, l)]

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
        plt.xlabel('Number of stars within fov')
        plt.ylabel('Probability%')
        plt.grid(True)
    else:
        ax.bar(num_stars, probability)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        ax.set_xlabel('Number of stars within fov')
        ax.set_ylabel('Probability%')
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
        

def filter_catalogue(catalogue: pd.DataFrame, num_limit: int, mag_limit: float=6.0, agd_limit: float=0.5, num_sector: int=4, fov: int=20, f: float=58e-3, num_vec: int=100, uniform: bool = True) -> pd.DataFrame:
    '''
        Filter navigation stars.
        Referred [1](http://www.opticsjournal.net/Articles/Abstract?aid=OJbf48ddeef697ba09)
                 [2](https://www.cnki.com.cn/Article/CJFDTotal-HWYJ201501059.htm)
    Args:
        catalogue: the original catalogue
        num_limit: minimum number of stars in each circular area
        mag_limit: the magnitude limit
        agd_limit: the angular distance limit in degrees(if two stars' angular distance is less than this limit, choose the brighter one)
        num_sector: the number of sectors
        fov: the field of view in degrees
        f: the focal length of the lens
        num_vec: the number of vectors to be generated
        uniform: whether to homogenize the stars in the celestial sphere
    Returns:    
        the filtered catalogue
    '''

    # calculate the half of fov diagonal distance
    R = sqrt((radians(fov)**2)+(radians(fov)**2))/2

    num_dark_star_excl = len(catalogue)

    # eliminate the stars with magnitude > mag_limit
    catalogue = catalogue[catalogue['Magnitude'] <= mag_limit].reset_index(drop=True)

    num_dark_star_excl -= len(catalogue)
    print('the number of dark stars excluded: ', num_dark_star_excl)

    # calculate the angular distance between each pair of stars
    catalogue['X'] = np.cos(catalogue['Ra'])*np.cos(catalogue['De'])
    catalogue['Y'] = np.sin(catalogue['Ra'])*np.cos(catalogue['De'])
    catalogue['Z'] = np.sin(catalogue['De'])

    positions = catalogue[['X', 'Y', 'Z']].to_numpy()
    angular_distance = np.arccos(np.dot(positions, positions.T))

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

    print('the number of darker star pairs excluded: ', len(darker_idxs))

    if not uniform:
        # remove X,Y,Z columns
        catalogue = catalogue[['Star ID', 'Ra', 'De', 'Magnitude']]
        return catalogue

    num_uniform_star_excl = len(catalogue)
    ras = np.arange(0, 2*pi, 2*pi/num_vec)
    des = np.arcsin(np.arange(-1, 1, 2/num_vec))
    for ra in ras:
        for de in des:
            # fov restriction

            if True:
                sensor = np.array([cos(ra)*cos(de), sin(ra)*cos(de), sin(de)]).transpose()
                # fov restriction
                angle = catalogue[['X', 'Y', 'Z']].dot(sensor)
                stars_within_fov = catalogue[angle <= cos(radians(fov/2))].copy()
            else:
                # get the rough range of ra & de
                ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
                de1, de2 = (de - R), (de + R)
                # get the num of stars within fov
                stars_within_fov = catalogue[(ra1 <= catalogue['Ra']) & (catalogue['Ra'] <= ra2) & (de1 <= catalogue['De']) & (catalogue['De'] <= de2)].copy()
            
            if len(stars_within_fov) < num_limit:
                continue
            
            # # rename columns
            stars_within_fov = stars_within_fov.rename(columns={'X': 'X1', 'Y': 'Y1', 'Z': 'Z1'})
            
            # convert to star sensor coordinate system
            M = get_rotation_matrix(ra, de, 0)
            stars_within_fov[['X2', 'Y2', 'Z2']] = stars_within_fov[['X1', 'Y1', 'Z1']].dot(M)
            
            # convert to image coordinate system
            stars_within_fov['X3'] = f*(stars_within_fov['X2']/stars_within_fov['Z2'])
            stars_within_fov['Y3'] = f*(stars_within_fov['Y2']/stars_within_fov['Z2'])
            
            # exclude stars beyond range
            l = tan(radians(fov/2))*f
            stars_within_fov = stars_within_fov[stars_within_fov['X3'].between(-l, l) & stars_within_fov['Y3'].between(-l, l)]

            if len(stars_within_fov) < num_limit:
                continue

            # claculate angle of each star
            stars_within_fov['Angle'] = np.arctan2(stars_within_fov['Y3'], stars_within_fov['X3'])
            for i in range(num_sector):
                ag1, ag2 = i*2*pi/num_sector-pi, (i+1)*2*pi/num_sector-pi
                stars_within_sector = stars_within_fov[stars_within_fov['Angle'].between(ag1, ag2)]
                if len(stars_within_sector) < num_limit//num_sector+1:
                    continue
                # screen the top th1 brightest stars in each sector region
                tot = len(stars_within_sector)
                stars_within_sector = stars_within_sector.nlargest(tot-num_limit//num_sector-1, 'Magnitude')
                catalogue = catalogue[~catalogue.index.isin(stars_within_sector.index)]

    num_uniform_star_excl -= len(catalogue)
    print('the number of uniform stars excluded: ', num_uniform_star_excl)

    # remove X,Y,Z columns
    catalogue = catalogue[['Star ID', 'Ra', 'De', 'Magnitude']]

    return catalogue


if __name__ == '__main__':
    # filter parameters
    fov = 15
    f = 58e-3
    num_limit = 20
    mag_limit = 5.5
    agd_limit = 0.2

    raw_file = 'raw_catalogue/sao_j2000.dat'
    parsed_file = 'catalogue/sao.csv'
    limit_parsed_file = 'catalogue/sao7.0.csv'
    filtered_file = f'catalogue/sao{mag_limit}_d{agd_limit}.csv' # process double star and magnitude threshold
    uniform_filtered_file = f'catalogue/sao{mag_limit}_d{agd_limit}_{fov}_{num_limit}.csv'

    df = parse_heasarc_sao(raw_file, parsed_file)
    df = df[df['Magnitude'] <= 7.0].reset_index(drop=True)
    if not os.path.exists(limit_parsed_file):
        df.to_csv(limit_parsed_file)

    if os.path.exists(filtered_file):
        f_df = pd.read_csv(filtered_file)
    else:
        f_df = filter_catalogue(df, num_limit, mag_limit, agd_limit, fov=fov, f=f, uniform=False).reset_index(drop=True)
        f_df.to_csv(filtered_file)

    draw_star_distribution(f_df)
    # draw_probability_versus_star_num_within_fov(f_df, fov=fov, f=f, num_vec=3000)
    
    # if os.path.exists(uniform_filtered_file):
    #     uf_df = pd.read_csv(uniform_filtered_file)
    # else:
    #     uf_df = filter_catalogue(df, num_limit, mag_limit, agd_limit, fov=fov, f=f).reset_index(drop=True)
    #     uf_df.to_csv(uniform_filtered_file)
    
    # draw_star_distribution(uf_df)
    # draw_probability_versus_star_num_within_fov(uf_df, fov=fov, f=f, num_vec=3000)

    # fig1, axs1 = plt.subplots(1, 2)
    # draw_star_distribution(df, axs1[0], "Original")
    # draw_star_distribution(uf_df, axs1[1], "Filtered")

    # fig2, axs2 = plt.subplots(1, 2)
    # draw_probability_versus_star_num_within_fov(df, axs2[0], "Original", fov=fov, f=f, num_vec=1000)
    # draw_probability_versus_star_num_within_fov(uf_df, axs2[1], "Filtered", fov=fov, f=f, num_vec=1000)
