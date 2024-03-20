import os
from math import radians, degrees, sin, cos, tan, sqrt, exp
import numpy as np
import pandas as pd


# region of interest for point spread function
ROI = 2

# star sensor pixel num
w = 2048
h = 2048

# star sensor foucs in metres
f = 58e-3

# field of view angle in degrees
FOV = 20

# camera total length and width in metres
mtot = 2*tan(radians(FOV/2))*f

# pixel num per length
xpixel = w/mtot
ypixel = h/mtot

# star catalogue path
catalogue_path = 'catalogue/filtered_below_5.6_SAO.csv'


def create_star_image(ra: float, de: float, roll: float, white_noise_std: float = 10, pos_noise_std: int = 0, mv_noise_std: float = 0, num_false_star: int = 0) -> tuple[np.ndarray, list]:
    """
        Create a star image from the given right ascension, declination and roll angle.
    Args:
        ra: right ascension in degrees
        de: declination in degrees
        roll: roll in degrees
        white_noise_std: the standard deviation of white noise
        pos_noise_std: the standard deviation of positional noise
        mv_noise_std: the standard deviation of maginatitude noise
        num_false_star: the number of false stars
    Returns:
        img: the simulated star image
        stars: stars drawn in the image
    """

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

    def draw_star(x: int, y: int, magnitude: float, img: np.ndarray) -> np.ndarray:
        """
            Draw star at yth row and xth column in the image.
        Args:
            x: the x coordinate in pixel (starting from left to right)
            y: the y coordinate in pixel (starting from top to bottom)
            magnitude: the stellar magnitude
            img: background image
        Returns:
            img: the image with the star drawn
        """
        # stellar magnitude to intensity
        H = 30/(2.51**(magnitude-6))

        for u in range(x-ROI, x+ROI+1):
            if u < 0 or u >= len(img[0]):
                continue
            for v in range(y-ROI, y+ROI+1):
                if v < 0 or v >= len(img):
                    continue
                if (u-x)**2+(v-y)**2 > ROI**2:
                    continue
                raw_intensity = int(H*exp(-((u-x)**2+(v-y)**2)/(2*ROI**2)))
                img[v ,u] = raw_intensity

        return img

    def add_white_noise(img: np.ndarray) -> np.ndarray:
        """
            Adds white noise to an image.
        Args:
            img: the image to put noise on
        Returns:
            noised_img: the image with white noise
        """
        noise = np.random.normal(0, white_noise_std, img.shape)
        # make sure no pixel value is less than 0 or greater than 255
        noised_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noised_img

    def add_false_stars(img: np.ndarray, num: int) -> tuple[np.ndarray, list]:
        '''
            Add false stars to the image.
        '''
        false_stars = []
        for _ in range(num):
            x = np.random.randint(ROI, w-ROI)
            y = np.random.randint(ROI, h-ROI)
            img = draw_star(x, y, 5.7, img)
            false_stars.append([-1, (y, x), 5.7])
        return img, false_stars

    # right ascension, declination and roll in radians
    ra = radians(float(ra))
    de = radians(float(de))
    roll = radians(float(roll))

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll)

    # read star catalogue
    col_list = ["Star ID", "RA", "DE", "Magnitude"]
    star_catalogue = pd.read_csv(catalogue_path, usecols=col_list)

    # search for image-able stars
    R = sqrt((radians(FOV)**2)+(radians(FOV)**2))/2
    ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
    de1, de2 = (de - R), (de + R)
    assert ra1 < ra2 and de1 < de2

    stars_within_FOV = star_catalogue[(ra1 <= star_catalogue['RA']) & (star_catalogue['RA'] <= ra2) & 
                                      (de1 <= star_catalogue['DE']) & (star_catalogue['DE'] <= de2)].copy()

    # convert to celestial rectangular coordinate system
    stars_within_FOV['X1'] = np.cos(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
    stars_within_FOV['Y1'] = np.sin(stars_within_FOV['RA'])*np.cos(stars_within_FOV['DE'])
    stars_within_FOV['Z1'] = np.sin(stars_within_FOV['DE'])

    # convert to star sensor coordinate system
    stars_within_FOV[['X2', 'Y2', 'Z2']] = stars_within_FOV[['X1', 'Y1', 'Z1']].dot(M)
    
    # convert to image coordinate system
    stars_within_FOV['X3'] = f*(stars_within_FOV['X2']/stars_within_FOV['Z2'])
    stars_within_FOV['Y3'] = f*(stars_within_FOV['Y2']/stars_within_FOV['Z2'])

    # convert to pixel coordinate system
    stars_within_FOV['X4'] = np.round(w/2+stars_within_FOV['X3']*xpixel).astype(int)
    stars_within_FOV['Y4'] = np.round(h/2-stars_within_FOV['Y3']*ypixel).astype(int)

    # exclude stars beyond range
    stars_within_FOV = stars_within_FOV[stars_within_FOV['X4'].between(ROI, w-ROI) & stars_within_FOV['Y4'].between(ROI, h-ROI)]

    star_positions = list(zip(stars_within_FOV['X4'], stars_within_FOV['Y4']))
    star_magnitudes = list(stars_within_FOV['Magnitude'])
    star_ids = list(stars_within_FOV['Star ID'])
    
    # initialize image & star info list to return
    img = np.zeros((h,w))
    stars = []
    for i in range(len(star_magnitudes)):
        # draw imagable star at (row, col)
        col, row = star_positions[i]
        if pos_noise_std > 0:
            col += np.random.randint(0, pos_noise_std+1)
            row += np.random.randint(0, pos_noise_std+1)
        if mv_noise_std > 0:
            star_magnitudes[i] += np.random.normal(0, mv_noise_std)
        img = draw_star(col, row, star_magnitudes[i], img)
        stars.append([star_ids[i], (row, col), star_magnitudes[i]])

    # add false stars with random magitudes at random positions
    if num_false_star > 0:
        img, false_stars = add_false_stars(img, num_false_star)
        stars.extend(false_stars)

    # add white noise
    img = add_white_noise(img)

    return img, stars


if __name__ == '__main__':
    # simulation accuracy check
    col_list = ["Star ID", "RA", "DE", "Magnitude"]
    df = pd.read_csv(catalogue_path, usecols=col_list)
    for i in range(len(df)):
        ra, de = df.loc[i, 'RA'], df.loc[i, 'DE']
        img, stars = create_star_image(degrees(ra), degrees(de), 0)
        star_table = dict(map(lambda x: (x[1], x[0]), stars))
        # when using the ra & de in star catalogue, one star must be placed in the center of image
        if star_table.get((w/2, h/2), -1) == -1:
            print(i, 'ra:', round(degrees(ra), 2), 'de:', round(degrees(de), 2), 'f:',f)
            break
