import os
from math import radians, sin, cos, tan, sqrt, exp
import numpy as np
import pandas as pd

# region of interest for point spread function
roi = 3

# star sensor pixel num
w = 512
h = 512

# star sensor foucs in metres
f = 58e-3

# field of view angle in degrees
fov = 15

# camera total length and width in metres
mtot = 2*tan(radians(fov/2))*f

# camera magnitude sensitivity limitation
mv_limit = 6.0

# pixel num per length
xpixel = w/mtot
ypixel = h/mtot

# star catalogue path
catalogue_path = f'catalogue/sao6.0_d0.2.csv'

# read star catalogue
col_list = ["Star ID", "RA", "DE", "Magnitude"]
catalogue = pd.read_csv(catalogue_path, usecols=col_list)

# define simulation config
sim_cfg = f"{os.path.basename(catalogue_path).rsplit('.', 1)[0]}_{w}x{h}_{fov}x{fov}_{mv_limit}"


def cal_avg_star_num_within_fov(mv_limit: float=mv_limit, fov: float=fov) -> float:
    '''
        Calculate the average number of stars within the field of view.
    '''
    N = 6.57 * np.exp(1.08*mv_limit) * (1 - cos(radians(fov)/2)) / 2
    return N


def add_stary_light_noise(img: np.ndarray, rc: int, cc: int, std: int, A: int) -> np.ndarray:
    '''
        Add stary light noise to the image.
    Args:
        img: the image to add stary light noise
    Returns:
        noised_img: the image with stary light noise
    '''
    h, w = img.shape

    row = np.arange(w).reshape(-1, 1) - rc
    col = np.arange(h).reshape(1, -1) - cc

    stary = 200 * np.exp(-(row**2 + col**2) / (2 * std**2))
    noised_img = np.clip(img + stary, 0, 255).astype(np.uint8)

    return noised_img


def add_gaussian_and_pepper_noise(img: np.ndarray, sigma_g: float, prob_p: float) -> np.ndarray:
    """
        Adds white noise to an image.
    Args:
        img: the image to put noise on
    Returns:
        noised_img: the image with white noise
    """
    # normalize image
    img = img / 255.0

    # add pepper noise
    num_pepper = int(prob_p * img.size)
    for _ in range(num_pepper):
        x, y = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
        # if img[x, y] != 0:
        #     continue
        if np.random.rand() > 0.5:
            img[x, y] = 0
        else:
            img[x, y] = 1.0

    # add gaussian noise
    noise = np.random.normal(0, sigma_g, img.shape)
    noised_img = np.clip(img + noise, 0, 1.0)

    # denormalize image
    noised_img = (noised_img * 255).astype(np.uint8)

    return noised_img


def get_stellar_intensity(magnitude: float) -> float:
    """
        Get the stellar intensity from the stellar magnitude.
    Args:
        magnitude: the stellar magnitude
    Returns:
        H: the stellar intensity
    """
    # stellar magnitude to intensity
    H = 101 * 2.512 ** (6 - magnitude)
    return H


def draw_star(position: tuple[float, float], magnitude: float, img: np.ndarray, sigma: float=1.5) -> np.ndarray:
    """
        Draw star at position[0](row) and position[1](column) in the image.
    Args:
        position: (starting from top to bottom, starting from left to right)
        magnitude: the stellar magnitude
        img: background image
        sigma: the standard deviation of the point spread function
    Returns:
        img: the image with the star drawn
    """
    H = get_stellar_intensity(magnitude)

    x, y = position
    top, bottom = int(max(0, x-roi)), int(min(h, x+roi+1))
    left, right = int(max(0, y-roi)), int(min(w, y+roi+1))

    # print(x, y, top, bottom, left, right)
    for u in range(top, bottom):
        for v in range(left, right):
            # msaa
            intensity = 0
            for (du, dv) in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]: 
                dd = (u+du-x)**2+(v+dv-y)**2
                if dd > roi**2:
                    continue
                intensity += H*exp(-dd/(2*sigma**2))
            img[u ,v] = intensity // 4
    return img


def create_star_image(ra: float, de: float, roll: float, sigma_g: float = 0.0, prob_p: float = 0.0, pos_noise_std: float = 0, mv_noise_std: float = 0, ratio_false_star: int = 0, pure_point: bool = False, simulate_test: bool = False, background: float=np.inf) -> tuple[np.ndarray, list]:
    """
        Create a star image from the given right ascension, declination and roll angle.
    Args:
        ra: right ascension in radians
        de: declination in radians
        roll: roll in radians
        sigma_g: the nomalized standard deviation of gaussian noise
        prob_p: the probability of pepper noise
        pos_noise_std: the standard deviation of positional noise
        mv_noise_std: the standard deviation of maginatitude noise
        ratio_false_star: the ratio of false stars
        pure_point: no need to draw the star image, just pure point data(for generate.py)
        background: the background intensity
    Returns:
        img: the simulated star image
        stars: stars drawn in the image
        stars_within_fov: stars within the field of view(dataframe), if simulate_test is True
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

    def add_false_stars(img: np.ndarray, num: int, pos: np.array, min_d: int=4*roi) -> tuple[np.ndarray, list]:
        '''
            Add false stars to the image.
        Args:
            img: the image to add false stars
            num: the number of false stars
            pos: the positions of true stars
            min_d: the minimum distance between false stars and true stars
        '''
        false_stars = []
        while len(false_stars) < num:
            x = np.random.randint(roi, w-roi)
            y = np.random.randint(roi, h-roi)
            if len(pos) > 0:
                ds = np.linalg.norm(pos-(x, y), axis=1)
                if ds.min() < min_d:
                    continue
            mv = 5.0 + np.random.rand()
            img = draw_star(x, y, mv, img)
            false_stars.append([-1, (y, x), mv])
        return img, false_stars

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll)

    # search for image-able stars
    R = sqrt((radians(fov)**2)+(radians(fov)**2))/2
    ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
    de1, de2 = (de - R), (de + R)
    assert ra1 < ra2 and de1 < de2

    stars_within_fov = catalogue[(ra1 <= catalogue['RA']) & (catalogue['RA'] <= ra2) & 
                                (de1 <= catalogue['DE']) & (catalogue['DE'] <= de2)].copy()

    # convert to celestial cartesian coordinate system
    stars_within_fov['X1'] = np.cos(stars_within_fov['RA'])*np.cos(stars_within_fov['DE'])
    stars_within_fov['Y1'] = np.sin(stars_within_fov['RA'])*np.cos(stars_within_fov['DE'])
    stars_within_fov['Z1'] = np.sin(stars_within_fov['DE'])

    # convert to star sensor coordinate system
    stars_within_fov[['X2', 'Y2', 'Z2']] = stars_within_fov[['X1', 'Y1', 'Z1']].dot(M)
    
    # convert to image coordinate system
    stars_within_fov['X3'] = f*(stars_within_fov['X2']/stars_within_fov['Z2'])
    stars_within_fov['Y3'] = f*(stars_within_fov['Y2']/stars_within_fov['Z2'])

    # convert to pixel coordinate system
    stars_within_fov['X4'] = w/2+stars_within_fov['X3']*xpixel
    stars_within_fov['Y4'] = h/2+stars_within_fov['Y3']*ypixel
    
    # add positional noise if needed
    if pos_noise_std > 0:
        stars_within_fov['X4'] += np.random.normal(0, pos_noise_std, size=len(stars_within_fov['X4']))
        stars_within_fov['Y4'] += np.random.normal(0, pos_noise_std, size=len(stars_within_fov['Y4']))
    
    # exclude stars beyond range
    stars_within_fov = stars_within_fov[stars_within_fov['X4'].between(roi, w-roi) & stars_within_fov['Y4'].between(roi, h-roi)]

    # add magnitude noise if needed
    if mv_noise_std > 0:
        stars_within_fov['Magnitude'] += np.random.normal(0, mv_noise_std, size=len(stars_within_fov['Magnitude']))

    # exclude stars too dark to identify
    stars_within_fov = stars_within_fov[stars_within_fov['Magnitude'] <= mv_limit]

    star_positions = list(zip(stars_within_fov['Y4'], stars_within_fov['X4']))
    star_magnitudes = list(stars_within_fov['Magnitude'])
    star_ids = list(stars_within_fov['Star ID'])
    
    # background intensity
    if background == np.inf:
        img = np.zeros((h,w))
    else:
        img = get_stellar_intensity(background) * np.ones((h,w))

    # initialize star info list to return
    stars = []
    for i in range(len(star_magnitudes)):
        # draw imagable star at (Y4, X4) (Y4 is row number, X4 is column number)
        star_positions[i] = round(star_positions[i][0], 3), round(star_positions[i][1], 3)
        if not pure_point:
            img = draw_star(star_positions[i], star_magnitudes[i], img)
        stars.append([star_ids[i], star_positions[i], star_magnitudes[i]])

    # add false stars with random magitudes at random positions
    if ratio_false_star > 0:
        img, false_stars = add_false_stars(img, max(1, int(ratio_false_star*len(star_ids))), np.array(star_positions))
        stars.extend(false_stars)

    # add noise
    img = add_gaussian_and_pepper_noise(img, sigma_g, prob_p)

    if simulate_test:
        stars_within_fov = stars_within_fov.reset_index(drop=True)
        return img, stars_within_fov
    
    return img, stars
