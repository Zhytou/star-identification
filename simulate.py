from math import radians, sin, cos, tan, sqrt, exp
import numpy as np
import pandas as pd


# read star catalogue
cata_path = 'catalogue/sao.csv'
catalogue = pd.read_csv(cata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])


def cal_avg_star_num_within_fov(mv_limit: float=6.0, fov: float=15) -> float:
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
    h, w = img.shape

    # normalize image
    img = img / 255.0

    # add pepper noise
    num_pepper = int(prob_p * img.size)
    for _ in range(num_pepper):
        x, y = np.random.randint(0, h), np.random.randint(0, w)

        if img[x, y] > 50:
            continue

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


def draw_star(position: tuple[float, float], magnitude: float, img: np.ndarray, sigma: float=1.5, roi: int=2, msaa: bool=False) -> np.ndarray:
    """
        Draw star at position[0](row) and position[1](column) in the image.
    Args:
        position: (starting from top to bottom, starting from left to right)
        magnitude: the stellar magnitude
        img: background image
        sigma: the standard deviation of the point spread function
        roi: the region of interest
        msaa: whether to use multi-sample anti-aliasing
    Returns:
        img: the image with the star drawn
    """
    h, w = img.shape

    H = get_stellar_intensity(magnitude)

    x, y = position
    top, bottom = int(max(0, x-roi)), int(min(h, x+roi+1))
    left, right = int(max(0, y-roi)), int(min(w, y+roi+1))

    # print(x, y, top, bottom, left, right)
    for u in range(top, bottom):
        for v in range(left, right):
            intensity = 0
            if msaa:
                for (du, dv) in [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]: 
                    dd = (u+du-x)**2+(v+dv-y)**2
                    # if dd > roi**2:
                    #     continue
                    intensity += H*exp(-dd/(2*sigma**2))
                img[u ,v] = intensity // 4 if intensity < 1024 else 255
            else:
                dd = (u+0.5-x)**2+(v+0.5-y)**2
                # if dd > roi**2:
                #     continue
                intensity = H*exp(-dd/(2*sigma**2))
                img[u, v] = intensity if intensity < 256 else 255
    return img


def create_star_image(ra: float, de: float, roll: float, sigma_g: float=0.0, prob_p: float=0.0, sigma_pos: float=0.0, sigma_mag: float=0.0, num_fs: int=0, num_ms: int=0, background: float=np.inf, limit_mag: float=7.0,  fov: float=10, h: int=512, w: int=512, f: float=58e-3, roi: int=2, sigma_psf: float=1.0, coords_only: bool=False) -> tuple[np.ndarray, list]:
    """
        Create a star image from the given right ascension, declination and roll angle.
    Args:
        ra: right ascension in radians
        de: declination in radians
        roll: roll in radians
        sigma_g: the nomalized standard deviation of gaussian noise
        prob_p: the probability of pepper noise
        sigma_pos: the standard deviation of positional noise
        sigma_mag: the standard deviation of maginatitude noise
        num_fs: the number of false stars
        num_ms: the number of missing stars
        background: the background intensity in Mag
        limit_mag: the limit of magnitude
        cata_path: the path of star catalogue
        fov: the field of view in degrees
        h: the height of the image
        w: the width of the image
        f: the focal length of the camera
        roi: the region of interest
        coords_only: whether to return only the coordinates of stars
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

        def rotate_x(theta: float) -> np.ndarray:
            '''
                Rotate around x-axis.
            '''
            return np.array([[1, 0, 0],
                             [0, cos(theta), sin(theta)],
                             [0, -sin(theta), cos(theta)]])

        def rotate_z(theta: float) -> np.ndarray:
            '''
                Rotate around z-axis.
            '''
            return np.array([[cos(theta), sin(theta), 0],
                             [-sin(theta), cos(theta), 0],
                             [0, 0, 1]])
        
        # a1 = sin(ra)*cos(roll) - cos(ra)*sin(de)*sin(roll)
        # a2 = -sin(ra)*sin(roll) - cos(ra)*sin(de)*cos(roll)
        # a3 = -cos(ra)*cos(de)
        # b1 = -cos(ra)*cos(roll) - sin(ra)*sin(de)*sin(roll)
        # b2 = cos(ra)*sin(roll) - sin(ra)*sin(de)*cos(roll)
        # b3 = -sin(ra)*cos(de)
        # c1 = cos(ra)*sin(roll)
        # c2 = cos(de)*cos(roll)
        # c3 = -sin(de)
        
        # M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])

        M = rotate_z(roll) @ rotate_x(np.pi/2-de) @ rotate_z(np.pi/2+ra)
        
        return M

    def gen_false_stars(num: int, pos: np.array, min_d: int=4*roi) -> np.ndarray:
        '''
            Add false stars to the image.
        Args:
            img: the image to add false stars
            num: the number of false stars
            pos: the positions of true stars
            min_d: the minimum distance between false stars and true stars
        Returns:
            false_stars: the false stars
        '''
        false_stars = []
        while len(false_stars) < num:
            # rand pixel
            row = np.random.randint(roi, h-roi)
            col = np.random.randint(roi, w-roi)
            # offset
            row += np.random.rand()
            col += np.random.rand()
            if len(pos) > 0:
                ds = np.linalg.norm(pos-(row, col), axis=1)
                if ds.min() < min_d:
                    continue
            mag = 3.0 + np.random.normal(0, 2)
            false_stars.append([-1, row, col, 0, 0, mag])
        return np.array(false_stars)

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll)

    # search for image-able stars
    if True:
        R = sqrt((radians(fov)**2)+(radians(fov)**2))/2
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        assert ra1 < ra2 and de1 < de2

        stars_within_fov = catalogue[(ra1 <= catalogue['Ra']) & (catalogue['Ra'] <= ra2) & 
                                    (de1 <= catalogue['De']) & (catalogue['De'] <= de2)].copy()
    else:
        # star sensor coord
        sensor = np.array([cos(ra)*cos(de), sin(ra)*cos(de), sin(de)]).transpose()

        # fov restriction
        catalogue['Angle'] = np.arccos(catalogue[['X', 'Y', 'Z']].dot(sensor))
        stars_within_fov = catalogue[catalogue['Angle'] >= cos(radians(fov/2))].copy()

    # print(f"Found {len(stars_within_fov)} stars within the field of view.")

    # add magnitude noise if needed
    if sigma_mag > 0:
        stars_within_fov['Magnitude'] += np.random.normal(0, sigma_mag, size=len(stars_within_fov['Magnitude']))

    # exclude stars too dark to identify
    stars_within_fov = stars_within_fov[stars_within_fov['Magnitude'] <= limit_mag]

    # print(f"Found {len(stars_within_fov)} stars within the field of view after mag filtering.")

    # convert to celestial coordinate system
    stars_within_fov['X1'] = np.cos(stars_within_fov['Ra'])*np.cos(stars_within_fov['De'])
    stars_within_fov['Y1'] = np.sin(stars_within_fov['Ra'])*np.cos(stars_within_fov['De'])
    stars_within_fov['Z1'] = np.sin(stars_within_fov['De'])

    # convert to star sensor coordinate system
    stars_within_fov[['X2', 'Y2', 'Z2']] = stars_within_fov[['X1', 'Y1', 'Z1']].dot(M.T)
    
    # convert to image coordinate system
    stars_within_fov['X3'] = w/2*(1+stars_within_fov['X2']/stars_within_fov['Z2']/tan(radians(fov)/2))
    stars_within_fov['Y3'] = h/2*(1+stars_within_fov['Y2']/stars_within_fov['Z2']/tan(radians(fov)/2))
    
    # add positional noise if needed
    if sigma_pos > 0:
        stars_within_fov['X3'] += np.random.normal(0, sigma_pos, size=len(stars_within_fov['X3']))
        stars_within_fov['Y3'] += np.random.normal(0, sigma_pos, size=len(stars_within_fov['Y3']))

    # exclude stars beyond range
    stars_within_fov = stars_within_fov[stars_within_fov['X3'].between(roi, w-roi) & stars_within_fov['Y3'].between(roi, h-roi)]

    # print(f"Found {len(stars_within_fov)} stars within the field of view after pos filtering.")

    # background intensity
    if background == np.inf:
        img = np.zeros((h,w))
    else:
        img = get_stellar_intensity(background) * np.ones((h,w))

    stars_within_fov.rename(columns={'X3': 'Col', 'Y3': 'Row'}, inplace=True)
    stars_within_fov = stars_within_fov[['Star ID', 'Row', 'Col', 'Ra', 'De','Magnitude']].reset_index(drop=True)

    # exclude missing stars if needed
    if num_ms > 0:
        stars_within_fov = stars_within_fov.sample(n=max(1, len(stars_within_fov)-num_ms), random_state=1).reset_index(drop=True)
    
    # stars have to be id, row, col, ra, de, mag
    stars = stars_within_fov.to_numpy()

    # add false stars if needed
    if num_fs > 0:
        false_stars = gen_false_stars(num_fs, stars_within_fov[['Row', 'Col']].to_numpy())
        stars = np.concatenate((stars, false_stars), axis=0)
    
    # sort stars by magnitude
    stars = stars[np.argsort(stars[:, -1])]

    if not coords_only:
        for i in range(len(stars)):
            # draw (row, col) mag
            img = draw_star((stars[i][1], stars[i][2]), stars[i][-1], img, sigma_psf, roi=roi)

        # add gaussian and pepper noise
        img = add_gaussian_and_pepper_noise(img, sigma_g, prob_p)

    return img, stars
    

if __name__ == '__main__':
    h=w=512
    ids, ras, des = catalogue['Star ID'], catalogue['Ra'], catalogue['De']

    for id, ra, de in zip(ids, ras, des):
        print(f"Star ID: {id}, RA: {ra}, DE: {de}")
        _, stars = create_star_image(ra, de, 0, w=w, h=h, coords_only=True)

        idx = np.where(stars[:, 0] == id)[0][0]
        print(f"Star ID: {stars[idx][0]}, Row: {stars[idx][1]}, Col: {stars[idx][2]}, RA: {stars[idx][3]}, DE: {stars[idx][4]}, Mag: {stars[idx][5]}")