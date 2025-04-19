from math import radians, degrees, sin, cos, tan, sqrt, exp, atan
import cv2
import numpy as np
import pandas as pd

from utils import convert_rade2deg, draw_img_with_id_label


# read star catalogue
cata_path = 'catalogue/sao.csv'
catalogue = pd.read_csv(cata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])
catalogue['X'] = np.cos(catalogue['Ra'])*np.cos(catalogue['De'])
catalogue['Y'] = np.sin(catalogue['Ra'])*np.cos(catalogue['De'])
catalogue['Z'] = np.sin(catalogue['De'])


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


def gen_false_stars(num: int, pos: np.array, min_d: int=6, mag_range: tuple=(3, 6), size: tuple=(512, 512)) -> np.ndarray:
    '''
        Add false stars to the image.
    Args:
        img: the image to add false stars
        num: the number of false stars
        pos: the positions of true stars
        min_d: the minimum distance between false stars and true stars
        mag_range: the range of magnitude of false stars
    Returns:
        false_stars: the false stars
    '''
    h, w = size
    false_stars = []
    while len(false_stars) < num:
        # rand pixel
        row = np.random.randint(2, h-2)
        col = np.random.randint(2, w-2)
        # offset
        row += np.random.rand()
        col += np.random.rand()
        if len(pos) > 0:
            ds = np.linalg.norm(pos-(row, col), axis=1)
            if ds.min() < min_d:
                continue
        mag = np.random.uniform(mag_range[0], mag_range[1])
        false_stars.append([-1, row, col, 0, 0, mag])
    return np.array(false_stars)


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


def get_rotation_matrix(ra: float, de: float, roll: float) -> np.ndarray:
    """
        Get the rotation matrix from celestial coordinates to star sensor coordinates. (W = M * V)
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
    
    a1 = sin(ra)*cos(roll) - cos(ra)*sin(de)*sin(roll)
    a2 = -cos(ra)*cos(roll) - sin(ra)*sin(de)*sin(roll)
    a3 = cos(de)*sin(roll)
    b1 = -sin(ra)*sin(roll) - cos(ra)*sin(de)*cos(roll)
    b2 = cos(ra)*sin(roll) - sin(ra)*sin(de)*cos(roll)
    b3 = cos(de)*cos(roll)
    c1 = -cos(ra)*cos(de)
    c2 = -sin(ra)*cos(de)
    c3 = -sin(de)
    
    M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])

    MM = rotate_z(roll) @ rotate_x(np.pi/2+de) @ rotate_z(ra-np.pi/2)
    
    assert np.allclose(M, MM), f"Rotation matrix is not correct. {M} != {MM}"

    return M


def create_star_image(ra: float, de: float, roll: float, sigma_g: float=0.0, prob_p: float=0.0, sigma_pos: float=0.0, sigma_mag: float=0.0, num_fs: int=0, num_ms: int=0, background: float=np.inf, limit_mag: float=7.0, fovy: float=10, fovx: float=10, h: int=512, w: int=512, pixel: float=67e-6, roi: int=2, sigma_psf: float=1.0, coords_only: bool=False) -> tuple[np.ndarray, list]:
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

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll)

    # get field of view
    # ? what happern, when fovx != fovy
    fov = max(fovx, fovy)
    # fov = sqrt(fovx**2 + fovy**2)

    f1 = pixel * w / (2*tan(radians(fovx/2)))
    f2 = pixel * h / (2*tan(radians(fovy/2)))

    assert np.abs(f1-f2) <= 1e-2, "Focal length should be the same in both directions."

    # search for image-able stars
    if False:
        R = sqrt((radians(fov)**2)+(radians(fov)**2))/2
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        assert ra1 < ra2 and de1 < de2
        if ra1 >= 0 and ra2 <= 2*np.pi:
            stars_within_fov = catalogue[(ra1 <= catalogue['Ra']) & (catalogue['Ra'] <= ra2) & 
                                        (de1 <= catalogue['De']) & (catalogue['De'] <= de2)].copy()
        elif ra2 > 2*np.pi:
            ra2 -= 2*np.pi
            stars_within_fov = catalogue[((ra1 <= catalogue['Ra']) & (catalogue['Ra'] <= 2*np.pi)) | 
                                        ((0 <= catalogue['Ra']) & (catalogue['Ra'] <= ra2)) & 
                                        (de1 <= catalogue['De']) & (catalogue['De'] <= de2)].copy()
        else:
            ra1 += 2*np.pi
            stars_within_fov = catalogue[((ra1 <= catalogue['Ra']) & (catalogue['Ra'] <= 2*np.pi)) | 
                                        ((0 <= catalogue['Ra']) & (catalogue['Ra'] <= ra2)) & 
                                        (de1 <= catalogue['De']) & (catalogue['De'] <= de2)].copy()
    else:
        # star sensor coord
        sensor = np.array([cos(ra)*cos(de), sin(ra)*cos(de), sin(de)]).transpose()

        # fov restriction
        catalogue['Angle'] = catalogue[['X', 'Y', 'Z']].dot(sensor)
        stars_within_fov = catalogue[catalogue['Angle'] >= cos(radians(fov/2))].copy()

    # print(f"Found {len(stars_within_fov)} stars within the field of view.")

    # add magnitude noise if needed
    if sigma_mag > 0:
        stars_within_fov['Magnitude'] += np.random.normal(0, sigma_mag, size=len(stars_within_fov['Magnitude']))

    # exclude stars too dark to identify
    stars_within_fov = stars_within_fov[stars_within_fov['Magnitude'] <= limit_mag]

    # print(f"Found {len(stars_within_fov)} stars within the field of view after mag filtering.")

    # convert from celestial coordinate system to star sensor coordinate system
    stars_within_fov[['X', 'Y', 'Z']] = stars_within_fov[['X', 'Y', 'Z']].dot(M.T)
    
    # convert to image coordinate system
    stars_within_fov['X'] = w/2*(1+stars_within_fov['X']/stars_within_fov['Z']/tan(radians(fovx)/2))
    stars_within_fov['Y'] = h/2*(1-stars_within_fov['Y']/stars_within_fov['Z']/tan(radians(fovy)/2))
    
    # add positional noise if needed
    if sigma_pos > 0:
        stars_within_fov['X'] += np.random.normal(0, sigma_pos, size=len(stars_within_fov['X']))
        stars_within_fov['Y'] += np.random.normal(0, sigma_pos, size=len(stars_within_fov['Y']))

    # exclude stars beyond range
    stars_within_fov = stars_within_fov[stars_within_fov['X'].between(roi, w-roi) & stars_within_fov['Y'].between(roi, h-roi)]

    # print(f"Found {len(stars_within_fov)} stars within the field of view after pos filtering.")

    # background intensity
    if background == np.inf:
        img = np.zeros((h,w))
    else:
        img = get_stellar_intensity(background) * np.ones((h,w))

    stars_within_fov.rename(columns={'X': 'Col', 'Y': 'Row'}, inplace=True)
    stars_within_fov = stars_within_fov[['Star ID', 'Row', 'Col', 'Ra', 'De','Magnitude']].reset_index(drop=True)

    # exclude missing stars if needed
    if num_ms > 0:
        stars_within_fov = stars_within_fov.sample(n=max(1, len(stars_within_fov)-num_ms), random_state=1).reset_index(drop=True)
    
    # stars have to be id, row, col, ra, de, mag
    stars = stars_within_fov.to_numpy()

    # add false stars if needed
    if num_fs > 0:
        false_stars = gen_false_stars(num_fs, stars_within_fov[['Row', 'Col']].to_numpy(), size=(h, w))
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
    h, w = 1024, 1280

    # test 1
    # ra, de, roll = radians(249.2104), radians(-12.0386), radians(13.3845)

    # test 2
    R = np.array([
        [-0.433199091912544, 0.824750788118732, -0.363489593061036,],
        [0.821815221905931, 0.195853597987896, -0.535033745850578,],
        [-0.370078758928222, -0.530497413426989, -0.762636352751049]
    ])
    ra, de, roll = np.arctan(R[2][1]/R[2][0]), -np.arcsin(R[2][2]), np.arctan(R[0][2]/R[1][2])
    print(np.degrees(0.97269308), np.degrees(0.83405023))
    print(convert_rade2deg(np.degrees(ra), np.degrees(de)))
    f = 35269.52
    pixel = 5.5
    fovx = degrees(2 * atan(w * pixel / (2 * f)))
    fovy = degrees(2 * atan(h * pixel / (2 * f)))
    limit_mag = 5.5

    # test 3
    # ra, de, roll = 0.84016492, -1.00045128, 0
    # h = w = 512
    # fovx = fovy = 12
    # limit_mag = 6

    print(np.degrees(ra), np.degrees(de), np.degrees(roll))
    print(fovx, fovy)
    img, stars = create_star_image(ra, de, roll, h=h, w=w, limit_mag=limit_mag, fovx=fovx, fovy=fovy)

    # ids = np.array([38787, 39053, 39336, 24412, 39404, 38980, 38597, 38890, 38872, 24531, 24314, 38849, 38768])
    # coords = stars[np.isin(stars[:, 0], ids), 1:3]
    # ids = np.intersect1d(ids, stars[:, 0]).astype(int)
    
    ids = stars[:, 0].astype(int)
    coords = stars[:, 1:3].astype(int)
    # print(ids, coords)

    draw_img_with_id_label(img, coords, ids)

    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for row, col in coords:
    #     cv2.circle(img, (int(col), int(row)), 5, (0, 0, 255), -1)
    
    cv2.imwrite('img.png', img)
    # cv2.waitKey(-1)