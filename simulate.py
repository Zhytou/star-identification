import cv2
from math import radians, degrees, sin, cos, tan, sqrt, exp, atan
import numpy as np
import pandas as pd

from utils import label_star_image


# read star catalogue
cata_path = 'catalogue/sao.csv'
cata = pd.read_csv(cata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])
cata['X'] = np.cos(cata['Ra'])*np.cos(cata['De'])
cata['Y'] = np.sin(cata['Ra'])*np.cos(cata['De'])
cata['Z'] = np.sin(cata['De'])


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


def get_rotation_matrix(ra: float, de: float, roll: float, method: int=1) -> np.ndarray:
    """
        Get the the three-dimensional rotation matrix from celestial coordinates to star sensor coordinates. (W = M * V)
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
        return np.array([
            [1, 0, 0],
            [0, cos(theta), sin(theta)],
            [0, -sin(theta), cos(theta)]
        ])

    def rotate_z(theta: float) -> np.ndarray:
        '''
            Rotate around z-axis.
        '''
        return np.array([
            [cos(theta), sin(theta), 0],
            [-sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
    
    if method == 1:
        # method1 高精度星敏感器星图识别算法研究 卫昕
        # M = rotate_z(roll) @ rotate_x(np.pi/2-de) @ rotate_z(np.pi/2+ra)
        a1 = -sin(roll)*sin(de)*cos(ra)-cos(roll)*sin(ra)
        a2 = -sin(roll)*sin(de)*sin(ra)+cos(roll)*cos(ra)
        a3 = sin(roll)*cos(de)
        b1 = -cos(roll)*sin(de)*cos(ra)+sin(roll)*sin(ra)
        b2 = -cos(roll)*sin(de)*sin(ra)-sin(roll)*cos(ra)
        b3 = cos(roll)*cos(de)
        c1 = cos(de)*cos(ra)
        c2 = cos(de)*sin(ra)
        c3 = sin(de)
    else:
        # method2 星图识别算法 张广军
        # M = rotate_z(roll) @ rotate_x(np.pi/2+de) @ rotate_z(-np.pi/2+ra)
        a1 = sin(ra)*cos(roll) - cos(ra)*sin(de)*sin(roll)
        a2 = -cos(ra)*cos(roll) - sin(ra)*sin(de)*sin(roll)
        a3 = cos(de)*sin(roll)
        b1 = -sin(ra)*sin(roll) - cos(ra)*sin(de)*cos(roll)
        b2 = cos(ra)*sin(roll) - sin(ra)*sin(de)*cos(roll)
        b3 = cos(de)*cos(roll)
        c1 = -cos(ra)*cos(de)
        c2 = -sin(ra)*cos(de)
        c3 = -sin(de)
    
    MM = np.array([
        [a1, a2, a3],
        [b1, b2, b3],
        [c1, c2, c3]
    ])
    # assert np.allclose(M, MM), f"Rotation matrix is not correct. {M} != {MM}"

    return MM


def cal_zxz_euler(R: np.ndarray, method: int=1) -> tuple[float, float, float]:
    '''
        Get the ra, de and roll angles from the rotation matrix.
    '''
    assert R.shape == (3, 3), "Rotation matrix should be 3x3."
    assert np.allclose(R @ R.T, np.identity(3), atol=1e-3), "Rotation matrix is not orthogonal."

    if method == 1:
        # method1    
        ra = np.arctan2(R[2, 1], R[2, 0])
        de = np.arcsin(R[2, 2])
        roll = np.arctan2(R[0, 2], R[1, 2])
    else:
        # method2
        ra = np.arctan2(-R[2, 1], -R[2, 0])
        de = np.arcsin(-R[2, 2])
        roll = np.arctan2(R[0, 2], R[1, 2])

    if ra < 0:
        ra += 2*np.pi

    assert ra > 0 and ra < 2*np.pi, f"RA is not in range [0, 2pi]. {ra}"
    assert de > -np.pi/2 and de < np.pi/2, f"DE is not in range [-pi/2, pi/2]. {de}"

    return ra, de, roll


def create_star_image(ra: float, de: float, roll: float, sigma_g: float=0.0, prob_p: float=0.0, sigma_pos: float=0.0, sigma_mag: float=0.0, num_fs: int=0, num_ms: int=0, background: float=np.inf, limit_mag: float=7.0, fovy: float=10, fovx: float=10, h: int=512, w: int=512, roi: int=2, sigma_psf: float=1.0, coords_only: bool=False, rot_meth: int=1) -> tuple[np.ndarray, list]:
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
        pixel: the length of the pixel
        roi: the region of interest
        sigma_psf: the standard deviation of the point spread function
        coords_only: whether to return only the coordinates of stars
        rot_meth: the method used to calculate the rotation matrix
    Returns:
        img: the simulated star image
        stars: stars drawn in the image
    """

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll, rot_meth)

    # calculate fovx if not given
    if fovx == -1:
        fovx = degrees(atan(tan(radians(fovy/2)*w/h)))

    # get field of view
    # ? what happern, when fovx != fovy
    fov = sqrt(fovx**2 + fovy**2)

    f1 = w / (2*tan(radians(fovx/2)))
    f2 = h / (2*tan(radians(fovy/2)))
    assert np.isclose(f1, f2), f"Focal length {f1} and {f2} should be the same in both directions."

    # search for image-able stars
    if False:
        R = sqrt((radians(fov)**2)+(radians(fov)**2))/2
        ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
        de1, de2 = (de - R), (de + R)
        assert ra1 < ra2 and de1 < de2
        if ra1 >= 0 and ra2 <= 2*np.pi:
            stars_within_fov = cata[(ra1 <= cata['Ra']) & (cata['Ra'] <= ra2) & 
                                    (de1 <= cata['De']) & (cata['De'] <= de2)].copy()
        elif ra2 > 2*np.pi:
            ra2 -= 2*np.pi
            stars_within_fov = cata[((ra1 <= cata['Ra']) & (cata['Ra'] <= 2*np.pi)) | 
                                    ((0 <= cata['Ra']) & (cata['Ra'] <= ra2)) & 
                                    (de1 <= cata['De']) & (cata['De'] <= de2)].copy()
        else:
            ra1 += 2*np.pi
            stars_within_fov = cata[((ra1 <= cata['Ra']) & (cata['Ra'] <= 2*np.pi)) | 
                                    ((0 <= cata['Ra']) & (cata['Ra'] <= ra2)) & 
                                    (de1 <= cata['De']) & (cata['De'] <= de2)].copy()
    else:
        # star sensor coord
        sensor = np.array([cos(ra)*cos(de), sin(ra)*cos(de), sin(de)]).transpose()

        # fov restriction
        cata['Angle'] = cata[['X', 'Y', 'Z']].dot(sensor)
        stars_within_fov = cata[cata['Angle'] >= cos(radians(fov/2))].copy()

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
    stars_within_fov['Y'] = h/2*(1+stars_within_fov['Y']/stars_within_fov['Z']/tan(radians(fovy)/2))
    
    # add positional noise if needed
    if sigma_pos > 0:
        stars_within_fov['X'] += np.random.normal(0, sigma_pos, size=len(stars_within_fov['X']))
        stars_within_fov['Y'] += np.random.normal(0, sigma_pos, size=len(stars_within_fov['Y']))

    # exclude stars beyond range
    stars_within_fov = stars_within_fov[stars_within_fov['X'].between(0, w) & stars_within_fov['Y'].between(0, h)]

    # print(f"Found {len(stars_within_fov)} stars within the field of view after pos filtering.")

    # background intensity
    if background == np.inf:
        img = np.zeros((h,w))
    else:
        img = get_stellar_intensity(background) * np.ones((h,w))

    stars_within_fov.rename(columns={'X': 'Col', 'Y': 'Row'}, inplace=True)
    stars_within_fov = stars_within_fov[['Star ID', 'Row', 'Col', 'Ra', 'De','Magnitude']].reset_index(drop=True)

    # stars have to be id, row, col, ra, de, mag
    stars = stars_within_fov.to_numpy()

    # exclude missing stars if needed
    if num_ms > 0 and len(stars) > 0:
        n = len(stars)
        stars = stars[np.random.choice(n, max(1, n-num_ms), replace=False)]

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
    # test 1
    ra, de, roll = radians(249.2104), radians(-12.0386), radians(-13.3845)
    h, w = 1024, 1024
    fovx, fovy = 10, 10
    limit_mag = 6

    # test 2
    # picdata.mat
    R = np.array([
        [-0.4330, 0.8251, -0.3630],
        [-0.8218, -0.1958, 0.5350],
        [0.3703, 0.5300, 0.7629]
    ])
    h, w = 1024, 1280
    f = 35269.52
    pixel = 5.5
    limit_mag = 5

    # test 3
    # xie/20161227224732.bmp
    # R = np.array([
    #     [-0.1551, 0.0848, -0.9843],
    #     [-0.9841, 0.0741, 0.1614],
    #     [0.0866, 0.9936, 0.0720],
    # ])
    # xie/20161227225347.bmp
    # R = np.array([
    #     [-0.1574, 0.0803, -0.9843],
    #     [-0.9857, 0.0474, 0.1615],
    #     [0.0596, 0.9956, 0.0717],
    # ])
    # xie/20161227225550.bmp
    # R = np.array([
    #     [-0.1581, 0.0787, -0.9843],
    #     [-0.9861, 0.0386, 0.1614],
    #     [0.0507, 0.9962, 0.0715]
    # ])
    # xie/20161227225758.bmp
    # R = np.array([
    #     [-0.1586, 0.0774, -0.9843],
    #     [-0.9865, 0.0292, 0.1613],
    #     [0.0412, 0.9966, 0.0717],
    # ])
    # xie/20161227225959.bmp
    # R = np.array([
    #     [-0.1593, 0.0759, -0.9843],
    #     [-0.9867, 0.0208, 0.1612],
    #     [0.0327, 0.9969, 0.0716],
    # ])
    # xie/20161227230200.bmp
    # R = np.array([
    #     [-0.1597, 0.0745, -0.9844],
    #     [-0.9869, 0.0122, 0.1610],
    #     [0.0240, 0.9971, 0.0715]
    # ])
    # xie/20161227230412.bmp
    # R = np.array([
    #     [-0.1603, 0.0729, -0.9844],
    #     [-0.9870, 0.0026, 0.1609],
    #     [0.0143, 0.9973, 0.0715]
    # ])
    # h, w = 1024, 1280
    # f = 34000
    # pixel = 6.7
    # limit_mag = 5.5

    # test 4
    # Tsinghua 3P0/00001010_00000000019CFBA2.bmp
    # R = np.array([
    #     [0.6223, 0.0902, -0.7776],
    #     [-0.2887, 0.9498, -0.1208],
    #     [0.7276, 0.2997, 0.6171],
    # ])
    # Tsinghua 3P0/00001051_00000000019D162E.bmp
    # R = np.array([
    #     [0.6261, 0.0830, -0.7753],
    #     [-0.0305, 0.9962, 0.0821],
    #     [0.7791, -0.0277, 0.6263],
    # ])
    # Tsinghua 3P0/00001052_00000000019D169C.bmp
    # R = np.array([
    #     [0.6262, 0.0824, -0.7753],
    #     [-0.0300, 0.9962, 0.0816],
    #     [0.7791, -0.0278, 0.6263]
    # ])
    # Tsinghua 0P0/00000001_00000000019880C8.bmp
    # R = np.array([
    #     [0.6268, 0.0666, -0.7763],
    #     [-0.2633, 0.9558, -0.1306],
    #     [0.7333, 0.2863, 0.6167],
    # ])
    # h, w = 1040, 1288
    # f = 18500
    # pixel = 4.8
    # limit_mag = 5.5

    fovx = degrees(2 * atan(w * pixel / (2 * f)))
    fovy = degrees(2 * atan(h * pixel / (2 * f)))
    ra, de, roll = cal_zxz_euler(R, 1)
    M = get_rotation_matrix(ra, de, roll, 1)
    assert np.allclose(M, R, atol=1e-3), f"Rotation matrix is not correct. {M} != {R}"

    print(
        'Simulation',
        '\n----------------------',
        '\nh:', h, 'w:', w,
        '\nRa(deg):', round(degrees(ra), 3),
        '\nDe(deg):', round(degrees(de), 3),
        '\nRoll(deg): ', round(degrees(roll), 3),
        '\nFovx(deg):', fovx,
        '\nFovy(deg):', fovy,
        '\nRa(rad):', round(ra, 3),
        '\nDe(rad):', round(de, 3),
        '\nRoll(rad): ', round(roll, 3),
    )
    img, stars = create_star_image(ra, de, roll, h=h, w=w, limit_mag=limit_mag, fovx=fovx, fovy=fovy, rot_meth=1)

    # ids = np.array([38787, 39053, 39336, 24412, 39404, 38980, 38597, 38890, 38872, 24531, 24314, 38849, 38768])
    # coords = stars[np.isin(stars[:, 0], ids), 1:3]
    # ids = np.intersect1d(ids, stars[:, 0]).astype(int)
    
    ids = stars[:, 0].astype(int)
    coords = stars[:, 1:3].astype(int)

    label_star_image(img, coords, ids)

