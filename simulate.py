from math import radians, degrees, sin, cos, tan, sqrt, atan, pi, exp
import numpy as np
import pandas as pd
import cv2
import json


# star sensor pixel num
l = 1024
w = 1024

# star sensor foucs in metres
f = 58e-3

# field of view angle in degrees
FOVx = 12
FOVy = 12

# pixel size in metres
myux = f * tan(radians(FOVx)/2) * 2 / l
myuy = f * tan(radians(FOVy)/2) * 2 / w

# star catalogue path
catalogue_path = 'catalogues/Below_6.0_SAO.csv'

# the standard deviation of white noise 
noise_std = 10


def create_star_image(ra: float, de: float, roll: float):
    """
        Create a star image from the given right ascension, declination and roll angle.
    Args:
        ra: right ascension in degrees
        de: declination in degrees
        roll: roll in degrees
    Returns:
        img: the simulated star image
        stars: stars drawn in the image
    """

    def get_rotation_matrix(ra: float, de: float, roll: float):
        """
            Get the rotation matrix from star sensor coordinates to celestial coordinates. Note that M is an orthogonal matrix, which means the transpose of M represents the transformation matrix from celestial coordinates to star sensor coordinates.
        Args:
            ra: right ascension in radians
            de: declination in radians
            roll: roll angle of star sensor in radians
        Returns:
            M: rotation matrix
        """
        a1 = (sin(ra)*cos(roll)) - (cos(ra)*sin(de)*sin(roll))
        a2 = -(sin(ra)*sin(roll)) - (cos(ra)*sin(de)*cos(roll))
        a3 = -(cos(ra)*cos(de))
        b1 = -(cos(ra)*cos(roll)) - (sin(ra)*sin(de)*sin(roll))
        b2 = (cos(ra)*sin(roll)) - (sin(ra)*sin(de)*cos(roll))
        b3 = -(sin(ra)*cos(de))
        c1 = (cos(ra)*sin(roll))
        c2 = (cos(ra)*cos(roll))
        c3 = -(sin(de))
        M = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
            
        return M

    def draw_star(x: int, y: int, magnitude: float, img: np.ndarray, ROI: int=2):
        """
            Draw star at yth row and xth column in the image.
        Args:
            x: the x coordinate in pixel (starting from left to right)
            y: the y coordinate in pixel (starting from top to bottom)
            magnitude: the stellar magnitude
            img: background image
            ROI: The region of interest for each star in pixel radius
        Returns:
            img: the image with the star drawn
        """
        # stellar magnitude to intensity
        H = 20 * (6-magnitude) + 128

        for u in range(x-ROI, x+ROI+1):
            if u < 0 or u >= len(img[0]):
                continue
            for v in range(y-ROI, y+ROI+1):
                if v < 0 or v >= len(img):
                    continue
                raw_intensity = int(H*exp(-((u-x)**2+(v-y)**2)/(2*ROI**2)))
                img[v ,u] = raw_intensity

        return img

    def add_white_noise(img: np.ndarray):
        """
            Adds white noise to an image.
        Args:
            img: the image to put noise on
        Returns:
            noised_img: the image with white noise
        """
        # generate white noise whose mean is 0 and standard deviation is noise_std
        noise = np.random.normal(0, noise_std, img.shape)
        # make sure no pixel value is less than 0 or greater than 255
        noised_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return noised_img

    # right ascension, declination and roll in radians
    ra = radians(float(ra))
    de = radians(float(de))
    roll = radians(float(roll))

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll)
    M_transpose = np.round(np.matrix.transpose(M), decimals=5)

    # read star catalogue
    col_list = ["Star ID", "RA", "DE", "Magnitude"]
    star_catalogue = pd.read_csv(catalogue_path, usecols=col_list)

    # search for image-able stars
    R = (sqrt((radians(FOVx)**2)+(radians(FOVy)**2))/2)
    ra1, ra2 = (ra - (R/cos(de))), (ra + (R/cos(de)))
    de1, de2 = (de - R), (de + R)

    star_in_ra = star_catalogue[(ra1 <= star_catalogue['RA']) & (star_catalogue['RA'] <= ra2)]
    star_in_de = star_catalogue[(de1 <= star_catalogue['DE']) & (star_catalogue['DE'] <= de2)]
    star_in_de = star_in_de[['Star ID']].copy()
    stars_within_FOV = pd.merge(star_in_ra, star_in_de, on="Star ID")
    
    # convert to star sensor coordinate system
    star_ras = list(stars_within_FOV['RA'])
    star_des = list(stars_within_FOV['DE'])
    coordinates = []
    for i in range(len(star_ras)):
        x = (cos(star_ras[i])*cos(star_des[i]))
        y = (sin(star_ras[i])*cos(star_des[i]))
        z = (sin(star_des[i]))
        coord = M_transpose.dot(np.array([[x], [y], [z]]))
        coordinates.append(coord)

    # convert to image coordinate system
    image_coordinates = []
    for coord in coordinates:
        # coord is like [[x], [y], [z]]
        x = f*(coord[0]/coord[2])[0]
        y = f*(coord[1]/coord[2])[0]
        image_coordinates.append((x, y))

    # calculate pixel num per length
    xtot = 2*tan(radians(FOVx)/2)*f
    ytot = 2*tan(radians(FOVy)/2)*f
    xpixel = l/xtot
    ypixel = w/ytot

    # rescale to pixel sizes
    pixel_coordinates = []
    star_magnitudes = list(stars_within_FOV['Magnitude'])
    star_ids = list(stars_within_FOV['Star ID'])
    filtered_magnitudes = []
    filtered_ids = []
    for i, (x, y) in enumerate(image_coordinates):
        x = round(xpixel*x)
        y = round(ypixel*y)
        if abs(x) > l/2 or abs(y) > w/2:
            continue
        pixel_coordinates.append((x, y))
        filtered_magnitudes.append(star_magnitudes[i])
        filtered_ids.append(star_ids[i])

    # initialize image
    img = np.zeros((w,l))

    # draw imagable stars
    for i in range(len(filtered_magnitudes)):
        x = round(l/2 + pixel_coordinates[i][0])
        y = round(w/2 - pixel_coordinates[i][1])
        img = draw_star(x, y, filtered_magnitudes[i], img)

    # add false stars with random magitudes at random positions
    # false_stars = add_false_stars(img, 5)

    # add white noise
    img = add_white_noise(img)

    # the star infomation
    stars = list(zip(filtered_ids, pixel_coordinates))

    return img, stars


if __name__ == '__main__':
    img, stars = create_star_image(69, -12, -13)
    cv2.imwrite("test2.png", img)
    json.dump(stars, open("test2.json", "w"))
