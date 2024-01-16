from math import radians, degrees, sin, cos, tan, sqrt, atan, pi, exp
import numpy as np
import pandas as pd
import cv2


def create_star_image(ra: float, de: float, roll: float, f: float=0.00304, myu=1.12*(10**-6)):
    """
        Create a star image from the given right ascension, declination and roll angle.
    Args:
        ra: right ascension in degrees
        de: declination in degrees
        roll: roll in degrees
    Returns:
        img: the simulated star image
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

    def draw_star(x: int, y: int, magnitude: float, background: np.ndarray, ROI: int=5):
        """
            Draw the star in the background image.
        Args:
            x: the x coordinate in pixel (starting from left to right)
            y: the y coordinate in pixel (starting from top to bottom)
            magnitude: the stellar magnitude
            background: background image
            ROI: The region of interest for each star in pixel radius
        """
        # stellar magnitude to intensity
        H = pow(10, 6-magnitude)

        # gaussian distribution variance
        sigma = 1

        for u in range(x-ROI, x+ROI+1):
            if u < 0 or u >= len(background[0]):
                continue
            for v in range(y-ROI, y+ROI+1):
                if v < 0 or v >= len(background):
                    continue
                # gaussian distribution probability density function
                p = exp(-((u-x)^2+(v-y)^2)/(2*(sigma^2)))
                raw_intensity = int(round((H/(2*pi*(sigma**2)))*p))
                background[v ,u] = raw_intensity
        
        return background

    def add_white_noise(low, high, background):
        """
            Adds white noise to an image.
        Args:
            low: lower threshold of the noise generated
            high: maximum pixel value of the noise generated
            background: the image that is put noise on
        """
        row, col = np.shape(background)
        background = background.astype(int)
        noise = np.random.randint(low, high=high, size=(row, col))
        noised_img = cv2.addWeighted(noise, 0.1, background, 0.9, 0)
        return noised_img

    # right ascension, declination and roll in radians
    ra = radians(float(ra))
    de = radians(float(de))
    roll = radians(float(roll))

    # star sensor pixel
    l = 3280
    w = 2464

    # star sensor FOV
    FOVy = degrees(2*atan((myu*w/2)/f))
    FOVx = degrees(2*atan((myu*l/2)/f))

    # get rotation matrix
    M = get_rotation_matrix(ra, de, roll)
    M_transpose = np.round(np.matrix.transpose(M), decimals=5)

    # read star catalogue
    col_list = ["Star ID", "RA", "DE", "Magnitude"]
    star_catalogue = pd.read_csv('catalogues/Below_6.0_SAO.csv', usecols=col_list)
    print(len(star_catalogue))
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
    magnitude_mv = list(stars_within_FOV['Magnitude'])
    filtered_magnitude = []
    for i, (x, y) in enumerate(image_coordinates):
        x = round(xpixel*x)
        y = round(ypixel*y)
        if abs(x) > l/2 or abs(y) > w/2:
            continue
        pixel_coordinates.append((x, y))
        filtered_magnitude.append(magnitude_mv[i])

    print(pixel_coordinates)

    # initialize image
    img = np.zeros((w,l))

    # draw stars
    for i in range(len(filtered_magnitude)):
        x = round(l/2 + pixel_coordinates[i][0])
        y = round(w/2 - pixel_coordinates[i][1])
        img = draw_star(x, y, filtered_magnitude[i], img)

    # add white noise
    img = add_white_noise(0, 50, background=img)

    return img


if __name__ == '__main__':
    img = create_star_image(-180, -90, 0)
    print(img)
    cv2.imwrite("test.png", img)