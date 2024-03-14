import cv2
import numpy as np
from skimage import measure

from simulate import create_star_image

def get_star_centroids(img: np.ndarray) -> list[tuple[int, int]]:
    '''
        Get the centroids of the stars in the image.
    Args:
        img: the image to be processed
    Returns:
        centroids: the centroids of the stars in the image
    '''

    def cal_multiwind_threshold(img: np.ndarray, wind_len: int=40, num_wind: int=20) -> int:
        """
            Calculate the threshold of the image using the method "multi-window threshold division" from https://ieeexplore.ieee.org/abstract/document/1008988.
        Args:
            wind_len: the length of the window
            num_wind: the number of the windows
        Returns:
            threshold: the threshold of the image
        """        
        # initialize random windows
        threshold = 0

        for i in range(num_wind):
            x = np.random.randint(0, w - wind_len)
            y = np.random.randint(0, h - wind_len)
    
            wind = img[y:y+wind_len, x:x+wind_len]    
            mean = np.mean(wind)  
            std = np.std(wind)
            threshold += mean + 5 * std

        return round(threshold/num_wind)

    def group_star(img: np.ndarray, method: int) -> list[list[tuple[int, int]]]:
        """
            Group the facula(potential star) in the image.
        Args:
            img: the image to be processed
            method: method of connectivity
        Returns:
        """
        # if img[u, v] > 0: 1, else: 0
        _, binary_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)

        # label connected regions of the same value in the binary image
        labeled_img, label_num = measure.label(binary_img, return_num=True, connectivity=method)

        group_coords = []
        for label in range(1, label_num + 1):
            # get the coords for each label
            rows, cols = np.nonzero(labeled_img == label)
            coords = list(zip(rows, cols))
            # two small or too big to be a star
            if len(coords) < 9 or len(coords) > 100:
                continue
            group_coords.append(coords)

        return group_coords

    # get the image size
    h, w = img.shape

    # low pass filter for noise
    img1 = cv2.GaussianBlur(img, (3, 3), 9)

    # calaculate the threshold
    threshold = cal_multiwind_threshold(img1)

    # if img[u, v] < threshold + 20: 0, else: img[u, v]
    _, nimg = cv2.threshold(img1, threshold, 255, cv2.THRESH_TOZERO)

    # cv2.imwrite('img.png', img)
    # cv2.imwrite('img1.png', img1)
    # cv2.imwrite('nimg.png', nimg)

    # rough group star using connectivity
    group_coords = group_star(nimg, 2)

    # calculate the centroid coordinate with threshold and weight
    centroids = []
    for coords in group_coords:
        row_sum = 0
        col_sum = 0
        gray_sum = 0
        for row, col in coords:
            row_sum += row * (img[row][col] - threshold)
            col_sum += col * (img[row][col] - threshold)
            gray_sum += img[row][col] - threshold
        centroids.append((round(row_sum/gray_sum), round(col_sum/gray_sum)))

    return centroids


if __name__ == '__main__':
    # random ra & de test
    num_test = 500
    # generate random right ascension[0, 360] and declination[-90, 90]
    ras = np.random.uniform(-180, 180, num_test)
    des = np.degrees(np.arcsin(np.random.uniform(-1, 1, num_test)))
    # centroid position error
    pos_error = 0
    cnt = 0
    # generate the star image
    for i in range(num_test):
        img, star_info = create_star_image(ras[i], des[i], 0)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))
        # get the centroids of the stars in the image
        stars = get_star_centroids(img)
        if len(stars) > len(star_info):
            print('get wrong star number')
            print(len(stars), len(star_info))
            print(stars, star_table)  
            break
        # compare the centroids with the star_table and calculate the error
        for star in stars:
            if star in star_table:
                continue
            # try to check if the pixel around the star centroid in the star_table due to precision
            (x, y) = star
            if (x-1, y) in star_table or (x+1, y) in star_table or (x, y-1) in star_table or (x, y+1) in star_table:
                pos_error += 1
                cnt += 1
                continue
            real_stars = np.array(list(star_table.keys()))
            distances = np.linalg.norm(real_stars-np.array(list(star)), axis=1)
            idx = np.argmin(distances)
            print(star, real_stars[idx])
            pos_error += np.sqrt((star[0] - real_stars[idx][0])**2 + (star[1] - real_stars[idx][1])**2)
            cnt += 1
    print(f'cnt: {cnt}, pos_error: {pos_error/cnt}')
