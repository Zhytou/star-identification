import os
import uuid
import re
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.spatial.distance import cdist

from simulate import w, h, FOV, sim_cfg, create_star_image
from preprocess import get_star_centroids


# guide star catalogue for pattern match database and nn dataset generation
gcata_path = 'catalogue/sao5.6_15_20.csv'
# use for generation config
gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
# guide star catalogue
gcatalogue = pd.read_csv(gcata_path, usecols= ["Star ID", "RA", "DE", "Magnitude"])

# number of reference star
num_class = len(gcatalogue)

# define the path to store the database and pattern as well as dataset
database_path = f'database/{sim_cfg}'
test_path = f'test/{sim_cfg}'
dataset_path = f'dataset/{sim_cfg}'


def generate_pm_database(gen_params: dict, use_preprocess: bool = False, num_thread: int = 10):
    '''
        Generate the pattern database for the given star catalogue.
    Args:
        gen_params: the parameters for the test sample generation
            'grid': grid algorithm
                rb: the radius of buffer region in degrees
                rp: the radius of pattern region in degrees
                Lg: the lenght of grid
            'lpt': log-polar transform algorithm
                rb: the radius of buffer region in degrees
                rp: the radius of pattern region in degrees
                Nd: the number of distance bins
                Nt: the number of theta bins
            'rac': radial and cyclic algorithm
                rb: the radius of buffer region in degreess
                rr: the radius of pattern region for radial features in degrees
                rc: the radius of pattern region for cyclic features in degrees
                Nr: the number of rings
                (the amount of sectors to construct cyclic feature is fixed to 8)
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
    '''
    
    def generate_database(method: str, idxs: pd.Index):
        '''
            Generate the pattern database for the given star catalogue.
        Args:
            method: the method used to generate the database
            idxs: the indexes of star catalogue used to generate database
        Return:
            database: the pattern database
        '''

        database = []
        for ra, de in zip(gcatalogue.loc[idxs, 'RA'], gcatalogue.loc[idxs, 'DE']):
            # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
            img, star_info = create_star_image(ra, de, 0, pure_point=not use_preprocess)
            # generate star_table: (row, col) -> star_id
            star_table = dict(map(lambda x: (x[1], x[0]), star_info))
            # get the centroids of the stars in the image
            if use_preprocess:
                stars = np.array(get_star_centroids(img))
            else:
                stars = np.array(list(star_table.keys()))

            star_id = star_table.get((h/2, w/2), -1)
            if star_id == -1:
                print('The star is not in the center of the image!')
                continue
            # calculate the relative coordinates
            stars = stars-(h/2, w/2)
            # calculate the distance between the star and the center of the image
            distances = np.linalg.norm(stars, axis=1)
            # sort the stars by distance with accending order
            stars = stars[distances.argsort()]
            distances = np.sort(distances)
            # exclude the reference star (h/2, w/2)
            assert stars[0][0] == 0 and stars[0][1] == 0 and distances[0] == 0
            stars, distances = stars[1:], distances[1:]
            if len(stars) < 2:
                continue
            if method == 'grid':
                # exclude stars out of region
                stars = stars[(distances >= Rb) & (distances <= Rp)] 
                # find the nearest neighbor star
                if len(stars) == 0:
                    continue
                nearest_star = stars[0]
                # calculate rotation angle & matrix
                angle = np.arctan2(nearest_star[1], nearest_star[0])
                M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                rotated_stars = np.dot(stars, M)
                assert round(rotated_stars[0][1])==0
                # calculate the pattern
                grid = np.zeros((Lg, Lg), dtype=int)
                for star in rotated_stars:
                    row = int((star[0]/Rp+1)/2*Lg)
                    col = int((star[1]/Rp+1)/2*Lg)
                    grid[row][col] = 1
                # store the 1's position of the grid
                database.append({
                    'pattern': ' '.join(map(str, np.flatnonzero(grid))),
                    'id': star_id
                })
            elif method == 'lpt':
                # exclude stars out of region
                stars = stars[(distances >= Rb) & (distances <= Rp)]
                distances = distances[(distances >= Rb) & (distances <= Rp)]
                if len(stars) == 0:
                    continue
                # do log-polar transform 
                # distance = ln(sqrt(y^2+x^2)), theta = actan(y/x)
                thetas = np.arctan2(stars[:, 1], stars[:, 0])
                # make sure the nearest star's theta is 0
                thetas = thetas - thetas[0]
                # make sure the thetas are in the range of [-pi, pi]
                thetas %= 2*np.pi
                thetas[thetas >= np.pi] -= 2*np.pi
                thetas[thetas < -np.pi] += 2*np.pi
                # generate the pattern
                pattern = [int((d-Rb)/(Rp-Rb)*Nd)*Nt+int((t+np.pi)/(2*np.pi)*Nt) for d, t in zip(distances, thetas)]
                database.append({
                    'pattern': ' '.join(map(str, pattern)),
                    'id': star_id
                })
            else:
                # count the number of stars in each ring
                r_cnts, _ = np.histogram(distances, bins=N, range=(Rb, Rr))
                # generate radial pattern 01 sequence
                r_pattern = np.zeros(N, dtype=int)
                r_pattern[np.nonzero(r_cnts)] = 1

                # exclude stars out of region
                stars = stars[(distances >= Rb) & (distances <= Rc)] 
                # calculate the angles between the star and the center of the image
                angles = np.arctan2(stars[:, 1], stars[:, 0])
                # rotate the stars until the nearest star lies on the horizontal axis
                angles = angles - angles[0]
                # make sure angles are in the range of [-pi, pi]
                angles %= 2*np.pi
                angles[angles >= np.pi] -= 2*np.pi
                angles[angles < -np.pi] += 2*np.pi
                # count the number of stars in each sector
                s_cnts, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
                # generate cyclic pattern 01 sequence
                c_pattern = np.zeros(8, dtype=int)
                c_pattern[np.nonzero(s_cnts)] = 1
                database.append({
                    'pattern': ' '.join(map(str, np.concatenate([r_pattern, c_pattern]))),
                    'id': star_id
                })

        # store the rest of the results
        return pd.DataFrame(database)

    # use thread pool to generate the database
    pool = ThreadPoolExecutor(max_workers=num_thread)
    tasks = defaultdict(list)

    # iterate the methods
    for method in gen_params.keys():
        if method == 'grid':
            # parse parameters: buffer radius, pattern radius and grid length
            rb, rp, Lg = gen_params[method]
            # radius in pixels
            Rb, Rp = rb/FOV*w, rp/FOV*w
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{Lg}'
        elif method == 'lpt':
            # parse parameters: buffer radius, pattern radius, number of distance bins and number of theta bins
            rb, rp, Nd, Nt = gen_params[method]
            # radius in pixels
            Rb, Rp = rb/FOV*w, rp/FOV*w
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{Nd}_{Nt}'
        elif method == 'rac':
            # parse parameters: buffer radius, radial radius, cyclic radius, number of rings
            rb, rr, rc, N = gen_params[method]
            # radius in pixels
            Rb, Rr, Rc = rb/FOV*w, rr/FOV*w, rc/FOV*w
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rr}_{rc}_{N}'
        else:
            print('Invalid method!')
            return
        
        # number of round used for this method
        num_round = num_thread//len(gen_params)
        len_td = len(gcatalogue)//num_round
        # add task
        for i in range(num_round):
            beg, end = i*len_td, min((i+1)*len_td, len(gcatalogue))
            if beg >= end:
                continue
            task = pool.submit(generate_database, method, gcatalogue.index[beg:end])
            tasks[method].append(task)
    
    # wait all tasks to be done and merge all the results
    for method in tasks.keys():
        # make directory to store the database
        if method == 'grid':
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{Lg}'
        elif method == 'lpt':
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{Nd}_{Nt}'
        else:
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rr}_{rc}_{N}'
        if not os.path.exists(path):
            os.makedirs(path)

        # temporary results
        dfs = []
        for task in tasks[method]:
            df = task.result()
            if len(df) > 0:
                dfs.append(df)

        # merge the results
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(f'{path}/{method}.csv', index=False)


def generate_nn_dataset(method: str, gen_params: list, mode: str, num_vec: int, idxs: list, use_preprocess: bool, pos_noise_std: float, mv_noise_std: float, ratio_false_star: float):
    '''
        Generate radial and cyclic features dataset for NN model using the given star catalogue.
    Args:
        method: the method used to generate the dataset
            'rac_1dcnn': the 1st proposed algorithm
            'daa_1dcnn': the 2nd proposed algorithm
            'lpt_nn': log-polar transform based NN algorithm
        gen_params:
            'rac_1dcnn':
                r: the radius of the region in degrees
                arr_Nr: the array of ring number
                Ns: the number of sectors
                Nn: the number of reference neighboring star needed to construct cyclic feature
            'daa_1dcnn':
                r: the radius of the region in degrees
                arr_N: an array of histgram bins number for discretizing the feature sequences
            'lpt_nn':
                r: the radius of the region in degrees
                Nd: the number of distance bins
        mode: generation mode
            'random': use uniformed distributed vector on sphere
            'supplementary': use catalogue index to generate samples for specific star
        num_vec: the number of vectors to be generated
        idxs: the indexes of star catalogue used to generate dataset
        pos_noise_std: the standard deviation of the positional noise
        mv_noise_std: the standard deviation of the magnitude noise
        ratio_false_star: the number of false stars
    Returns:
        df: the dataset
    '''
    # store the label information
    labels = []

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    if mode == 'random':
        ras = np.random.uniform(0, 2*np.pi, num_vec)
        des = np.arcsin(np.random.uniform(-1, 1, num_vec))
    elif mode == 'supplementary':
        ras = np.clip(gcatalogue.loc[idxs, 'RA']+np.radians(np.random.normal(0, 5, len(idxs))), 0, 2*np.pi)
        des = np.clip(gcatalogue.loc[idxs, 'DE']+np.radians(np.random.normal(0, 5, len(idxs))), -np.pi/2, np.pi/2)
    else:
        print('Invalid mode!')
        return

    # generate the star image
    for ra, de in zip(ras, des):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ra, de, 0, pos_noise_std=pos_noise_std, mv_noise_std=mv_noise_std, ratio_false_star=ratio_false_star, pure_point=not use_preprocess)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))
        
        # get the centroids of the stars in the image
        if use_preprocess:
            stars = np.array(get_star_centroids(img))
        else:
            stars = np.array(list(star_table.keys()))
        if len(stars) < 4:
            continue

        # generate a unique img id for later accuracy calculation
        img_id = uuid.uuid1()

        # distances and angles between each star in FOV
        distances = cdist(stars, stars, 'euclidean')
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
            # check if false star or star not in guide catalogue
            star_id = star_table.get(tuple(star), -1)
            if star_id == -1 or star_id not in gcatalogue['Star ID'].values:
                continue
            # get catalogue index of the guide star
            cata_idx = gcatalogue[gcatalogue['Star ID'] == star_id].index.to_list()[0]

            # angles is sorted by distance with accending order
            ags = ags[np.argsort(ds)]
            ds = np.sort(ds)
            # remove the first element of ags & ds, which is reference star
            ds, ags = ds[1:], ags[1:]

            # generate label information for training and testing
            label = {
                'ra': ra,
                'de': de,
                'star_id': star_id,
                'cata_idx': cata_idx,
                'img_id': img_id
            }
            if method == 'rac_1dcnn':
                # parse the parameters:
                r, arr_Nr, Ns, Nn = gen_params
                # calculate the radius in pixels
                R = r/FOV*w
                if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                    continue
                # exclude angles outside the region
                ags = ags[ds <= R]
                # skip if only the reference star in the region
                if len(ags) == 0:
                    continue

                tot_Nr = 0
                for Nr in arr_Nr:
                    # count the number of stars in each ring
                    r_cnts, _ = np.histogram(ds, bins=Nr, range=(0, R))
                    for i, rc in enumerate(r_cnts):
                        i = tot_Nr+i
                        label[f'ring{i}'] = rc
                    tot_Nr += Nr
                # uses several neighbor stars as the starting angle to obtain the cyclic features
                for i, ag in enumerate(ags[:Nn]):
                    # make sure all angles stay in [-pi, pi] after rotating the first angle to 0 degree
                    rot_ags = ags - ag
                    rot_ags %= 2*np.pi
                    rot_ags[rot_ags > np.pi] -= 2*np.pi
                    rot_ags[rot_ags < -np.pi] += 2*np.pi
                    # count the number of stars in each sector
                    s_cnts, _ = np.histogram(rot_ags, bins=Ns, range=(-np.pi, np.pi))
                    for j, sc in enumerate(s_cnts):
                        label[f'n{i}_sector{j}'] = sc
                if len(ags) < Nn:
                    for i in range(len(ags), Nn):
                        for j in range(Ns):
                            label[f'n{i}_sector{j}'] = 0
                labels.append(label)
            elif method == 'daa_1dcnn':
                # parse the parameters:
                r, arr_N = gen_params
                # calculate the radius in pixels
                R = r/FOV*w
                if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                    continue
                # exclude angles and distances outside the region
                ags = ags[ds <= R]
                ds = ds[ds <= R]
                # skip if only the reference star in the region
                if len(ags) == 0:
                    continue

                # statistics of distances and angles
                stats = []
                for seq in [ds, ags]:
                    stats.extend([np.min(seq), np.max(seq), np.median(seq), np.mean(seq)])
                for i, stat in enumerate(stats):
                    label[f'stat{i}'] = stat

                # discretize the feature sequence(distances and angles) with different levels
                tot_N = 0
                for N in arr_N:
                    # density is set True
                    d_pdf, _ = np.histogram(ds, bins=N, range=(0, R), density=True)
                    for i, p in enumerate(d_pdf):
                        i += tot_N
                        label[f'dist{i}'] = p
                    a_pdf, _ = np.histogram(ags, bins=N, range=(-np.pi, np.pi), density=True)
                    for i, p in enumerate(a_pdf):
                        i += tot_N
                        label[f'angle{i}'] = p
                    tot_N += N
                labels.append(label)
            elif method == 'lpt_nn':
                # parse the parameters:
                r, Nd= gen_params
                # calculate the radius in pixels
                R = r/FOV*w
                if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                    continue
                # count the number of stars in each distance bin
                d_cnts, _ = np.histogram(ds, bins=Nd, range=(0, R))
                # normalize d_cnts
                for i, dc in enumerate(d_cnts):
                    label[f'dist{i}'] = dc
                labels.append(label)
            else:
                pass

    # return the dataframe dict
    return pd.DataFrame(labels)


def generate_test_samples(num_vec: int, gen_params: dict, use_preprocess: bool = False, pos_noise_std: float = 0, mv_noise_std: float = 0, ratio_false_star: float = 0):
    '''
        Generate pattern match test case.
    Args:
        num_vec: the number of vectors to be generated
        gen_params: the parameters for the test sample generation
            'rac_1dcnn':
                r: the radius of the region in degrees
                arr_Nr: the array of ring number
                Ns: the number of sectors
                Nn: the minimum number of neighbor stars in the region
            'rac_1dcnn':
                r: the radius of the region in degrees
                arr_N: the array of histgram bins number for discretizing the feature sequences
            'lpt_nn': 
                r: the radius of the region in degrees
                Nd: the number of distance bins
            'grid': grid algorithm
                rb: the radius of buffer region in degrees
                rp: the radius of pattern region in degrees
                Ng: the number of grids
            'lpt': log-polar transform algorithm
                rb: the radius of buffer region in degrees
                rp: the radius of pattern region in degrees
                Nd: the number of distance bins
                Nt: the number of theta bins
            'rac': radial and cyclic algorithm
                rb: the radius of buffer region in degreess
                rr: the radius of pattern region for radial features in degrees
                rc: the radius of pattern region for cyclic features in degrees
                Nr: the number of rings
                Ns: the number of sectors
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
        pos_noise_std: the standard deviation of the positional noise
        mv_noise_std: the standard deviation of the magnitude noise
        ratio_false_star: the number of false stars
    Returns:
        dict: method->dataframe
    '''

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    ras = np.random.uniform(0, 2*np.pi, num_vec)
    des = np.arcsin(np.random.uniform(-1, 1, num_vec))

    # the dict to store the results
    patterns = defaultdict(list)

    # generate the star image
    for ra, de in zip(ras, des):
        # star_info is a list of [star_ids[i], (row, col), star_magnitudes[i]]
        img, star_info = create_star_image(ra, de, 0, pos_noise_std=pos_noise_std, mv_noise_std=mv_noise_std, ratio_false_star=ratio_false_star, pure_point=not use_preprocess)
        # generate star_table: (row, col) -> star_id
        star_table = dict(map(lambda x: (x[1], x[0]), star_info))

        # get the centroids of the stars in the image
        if use_preprocess:
            stars = np.array(get_star_centroids(img))
        else:
            stars = np.array(list(star_table.keys()))

        # too few stars for quest algorithm to identify satellite attitude
        if len(stars) < 4:
            continue
        
        # R = gen_params['nn'][0]/FOV*w
        # # number of candidate primary stars in the region
        # in_rect  = np.logical_and(
        #     np.logical_and(stars[:, 0] >= R/2, stars[:, 0] <= h-R/2),
        #     np.logical_and(stars[:, 1] >= R/2, stars[:, 1] <= w-R/2)
        # )
        # # too few stars to identify satellite attitude
        # if np.sum(in_rect) < 3:
        #     continue

        # generate a unique img id for later accuracy calculation
        img_id = uuid.uuid1()

        distances = cdist(stars, stars, 'euclidean')
        # distances = np.linalg.norm(stars[:,None,:] - stars[None,:,:], axis=-1)
        angles = np.arctan2(stars[:, 1] - stars[:, 1][:, None], stars[:, 0] - stars[:, 0][:, None])
        # choose a guide star as the reference star
        for star, ds, ags in zip(stars, distances, angles):
            # check if false star or not in guide star catalogue
            star_id = star_table.get(tuple(star), -1)
            if star_id == -1 or star_id not in gcatalogue['Star ID'].values:
                continue

            # angles is sorted by distance with accending order
            ss, ags = stars[np.argsort(ds)], ags[np.argsort(ds)]
            ds = np.sort(ds)
            # remove the first element, which is reference star
            assert star[0] == ss[0][0] and star[1] == ss[0][1] and ds[0] == 0 and ags[0] == 0
            ss, ds, ags = ss[1:], ds[1:], ags[1:]
            # calculate the relative coordinates
            ss = ss-star
            if len(ss) < 2:
                continue

            methods = list(gen_params.keys())
            for method in methods:
                if method == 'rac_1dcnn':
                    # parse the parameters
                    r, arr_Nr, Ns, Nn = gen_params[method]
                    # radius in pixels
                    R = r/FOV*w
                    if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                        continue
                    # get catalogue index of the guide star
                    cata_idx = gcatalogue[gcatalogue['Star ID'] == star_id].index.to_list()[0]
                    pattern = {
                        'star_id': star_id,
                        'cata_idx': cata_idx,
                        'img_id': img_id
                    }
                    
                    tot_Nr = 0
                    for Nr in arr_Nr:
                        r_cnts, _ = np.histogram(ds, bins=Nr, range=(0, R))
                        for i, rc in enumerate(r_cnts):
                            i += tot_Nr
                            pattern[f'ring{i}'] = rc
                        tot_Nr += Nr
                    
                    # exlcude stars' angle out of region
                    excl_ags = ags[ds <= R]
                    # uses several neighbor stars as the starting angle to obtain the cyclic features
                    for i, ag in enumerate(excl_ags[:Nn]):
                        # rotate stars
                        rot_ags = excl_ags - ag
                        # make sure all angles stay in [-pi, pi]
                        rot_ags %= 2*np.pi
                        rot_ags[rot_ags > np.pi] -= 2*np.pi
                        rot_ags[rot_ags < -np.pi] += 2*np.pi
                        # calculate sectore counts using histogram
                        s_cnts, _ = np.histogram(rot_ags, bins=Ns, range=(-np.pi, np.pi))
                        for j, sc in enumerate(s_cnts):
                            pattern[f'n{i}_sector{j}'] = sc
                    # add trailing zero if there is not enough neighbors
                    if len(excl_ags) < Nn:
                        for i in range(len(excl_ags), Nn):
                            for j in range(Ns):
                                pattern[f'n{i}_sector{j}'] = 0

                    patterns[method].append(pattern)
                elif method == 'daa_1dcnn':
                    # parse the parameters
                    r, arr_N = gen_params[method]
                    # radius in pixels
                    R = r/FOV*w
                    if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                        continue
                    # get catalogue index of the guide star
                    cata_idx = gcatalogue[gcatalogue['Star ID'] == star_id].index.to_list()[0]
                    pattern = {
                        'star_id': star_id,
                        'cata_idx': cata_idx,
                        'img_id': img_id
                    }

                    # exclude angles and distances outside the region
                    excl_ds, excl_ags = ds[ds <= R], ags[ds <= R]
                    # skip if only the reference star in the region
                    if len(ags) == 0:
                        continue

                    # statistics of distances and angles
                    stats = []
                    for seq in [excl_ds, excl_ags]:
                        stats.extend([np.min(seq), np.max(seq), np.median(seq), np.mean(seq)])
                    for i, stat in enumerate(stats):
                        pattern[f'stat{i}'] = stat

                    # discretize the feature sequence(distances and angles) with different levels
                    tot_N = 0
                    for N in arr_N:
                        # density is set True
                        d_pdf, _ = np.histogram(excl_ds, bins=N, range=(0, R), density=True)
                        for i, p in enumerate(d_pdf):
                            i += tot_N
                            pattern[f'dist{i}'] = p
                        a_pdf, _ = np.histogram(excl_ags, bins=N, range=(-np.pi, np.pi), density=True)
                        for i, p in enumerate(a_pdf):
                            i += tot_N
                            pattern[f'angle{i}'] = p
                        tot_N += N
                    patterns[method].append(pattern)
                elif method == 'lpt_nn':
                    # parse the parameters
                    r, Nd = gen_params[method]
                    # radius in pixels
                    R = r/FOV*w
                    if star[0] < R/2 or star[0] > h-R/2 or star[1] < R/2 or star[1] > w-R/2:
                        continue
                    # get catalogue index of the guide star
                    cata_idx = gcatalogue[gcatalogue['Star ID'] == star_id].index.to_list()[0]
                    pattern = {
                        'star_id': star_id,
                        'cata_idx': cata_idx,
                        'img_id': img_id
                    }
                    # count the number of stars in each distance bin
                    d_cnts, _ = np.histogram(ds, bins=Nd, range=(0, R))
                    # normalize d_cnts
                    for i, dc in enumerate(d_cnts):
                        pattern[f'dist{i}'] = dc
                    patterns[method].append(pattern)
                elif method == 'grid':
                    # parse the parameters: buffer radius, pattern radius and grid length
                    rb, rp, Lg = gen_params[method]
                    # radius in pixels
                    Rb, Rp = rb/FOV*w, rp/FOV*w
                    # exclude stars outside the region
                    excl_ss, excl_ags = ss[(ds >= Rb) & (ds <= Rp)], ags[(ds >= Rb) & (ds <= Rp)]
                    if len(excl_ss) < 2:
                        continue
                    M = np.array([[np.cos(excl_ags[0]), -np.sin(excl_ags[0])], [np.sin(excl_ags[0]), np.cos(excl_ags[0])]])
                    # rotate stars
                    rot_ss = np.dot(excl_ss, M)
                    assert round(rot_ss[0][1])==0
                    # calculate the pattern
                    grid = np.zeros((Lg, Lg), dtype=int)
                    for s in rot_ss:
                        row = int((s[0]/Rp+1)/2*Lg)
                        col = int((s[1]/Rp+1)/2*Lg)
                        grid[row][col] = 1
                    # store the 1's position of grid
                    patterns[method].append({
                        'img_id': img_id,
                        'pattern': ' '.join(map(str, np.flatnonzero(grid))), 
                        'id': star_id
                    })
                elif method == 'lpt':
                    # parse parameters: buffer radius, pattern radius, number of distance bins and number of theta bins
                    rb, rp, Nd, Nt = gen_params[method]
                    # radius in pixels
                    Rb, Rp = rb/FOV*w, rp/FOV*w
                    # exclude stars outside the region
                    excl_ags, excl_ds = ags[(ds >= Rb) & (ds <= Rp)], ds[(ds >= Rb) & (ds <= Rp)]
                    if len(excl_ags) < 2:
                        continue
                    # log-polar transform is already done, where excl_ags is the thetas and excl_ds is the distances
                    # rotate the stars, so that the nearest star's theta is 0
                    rot_ags = excl_ags - excl_ags[0]
                    # make sure the thetas are in the range of [-pi, pi]
                    rot_ags %= 2*np.pi
                    rot_ags[rot_ags >= np.pi] -= 2*np.pi
                    rot_ags[rot_ags < -np.pi] += 2*np.pi
                    # generate the pattern
                    pattern = [int((d-Rb)/(Rp-Rb)*Nd)*Nt+int((t+np.pi)/(2*np.pi)*Nt) for d, t in zip(excl_ds, rot_ags)]
                    patterns[method].append({
                        'img_id': img_id,
                        'pattern': ' '.join(map(str, pattern)),
                        'id': star_id
                    })
                elif method == 'rac':
                    # parse the parameters: buffer radius, radial radius, cyclic radius, number of rings
                    rb, rr, rc, N = gen_params[method]
                    # radius in pixels
                    Rb, Rr, Rc = rb/FOV*w, rr/FOV*w, rc/FOV*w

                    # count the number of stars in each ring
                    r_cnts, _ = np.histogram(ds, bins=N, range=(Rb, Rr))
                    # generate radial pattern 01 sequence
                    r_pattern = np.zeros(N, dtype=int)
                    r_pattern[np.nonzero(r_cnts)] = 1

                    # exclude stars outside the region
                    pm2_ags = ags[(ds >= Rb) & (ds <= Rc)]
                    # rotate the stars until the nearest star lies on the horizontal axis
                    if len(pm2_ags) < 2:
                        continue
                    ags = ags-ags[0]
                    # make sure angles are in the range of [-pi, pi]
                    pm2_ags %= 2*np.pi
                    pm2_ags[pm2_ags > np.pi] -= 2*np.pi
                    pm2_ags[pm2_ags < -np.pi] += 2*np.pi
                    s_cnts, _ = np.histogram(pm2_ags, bins=8, range=(-np.pi, np.pi))
                    # generate cyclic pattern 01 sequence
                    c_pattern = np.zeros(8, dtype=int)
                    c_pattern[np.nonzero(s_cnts)] = 1

                    patterns[method].append({
                        'img_id': img_id,
                        'pattern': ' '.join(map(str, np.concatenate([r_pattern, c_pattern]))),
                        'id': star_id
                    })                    
                else:
                    print('Invalid method')

    # convert the results into dataframe
    for key in patterns:
        patterns[key] = pd.DataFrame(patterns[key])
    return patterns


def aggregate_nn_dataset(types: dict, gen_params: dict, use_preprocess: bool, default_ratio: float, pos_noise_stds: list = [], mv_noise_stds: list = [], ratio_false_stars: list = [], num_thread: int = 40, fine_grained: bool = False):
    '''
        Aggregate the dataset. Firstly, the number of samples for each class is counted. Then, roughly generate classes with too few samples using generate_nn_dataset function's 'random' mode. Lastly, the rest are finely generated to ensure that the number of samples in each class in the entire dataset reaches the standard.
    Args:
        types: key->the types of the dataset, values->the minumin number of samples for each class
        gen_params: the parameters for the test sample generation
            'rac_1dcnn': 
                rp: the radius of the pattern region in degrees
                arr_Nr: the array of ring number
                Ns: the number of sectors
                Nn: the number of neighbor stars
            'daa_1dcnn':
                rp: the radius of the pattern region in degrees
                arr_N: an array of histgram bins number for discretizing the feature sequences
            'lpt_nn':
                rp: the radius of the pattern region in degrees
                Nd: the number of distance bins
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
        default_ratio: the ratio of default dataset versus total dataset
        pos_noise_stds: list of positional noise
        mv_noise_stds: list of magnitude noise
        ratio_false_stars: list of false star ratio
        num_thread: the number of threads to generate the dataset
    '''

    def wait_tasks(tasks: dict, root_dirs: dict, file_name: str, col_name: str = None):
        '''
            Wait for all tasks to be done. Then, merge the result of async tasks and store it in labels.csv. 
        Args:
            tasks: the tasks to be done
            root_dirs: the root directory for each method to store the dataset
            col_name: the column name of the label
        Returns:
            tasks: the tasks done(reserved keys and empty list for values)
        '''
        for method in tasks:
            for key in tasks[method]:
                if len(tasks[method][key]) == 0:
                    continue

                # make directory for every type of dataset
                path = os.path.join(root_dirs[method], key)
                if not os.path.exists(path):
                    os.makedirs(path)

                # store the samples
                dfs = []
                for task in tasks[method][key]:
                    df = task.result()
                    if len(df) > 0:
                        df.to_csv(f"{path}/{uuid.uuid1()}", index=False)
                        dfs.append(df)
                
                # remain the old samples
                if os.path.exists(os.path.join(path, file_name)):
                    dfs.append(pd.read_csv(os.path.join(path, file_name)))

                # aggregate the dataset
                df = pd.concat(dfs, ignore_index=True)
                dfs.clear()
                
                # truncate and store the dataset
                df = df.groupby(col_name, group_keys=False).apply(lambda x: x.sample(n=min(len(x), types[key])))
                df.to_csv(f'{path}/{file_name}', index=False)
                                
                # print dataset distribution per class
                if col_name:
                    df_info = df[col_name].value_counts()
                    print(method, key, len(df_info))
                    print(df_info.tail(5))

    def parse_params(s: str):
        '''
            Parse special test parameters.
        Args:
            s: the string to be parsed
        Returns:
            pns: the positional noise standard deviation
            mns: the magnitude noise standard deviation
            rfs: the ratio of false stars
        '''
        pns, mns, rfs = 0, 0, 0
        match = re.match('.+\/(pos|mv|fs)([0-9]+\.?[0-9]*)', s)
        if match:
            test_type, number = match.groups()
            if test_type == 'pos':
                pns = float(number)
            elif test_type == 'mv':
                mns = float(number)
            else:  # test_type == 'fs'
                rfs = float(number)
        return pns, mns, rfs

    def aggregate_root_dir(root_dirs: dict):
        '''
            Aggregate the train, validate and test dataset labels.csv in the root directory.
        Args:
            root_dirs: the root directory for each method to store the dataset
        '''
        for method in root_dirs:
            # name in ['train', 'validate', 'test']
            for name in names:
                dfs = [pd.read_csv(os.path.join(root_dirs[method], key, 'labels.csv')) for key in types.keys() if name in key]
                if len(dfs) > 0:
                    df = pd.concat(dfs, ignore_index=True)
                    df.to_csv(os.path.join(root_dirs[method], name, 'labels.csv'), index=False)

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thread)
    
    # generate config for each dataset
    if len(pos_noise_stds)+len(mv_noise_stds)+len(ratio_false_stars) > 0:
        noised_ratio = (1-default_ratio)/(len(pos_noise_stds)+len(mv_noise_stds)+len(ratio_false_stars))
    names = list(types.keys())
    for name in names:
        for pns in pos_noise_stds:
            types[f'{name}/pos{pns}'] = max(int(types[name]*noised_ratio), 1)
        for mns in mv_noise_stds:
            types[f'{name}/mv{mns}'] = max(int(types[name]*noised_ratio), 1)
        for rfs in ratio_false_stars:
            types[f'{name}/fs{rfs}'] = max(int(types[name]*noised_ratio), 1)
        types[f'{name}/default'] = max(int(types.pop(name)*default_ratio), 1)
    print(types)

    # rough generation
    tasks = {}
    # the root directory for each method to store the dataset
    root_dirs = {}
    for method in gen_params:
        if method == 'rac_1dcnn':
            # parse parameters: radius, number of rings, number of sectors, number of neighbors
            r, arr_Nr, Ns, Nn = gen_params[method]
            # generate config
            gen_cfg = f'{gcata_name}_{int(use_preprocess)}_{r}_{arr_Nr}_{Ns}_{Nn}'
        elif method == 'daa_1dcnn':
            r, arr_N = gen_params[method]
            # generate config
            gen_cfg = f'{gcata_name}_{int(use_preprocess)}_{r}_{arr_N}'
        elif method == 'lpt_nn':
            r, Nd = gen_params[method]
            # generate config
            gen_cfg = f'{gcata_name}_{int(use_preprocess)}_{r}_{Nd}'
        else:
            pass
        
        tasks[method] = defaultdict(list)
        root_dirs[method] = os.path.join(dataset_path, method, gen_cfg)
        for key in types.keys():
            # reuse old samples
            files = []
            # the storage path for each method's old samples, e.g /dataset_path/rac_1dcnn/xxx/train/mv1 
            path = os.path.join(dataset_path, method, gen_cfg, key)
            if os.path.exists(path):
                files = os.listdir(path)
            pct = 0
            if len(files) > 0:
                df = pd.concat([pd.read_csv(os.path.join(path, file)) for file in files if file != 'labels.csv'])
                df.to_csv(os.path.join(path, 'labels.csv'), index=False)
                # count the number of samples for each class
                df = df['cata_idx'].value_counts()
                pct = len(df[df >= types[key]])/len(gcatalogue)
                if len(df[df < types[key]]) == 0:
                    avg_num = 0
                else:
                    avg_num = np.sum(types[key]-df[df < types[key]])/len(df[df < types[key]])
                print(method, key, ' pct: ', pct, ' len(df[df < types[key]]): ', len(df[df < types[key]]), ' avg_num:',
                       avg_num)
                if pct > 0.95 or avg_num <= 1:
                    print(f'{key} skip rough generation!')
                    continue
                if avg_num < 1:
                    avg_num = 1
            else:
                avg_num = types[key]
            # parse parameters
            pos_noise_std, mv_noise_std, ratio_false_star = parse_params(key)

            # roughly generate the samples for each class        
            num_round = min(num_thread//4, int(avg_num), int((1-pct)*len(gcatalogue)))
            num_vec = 1000
            print('random generate round: ', num_round)
            for _ in range(num_round):
                task = pool.submit(generate_nn_dataset, method, gen_params[method], 'random', num_vec, [], use_preprocess, pos_noise_std, mv_noise_std, ratio_false_star)
                tasks[method][key].append(task)

    # wait for all tasks to be done and merge the results
    wait_tasks(tasks, root_dirs, 'labels.csv', 'cata_idx')

    if not fine_grained:
        aggregate_root_dir(root_dirs)
        return

    tasks = {}
    # fine generation
    for method in gen_params:
        tasks[method] = defaultdict(list)
        for key in types.keys():
            df = pd.read_csv(os.path.join(root_dirs[method], key, 'labels.csv'))
            # count the number of samples for each class
            df = df['cata_idx'].value_counts()
            # add those even unexisting class(catalogue idx)
            df = df.reindex(gcatalogue.index, fill_value=0)
            # count class whose samples less than limit
            df = df[df < types[key]]
            # repeat each the index of df (which is catalogue idx needed)
            idxs = df.index.repeat(types[key]-df).to_list()
            print(len(df), len(idxs), idxs[:3], idxs[-3:])
            # parse parameters
            pos_noise_std, mv_noise_std, ratio_false_star = parse_params(key)

            # for class whose sample less than num_sample_per_class, generate the dataset til the number of samples reach the standard
            if len(idxs) > 1000:
                len_td = 1000
                num_round = len(idxs)//len_td
                if num_round > num_thread//2:
                    num_round = num_thread//2
                    len_td = len(idxs)//num_round
            else:
                len_td = len(idxs)
                num_round = 1
            for i in range(num_round+1):
                beg, end = i*len_td, min((i+1)*len_td, len(idxs))
                if beg >= end:
                    continue
                task = pool.submit(generate_nn_dataset, method, gen_params[method], 'supplementary', 0, idxs[beg: end], use_preprocess, pos_noise_std, mv_noise_std, ratio_false_star)
                tasks[method][key].append(task)
        
    wait_tasks(tasks, root_dirs, 'labels.csv', 'cata_idx')
    
    # aggregate the labels.csv in root directory
    aggregate_root_dir(root_dirs)


def aggregate_test_samples(num_vec: int, gen_params: dict, use_preprocess: bool = False, generate_default: bool = True, pos_noise_stds: list = [], mv_noise_stds: list = [], ratio_false_stars: list = [], num_thread: int = 20):
    '''
    Aggregate the test samples. 
    Args:
        num_vec: number of vectors used to generate test samples
        gen_params: the parameters for the test sample generation, possible methods include:
            'rac_1dcnn':
                r: the radius of the region in degrees
                arr_Nr: the array of ring number
                Ns: the number of sectors
                Nn: the minimum number of neighbor stars in the region
            'rac_1dcnn':
                r: the radius of the region in degrees
                arr_N: the array of histgram bins number for discretizing the feature sequences
            'lpt_nn': 
                r: the radius of the region in degrees
                Nd: the number of distance bins
            'grid': grid algorithm
                rb: the radius of buffer region in degrees
                rp: the radius of pattern region in degrees
                Ng: the number of grids
            'lpt': log-polar transform algorithm
                rb: the radius of buffer region in degrees
                rp: the radius of pattern region in degrees
                Nd: the number of distance bins
                Nt: the number of theta bins
            'rac': radial and cyclic algorithm
                rb: the radius of buffer region in degreess
                rr: the radius of pattern region for radial features in degrees
                rc: the radius of pattern region for cyclic features in degrees
                Nr: the number of rings
                Ns: the number of sectors
        use_preprocess: whether to avoid the error resulted from get_star_centroids function in preprocess stage
        generate_default: whether to generate test samples for default case
        pos_noise_stds: list of positional noise
        mv_noise_stds: list of magnitude noise
        ratio_false_stars: list of false star number
        num_thread: the number of threads to generate the test samples
    '''

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thread)
    # tasks for later aggregation
    tasks = defaultdict(list)

    if num_vec*len(gen_params) < 500:
        num_thread = 1
    elif num_vec*len(gen_params)//num_thread > 500:
        num_vec = num_vec//num_thread
    else:
        num_thread = num_vec*len(gen_params)//500
        num_vec = 500//len(gen_params)

    for _ in range(num_thread):
        task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess)
        if generate_default:
            tasks['default'].append(task)
        for pns in pos_noise_stds:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, pos_noise_std=pns)
            tasks[f'pos{pns}'].append(task)
        for mns in mv_noise_stds:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, mv_noise_std=mns)
            tasks[f'mv{mns}'].append(task)
        for rfs in ratio_false_stars:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, ratio_false_star=rfs)
            tasks[f'fs{rfs}'].append(task)

    # get the async task result and store the returned dataframe
    for key in tasks.keys():
        for task in tasks[key]:
            df_dict = task.result()
            for method in df_dict.keys():
                gen_cfg = f'{gcata_name}_{int(use_preprocess)}_'+'_'.join(map(str, gen_params[method]))
                path = os.path.join(test_path, method, gen_cfg, key)
                if not os.path.exists(path):
                    os.makedirs(path)
                df = df_dict[method]
                df.to_csv(os.path.join(path, str(uuid.uuid1())), index=False)

    # aggregate all the test patterns
    for method in gen_params:
        gen_cfg = f'{gcata_name}_{int(use_preprocess)}_'+'_'.join(map(str, gen_params[method]))
        path = os.path.join(test_path, method, gen_cfg)
        if not os.path.exists(path):
            continue
        # sub test dir names
        test_names = os.listdir(path)
        for tn in test_names:
            p = os.path.join(path, tn)
            dfs = [pd.read_csv(os.path.join(p, f)) for f in os.listdir(p) if f != 'labels.csv']
            print(p, len(dfs))
            if len(dfs) > 0:        
                df = pd.concat(dfs, ignore_index=True)
                df.to_csv(os.path.join(p, 'labels.csv'))


if __name__ == '__main__':
    # generate_pm_database({'grid': [0, 6, 50], 'lpt': [0, 6, 50, 50]})
    aggregate_nn_dataset({'train': 1}, {'daa_1dcnn': [6, [5]]}, use_preprocess=False, default_ratio=1, fine_grained=False, num_thread=4)
    # aggregate_test_samples(1000, {'rac_1dcnn': [6, 50, 16, 3], 'lpt_nn': [6, 50]}, use_preprocess=False, generate_default=False, mv_noise_stds=[0.1, 0.2, 0.3, 0.4, 0.5], pos_noise_stds=[0.5, 1, 1.5, 2, 2.5])
    # aggregate_test_samples(100, {'grid': [0, 6, 50], 'lpt': [0, 6, 50, 50]}, use_preprocess=False, generate_default=False, ratio_false_stars=[0.1, 0.2, 0.3, 0.4, 0.5])
