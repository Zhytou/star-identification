import os
import uuid
import re
import numpy as np
import pandas as pd
from math import radians, tan
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.spatial.distance import cdist

from simulate import create_star_image
from extract import get_star_centroids

# simulation configuration
h = w = 512
fov = 15
limit_mag = 6.5
f = 5.8e-3
pixel = 2*tan(radians(fov/2))*f/h

sim_cfg = f'{h}_{w}_{fov}_{limit_mag}'

# guide star catalogue for pattern match database and nn dataset generation
gcata_path = 'catalogue/sao5.6_d0.2.csv'
# use for generation config
gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
# guide star catalogue
gcatalogue = pd.read_csv(gcata_path, usecols= ["Star ID", "Ra", "De", "Magnitude"])

# number of reference star
num_class = len(gcatalogue)

# minimum number of stars for smaple and pattern generation
min_num_star = 7

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
        for star_id, ra, de in zip(gcatalogue.loc[idxs, 'Star ID'], gcatalogue.loc[idxs, 'Ra'], gcatalogue.loc[idxs, 'De']):
            img, stars = create_star_image(ra, de, 0, h=h, w=w, fov=fov, limit_mag=limit_mag, coords_only=not use_preprocess)
            # get the centroids of the stars in the image
            if use_preprocess:
                coords = np.array(get_star_centroids(img))
            else:
                coords = stars[:, 1:3]

            idx = np.where(stars[:, 0] == star_id)[0][0]
            if (coords[idx][0] - h/2)**2 + (coords[idx][1] - w/2)**2 > 0.1:
                print('The star is not in the center of the image!')
                continue
            # calculate the relative coordinates
            coords = coords - coords[idx]
            # calculate the distance between the star and the center of the image
            distances = np.linalg.norm(coords, axis=1)
            # sort the stars by distance with accending order
            coords = coords[distances.argsort()]
            distances = np.sort(distances)
            # exclude the reference star (h/2, w/2)
            coords, distances = coords[1:], distances[1:]
            
            if method == 'grid':
                # exclude stars out of region
                coords = coords[(distances >= Rb) & (distances <= Rp)] 
                if len(coords) < min_num_star:
                    continue
                # find the nearest neighbor star
                nearest_coord = coords[0]
                # calculate rotation angle & matrix
                angle = np.arctan2(nearest_coord[1], nearest_coord[0])
                M = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                rotated_coords = coords @ M
                assert round(rotated_coords[0][1])==0
                # calculate the pattern
                grid = np.zeros((Lg, Lg), dtype=int)
                for coord in rotated_coords:
                    row = int((coord[0]/Rp+1)/2*Lg)
                    col = int((coord[1]/Rp+1)/2*Lg)
                    grid[row][col] = 1
                # store the 1's position of the grid
                database.append({
                    'pattern': ' '.join(map(str, np.flatnonzero(grid))),
                    'id': star_id
                })
            elif method == 'lpt':
                # exclude stars out of region
                coords = coords[(distances >= Rb) & (distances <= Rp)]
                distances = distances[(distances >= Rb) & (distances <= Rp)]
                if len(coords) < min_num_star:
                    continue
                # do log-polar transform 
                # distance = ln(sqrt(y^2+x^2)), theta = actan(y/x)
                thetas = np.arctan2(coords[:, 1], coords[:, 0])
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
            Rb, Rp = tan(radians(rb))*f/pixel, tan(radians(rp))*f/pixel
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{Lg}'
        elif method == 'lpt':
            # parse parameters: buffer radius, pattern radius, number of distance bins and number of theta bins
            rb, rp, Nd, Nt = gen_params[method]
            # radius in pixels
            Rb, Rp = tan(radians(rb))*f/pixel, tan(radians(rp))*f/pixel
            path = f'{database_path}/{gcata_name}_{int(use_preprocess)}_{rb}_{rp}_{Nd}_{Nt}'
        elif method == 'rac':
            # parse parameters: buffer radius, radial radius, cyclic radius, number of rings
            rb, rr, rc, N = gen_params[method]
            # radius in pixels
            Rb, Rr, Rc = tan(radians(rb))*f/pixel, tan(radians(rr))*f/pixel, tan(radians(rc))*f/pixel
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


def generate_nn_dataset(method: str, gen_params: list, mode: str, num_vec: int, idxs: list, use_preprocess: bool, sigma_pos: float, sigma_mag: float, num_fs: int, num_ms: int):
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
        sigma_pos: the standard deviation of the positional noise
        sigma_mag: the standard deviation of the magnitude noise
        num_fs: the number of false stars
        num_ms: the number of missing stars
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
        ras = np.clip(gcatalogue.loc[idxs, 'Ra']+np.radians(np.random.normal(0, 5, len(idxs))), 0, 2*np.pi)
        des = np.clip(gcatalogue.loc[idxs, 'De']+np.radians(np.random.normal(0, 5, len(idxs))), -np.pi/2, np.pi/2)
    else:
        print('Invalid mode!')
        return

    # generate the star image
    for ra, de in zip(ras, des):
        # stars is a np array [[id, row, col, mag]]
        img, stars = create_star_image(ra, de, 0, h=h, w=w, fov=fov, limit_mag=limit_mag, sigma_pos=sigma_pos, sigma_mag=sigma_mag, num_fs=num_fs, num_ms=num_ms, coords_only=not use_preprocess)
        # get star ids and coordinates
        ids = stars[:, 0]
        coords = stars[:, 1:3]
        
        # get the centroids of the stars in the image
        if use_preprocess:
            coords = np.array(get_star_centroids(img))
        
        if len(coords) < 4:
            continue

        # generate a unique img id for later accuracy calculation
        img_id = uuid.uuid1()

        # distances and angles between each star in FOV
        distances = cdist(coords, coords, 'euclidean')
        angles = np.arctan2(coords[:, 1] - coords[:, 1][:, None], coords[:, 0] - coords[:, 0][:, None])

        # choose a guide star as the reference star
        for star_id, coord, ds, ags in zip(ids, coords, distances, angles):
            # check if false star or star not in guide catalogue
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
                R = tan(radians(r))*f/pixel
                if coord[0] < R/2 or coord[0] > h-R/2 or coord[1] < R/2 or coord[1] > w-R/2:
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
                R = tan(radians(r))*f/pixel
                if coord[0] < R/2 or coord[0] > h-R/2 or coord[1] < R/2 or coord[1] > w-R/2:
                    continue
                # exclude angles and distances outside the region
                ags = ags[ds <= R]
                ds = ds[ds <= R]
                # skip if only the reference star in the region
                if len(ags) == 0:
                    continue
                for i, seq in enumerate([ds, ags]):
                    # discretize the feature sequence(distances and angles) with different levels
                    tot_N = 0
                    for N in arr_N:
                        # density is set True
                        if i == 0:
                            pdf, _ = np.histogram(seq, bins=N, range=(0, R), density=True)
                        elif i == 1:
                            pdf, _ = np.histogram(seq, bins=N, range=(-np.pi, np.pi), density=True)
                        else:
                            print("wrong index")
                        for j, p in enumerate(pdf):
                            j += tot_N
                            label[f's{i}_feat{j}'] = p
                        tot_N += N
                    # statistics of distances and angles
                    stats = [np.min(seq), np.max(seq), np.median(seq), np.mean(seq)]
                    for j, stat in enumerate(stats):
                        j += tot_N
                        label[f's{i}_feat{j}'] = stat
                labels.append(label)
            elif method == 'lpt_nn':
                # parse the parameters:
                r, Nd= gen_params
                # calculate the radius in pixels
                R = tan(radians(r))*f/pixel
                if coord[0] < R/2 or coord[0] > h-R/2 or coord[1] < R/2 or coord[1] > w-R/2:
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


def generate_test_samples(num_vec: int, gen_params: dict, use_preprocess: bool, sigma_pos: float=0, sigma_mag: float=0, num_fs: int=0, num_ms: int=0):
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
        sigma_pos: the standard deviation of the positional noise
        sigma_mag: the standard deviation of the magnitude noise
        num_fs: the number of false stars
        num_ms: the number of missing stars
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
        img, stars = create_star_image(ra, de, 0, h=h, w=w, fov=fov, limit_mag=limit_mag, sigma_pos=sigma_pos, sigma_mag=sigma_mag, num_fs=num_fs, num_ms=num_ms, coords_only=not use_preprocess)
        # get star coordinates
        ids = stars[:, 0]
        coords = stars[:, 1:3]

        # get the centroids of the stars in the image
        if use_preprocess:
            stars = np.array(get_star_centroids(img))

        # too few stars for quest algorithm to identify satellite attitude
        if len(stars) < min_num_star:
            continue
        
        # generate a unique img id for later accuracy calculation
        img_id = uuid.uuid1()

        # distances = np.linalg.norm(stars[:,None,:] - stars[None,:,:], axis=-1)
        distances = cdist(coords, coords, 'euclidean')
        angles = np.arctan2(coords[:, 1] - coords[:, 1][:, None], coords[:, 0] - coords[:, 0][:, None])
        # choose a guide star as the reference star
        for star_id, coord, ds, ags in zip(ids, coords, distances, angles):
            # check if false star or not in guide star catalogue
            if star_id == -1 or star_id not in gcatalogue['Star ID'].values:
                continue

            # coords and angles are both sorted by distance with accending order
            cs, ags = coords[np.argsort(ds)], ags[np.argsort(ds)]
            ds = np.sort(ds)
            # remove the first element, which is reference star
            assert coord[0] == cs[0][0] and coord[1] == cs[0][1] and ds[0] == 0 and ags[0] == 0
            cs, ds, ags = cs[1:], ds[1:], ags[1:]
            # calculate the relative coordinates
            cs = cs-coord
            if len(cs) < min_num_star:
                continue
            
            methods = list(gen_params.keys())
            for method in methods:
                if method == 'rac_1dcnn':
                    # parse the parameters
                    r, arr_Nr, Ns, Nn = gen_params[method]
                    # radius in pixels
                    R = tan(radians(r))*f/pixel
                    if coord[0] < R/2 or coord[0] > h-R/2 or coord[1] < R/2 or coord[1] > w-R/2:
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
                    R = tan(radians(r))*f/pixel
                    if coord[0] < R/2 or coord[0] > h-R/2 or coord[1] < R/2 or coord[1] > w-R/2:
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
                    if len(excl_ags) == 0:
                        continue

                    for i, seq in enumerate([excl_ds, excl_ags]):
                        # discretize the feature sequence(distances and angles) with different levels
                        tot_N = 0
                        for N in arr_N:
                            # density is set True
                            if i == 0:
                                pdf, _ = np.histogram(seq, bins=N, range=(0, R), density=True)
                            elif i == 1:
                                pdf, _ = np.histogram(seq, bins=N, range=(-np.pi, np.pi), density=True)
                            else:
                                print("wrong index")
                            for j, p in enumerate(pdf):
                                j += tot_N
                                pattern[f's{i}_feat{j}'] = p
                            tot_N += N
                        # statistics of distances and angles
                        stats = [np.min(seq), np.max(seq), np.median(seq), np.mean(seq)]
                        for j, stat in enumerate(stats):
                            j += tot_N
                            pattern[f's{i}_feat{j}'] = stat
                    patterns[method].append(pattern)
                elif method == 'lpt_nn':
                    # parse the parameters
                    r, Nd = gen_params[method]
                    # radius in pixels
                    R = tan(radians(r))*f/pixel
                    if coord[0] < R/2 or coord[0] > h-R/2 or coord[1] < R/2 or coord[1] > w-R/2:
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
                    Rb, Rp = tan(radians(rb))*f/pixel, tan(radians(rp))*f/pixel
                    if coord[0] < Rp/2 or coord[0] > h-Rp/2 or coord[1] < Rp/2 or coord[1] > w-Rp/2:
                        continue
                    # exclude stars outside the region
                    excl_cs, excl_ags = cs[(ds >= Rb) & (ds <= Rp)], ags[(ds >= Rb) & (ds <= Rp)]
                    if len(excl_cs) < min_num_star:
                        continue
                    M = np.array([[np.cos(excl_ags[0]), -np.sin(excl_ags[0])], [np.sin(excl_ags[0]), np.cos(excl_ags[0])]])
                    # rotate stars
                    rot_cs = excl_cs @ M
                    assert round(rot_cs[0][1])==0
                    # calculate the pattern
                    grid = np.zeros((Lg, Lg), dtype=int)
                    for c in rot_cs:
                        row = int((c[0]/Rp+1)/2*Lg)
                        col = int((c[1]/Rp+1)/2*Lg)
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
                    Rb, Rp = tan(radians(rb))*f/pixel, tan(radians(rp))*f/pixel
                    # exclude stars outside the region
                    excl_ags, excl_ds = ags[(ds >= Rb) & (ds <= Rp)], ds[(ds >= Rb) & (ds <= Rp)]
                    if len(excl_ags) < min_num_star:
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
                    Rb, Rr, Rc = tan(radians(rb))*f/pixel, tan(radians(rr))*f/pixel, tan(radians(rc))*f/pixel

                    # count the number of stars in each ring
                    r_cnts, _ = np.histogram(ds, bins=N, range=(Rb, Rr))
                    # generate radial pattern 01 sequence
                    r_pattern = np.zeros(N, dtype=int)
                    r_pattern[np.nonzero(r_cnts)] = 1

                    # exclude stars outside the region
                    pm2_ags = ags[(ds >= Rb) & (ds <= Rc)]
                    # rotate the stars until the nearest star lies on the horizontal axis
                    if len(pm2_ags) < min_num_star:
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


def aggregate_nn_dataset(types: dict, gen_params: dict, use_preprocess: bool, default_ratio: float, sigma_pos: list=[], sigma_mag: list=[], num_fs: list=[], num_ms: list=[], num_thread: int=40, fine_grained: bool=False):
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
        sigma_pos: the standard deviation of the positional noise
        sigma_mag: the standard deviation of the magnitude noise
        num_fs: the number of false stars
        num_ms: the number of missing stars
        num_thread: the number of threads to generate the dataset
    '''

    def wait_tasks(tasks: dict, root_dirs: dict, file_name: str, col_name: str=None):
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
            pos, mag, fs, ms
        '''
        pos, mag, fs, ms = 0, 0, 0, 0
        match = re.match('.+\/(pos|mag|fs|ms)([0-9]+\.?[0-9]*)', s)
        if match:
            test_type, number = match.groups()
            if test_type == 'pos':
                pos = float(number)
            elif test_type == 'mag':
                mag = float(number)
            elif test_type == 'fs':
                fs = int(number)
            else:  # test_type == 'ms'
                ms = int(number)
        return pos, mag, fs, ms

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
    
    # generate config for each sub dataset
    num_noised_dataset = len(sigma_pos)+len(sigma_mag)+len(num_fs)+len(num_ms)
    if num_noised_dataset > 0:
        noised_ratio = (1-default_ratio)/num_noised_dataset
    else:
        noised_ratio = 0
    names = list(types.keys())
    for name in names:
        num_samples = types.pop(name)
        num_nosied_samples = max(int(num_samples*noised_ratio), 1)
        num_default_samples = max(int(num_samples*default_ratio), 1)

        for pos in sigma_pos:
            types[f'{name}/pos{pos}'] = num_nosied_samples
        for mag in sigma_mag:
            types[f'{name}/mag{mag}'] = num_nosied_samples
        for fs in num_fs:
            types[f'{name}/fs{fs}'] = num_nosied_samples
        for ms in num_ms:
            types[f'{name}/ms{ms}'] = num_nosied_samples
        types[f'{name}/default'] = num_default_samples
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
            pos, mag, fs, ms = parse_params(key)

            # roughly generate the samples for each class        
            num_round = min(num_thread//4, int(avg_num), int((1-pct)*len(gcatalogue)))
            num_vec = 1000
            print('random generate round: ', num_round)
            for _ in range(num_round):
                task = pool.submit(generate_nn_dataset, method, gen_params[method], 'random', num_vec, [], use_preprocess, pos, mag, fs, ms)
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
            print(method, key, len(df), len(idxs), idxs[:3], idxs[-3:])
            # parse parameters
            pos, mag, fs, ms = parse_params(key)

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
                task = pool.submit(generate_nn_dataset, method, gen_params[method], 'supplementary', 0, idxs[beg: end], use_preprocess, pos, mag, fs, ms)
                tasks[method][key].append(task)
        
    wait_tasks(tasks, root_dirs, 'labels.csv', 'cata_idx')
    
    # aggregate the labels.csv in root directory
    aggregate_root_dir(root_dirs)


def aggregate_test_samples(num_vec: int, gen_params: dict, use_preprocess: bool=False, default: bool=True, sigma_pos: list=[], sigma_mag: list=[], num_fs: list=[], num_ms: list=[], num_thread: int = 20):
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
        default: whether to generate test samples for default case
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
        if default:
            tasks['default'].append(task)
        for pos in sigma_pos:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, sigma_pos=pos)
            tasks[f'pos{pos}'].append(task)
        for mag in sigma_mag:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, sigma_mag=mag)
            tasks[f'mag{mag}'].append(task)
        for fs in num_fs:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, num_fs=fs)
            tasks[f'fs{fs}'].append(task)
        for ms in num_ms:
            task = pool.submit(generate_test_samples, num_vec, gen_params, use_preprocess, num_ms=ms)
            tasks[f'ms{ms}'].append(task)

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
    generate_pm_database({'grid': [0.3, 6, 50]})
    # generate_pm_database({'lpt': [0, 6, 50, 50]})
    # aggregate_nn_dataset({'train': 1, 'validate': 1, 'test': 1}, {'rac_1dcnn': [6, [20, 50, 80], 16, 3]}, use_preprocess=False, default_ratio=0.7, sigma_pos=[3], sigma_mag=[0.2], num_fs=[3], fine_grained=False, num_thread=20)
    # aggregate_nn_dataset({'train': 1, 'validate': 1, 'test': 1}, {'lpt_nn': [6, 50]}, use_preprocess=False, default_ratio=0.7, sigma_pos=[3], sigma_mag=[0.2], num_fs=[3], fine_grained=True, num_thread=20)
    aggregate_test_samples(300,  {'grid': [0.3, 6, 50]}, use_preprocess=False, sigma_pos=[0.5, 1, 1.5, 2, 2.5], default=False)
    # aggregate_test_samples(300,  {'lpt': [0, 6, 50, 50]}, use_preprocess=False, sigma_mag=[0.05, 0.1, 0.15, 0.2, 0.25], sigma_pos=[0.5, 1, 1.5, 2, 2.5], default=False)
    # aggregate_test_samples(300,  {'rac_1dcnn': [6, [10, 20, 30], 16, 3], 'lpt_nn': [6, 20]}, use_preprocess=False, sigma_mag=[0.05, 0.1, 0.15, 0.2, 0.25], sigma_pos=[0.5, 1, 1.5, 2, 2.5], default=False)
