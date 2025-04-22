import os
import uuid
import cv2
import numpy as np
import pandas as pd
from math import radians, tan, cos
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from scipy.spatial.distance import cdist

from simulate import create_star_image
from extract import get_star_centroids
from utils import get_angdist


# minimum number of stars in the region for pattern generation
min_num_star = 5


def get_rotation_matrix(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''
        Get the rotation matrix from v to w.
    '''
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    sin_theta = np.cross(v, w) / (norm_v * norm_w)

    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def gen_pattern(meth_params: dict, coords: np.ndarray, ids: np.ndarray, cata_idxs: np.ndarray, img_id: str, h: int, w: int, f: float, ra: float=None, de: float=None, roll: float=None, realshot: bool=False, max_num_samp: int=20):
    '''
        Generate the pattern with the given star coordinates for one star image.
    Args:
        meth_params: the parameters for the method
        coords: the coordinates of the stars
        ids: the ids of the stars
        cata_idxs: the catalogue index of the stars
        scale: the scale used for region radius calculation
        img_id: the id of the image
        h: the height of the image
        w: the width of the image
        f: the focal length of the camera(in pixel)
        ra/de/roll: the right ascension, declination and roll angle of the simulated star image
        max_num_samp: the maximum number of stars to be used as reference star
    Return:
        patterns: the pattern generated
    '''
    # the dict to store the patterns
    pats_dict = defaultdict(list)

    if len(coords) < min_num_star:
        return pats_dict

    assert len(coords) == len(ids) and coords.shape[1] == 2, "The coordinates and ids are not matched."
    n = len(coords)

    # distances between each stars
    # np.linalg.norm(stars[:,None,:] - stars[None,:,:], axis=-1)
    distances = cdist(coords, coords, 'euclidean')
    
    # angles with respect to the reference star
    angles = np.arctan2(coords[:, 1] - coords[:, 1][:, None], coords[:, 0] - coords[:, 0][:, None])
    
    # angular distances with respect to the origin
    points = np.hstack([
        coords-np.array([h/2, w/2]),
        np.full((n, 1), f)
    ])
    angdists = get_angdist(points)

    # choose top max_num_samp brightest star as the reference star
    n = min(n, max_num_samp+1) if realshot else n

    for star_id, cata_idx, coord, ds, ags, ads in zip(ids[:n], cata_idxs[:n], coords[:n], distances[:n], angles[:n], angdists[:n]):
        # get catalogue index of the guide star
        if not realshot and cata_idx == -1:
            # skip the star if it is not in the catalogue
            continue

        # coords, distance and angles are all sorted by angular distances with descending order
        # ! angle bigger, the angular distance(cos) is smaller
        cs, ds, ags = coords[np.argsort(ads)[::-1]], ds[np.argsort(ads)[::-1]], ags[np.argsort(ads)[::-1]]
        ads = np.sort(ads)[::-1]

        # remove the first element, which is reference star
        assert np.isclose(ads[0], 1) and np.allclose(cs[0], coord) and np.isclose(ds[0], 0) and np.isclose(ags[0], 0), "The first element is not the reference star."
        cs, ds, ags, ads = cs[1:], ds[1:], ags[1:], ads[1:]
        # calculate the relative coordinates
        cs = cs-coord

        for method in meth_params:
            # parse the buffer radius and pattern radius
            Rb, Rp = meth_params[method][:2]
            assert Rb < Rp, "Buffer radius should be smaller than pattern radius."
            
            # buffer and pattern radius in cos angular distance
            Rb, Rp = cos(radians(Rb)), cos(radians(Rp))

            # buffer and pattern radius in pixels
            # !the pattern radius is always half of the image size
            rb, rp = 0, min(h, w)/2

            # exclude stars outside the region
            exc_cs, exc_ds, exc_ags = cs[(ads > Rp) & (ads < Rb)], ds[(ads > Rp) & (ads < Rb)], ags[(ads > Rp) & (ads < Rb)]
            assert len(exc_cs) == len(exc_ds) == len(exc_ags), "The number of stars in the region is not matched."

            # ?due to the precision of the float, these assertions may fail
            # assert np.all((exc_cs > -rp) & (exc_cs < rp)), f"The coordinates of stars in the region is not in the range of [-rp, rp]. {exc_cs}"
            # assert np.all((exc_ds > rb) & (exc_ds < rp)), f"The distance of stars in the region is not in the range of [rb, rp]. {exc_ds}"
            if len(exc_cs) < min_num_star:
                continue

            # initialize the pattern
            pat = {
                'star_id': int(star_id),
                'cata_idx': int(cata_idx),
                'img_id': img_id,
                'row': coord[0],
                'col': coord[1],
            }

            # add the right ascension, declination and roll angle if they are given
            if ra is not None and de is not None and roll is not None:
                pat['ra'] = ra
                pat['de'] = de
                pat['roll'] = roll

            if method == 'rac_1dcnn':
                # parse the rest parameters
                arr_Nr, Ns, Nn = meth_params[method][2:]
                
                # construct radial features
                tot_Nr = 0
                for Nr in arr_Nr:
                    r_cnts, _ = np.histogram(exc_ds, bins=Nr, range=(rb, rp))
                    for i, rc in enumerate(r_cnts):
                        i += tot_Nr
                        pat[f'ring{i}'] = rc
                    tot_Nr += Nr
                
                # uses several neighbor stars as the starting angle to obtain the cyclic features
                for i, ag in enumerate(exc_ags[:Nn]):
                    # rotate stars
                    rot_ags = exc_ags - ag
                    # make sure all angles stay in [-pi, pi]
                    rot_ags %= 2*np.pi
                    rot_ags[rot_ags > np.pi] -= 2*np.pi
                    rot_ags[rot_ags < -np.pi] += 2*np.pi
                    assert np.all((rot_ags < np.pi) & (rot_ags > -np.pi))

                    # calculate sectore counts using histogram
                    s_cnts, _ = np.histogram(rot_ags, bins=Ns, range=(-np.pi, np.pi))
                    for j, sc in enumerate(s_cnts):
                        pat[f'n{i}_sector{j}'] = sc
                
                # add trailing zero if there is not enough neighbors
                if len(exc_ags) < Nn:
                    for i in range(len(exc_ags), Nn):
                        for j in range(Ns):
                            pat[f'n{i}_sector{j}'] = 0

            elif method == 'lpt_nn':
                # parse the rest parameters
                Nd = meth_params[method][2]

                # count the number of stars in each distance bin
                d_cnts, _ = np.histogram(ds, bins=Nd, range=(rb, rp))
                
                # normalize d_cnts
                for i, dc in enumerate(d_cnts):
                    pat[f'dist{i}'] = dc
                
            elif method == 'grid':
                # parse the rest parameter
                Lg = meth_params[method][2]

                # skip if the distance to nearest star is bigger than the distance of reference star to border
                border_d = min(
                    coord[0], coord[1],
                    h-coord[0], w-coord[1],
                )
                if exc_ds[0] > border_d:
                    continue

                # rotate stars
                v1, v2 = exc_cs[0], np.array([0, 1])
                M = get_rotation_matrix(v1, v2)
                rot_cs = exc_cs @ M.T
                assert np.allclose(rot_cs[0], [0, np.linalg.norm(v1)], atol=1e-3), "Rotation matrix is not correct."
                
                # calculate the pattern
                grid = np.zeros((Lg, Lg), dtype=int)
                for c in rot_cs:
                    # due to the percision of the float, the coords may be out of range
                    if c[0] < -rp or c[0] >= rp or c[1] < -rp or c[1] >= rp:
                        continue

                    row = int((c[0]/rp+1)/2*Lg)
                    col = int((c[1]/rp+1)/2*Lg)
                    grid[row][col] = 1
                
                # store the 1's position of grid
                pat['pat'] = ' '.join(map(str, np.flatnonzero(grid)))

            elif method == 'lpt':
                # parse the rest parameters
                Nd, Nt = meth_params[method][2:]

                # !log-polar transform is already done, where exc_ags is the thetas and exc_ds is the distances
                # rotate the stars, so that the nearest star's theta is 0
                rot_ags = exc_ags - exc_ags[0]
                # make sure the thetas are in the range of [-pi, pi]
                rot_ags %= 2*np.pi
                rot_ags[rot_ags >= np.pi] -= 2*np.pi
                rot_ags[rot_ags < -np.pi] += 2*np.pi
                
                # generate the grid
                grid = []
                for d, t in zip(exc_ds, rot_ags):
                    # due to the percision of the float, the distance may be out of range
                    if d < rb or d > rp:
                        continue

                    # get the grid index
                    i = int(d/rp*Nd)
                    j = int((t+np.pi)/(2*np.pi)*Nt)
                    
                    grid.append(i*Nt+j)
                    assert(i < Nd and j < Nt), f"Grid index out of range: {i}, {j}, {Nd}, {Nt}, {exc_ds[-1]}, {ads[-1]}, "
                
                pat['pat'] = ' '.join(map(str, grid))
            
            else:
                print('Invalid method')
    
            pats_dict[method].append(pat)

    return pats_dict


def gen_database(meth_params: dict, simu_params: dict, gcata_path: str, num_thd: int = 10):
    '''
        Generate the pattern database for the given star catalogue.
    '''
    
    def gen_sub_database(idxs: np.ndarray):
        '''
            Generate the pattern database for the given star catalogue.
        '''

        db_dict = defaultdict(list)
        id_ra_des = gcata.loc[idxs, ['Star ID', 'Ra', 'De']].to_numpy()
        for cata_idx, (star_id, ra, de) in zip(idxs, id_ra_des):
            _, stars = create_star_image(
                ra, de, 0, 
                h=simu_params['h'],
                w=simu_params['w'],
                fovx=simu_params['fovx'],
                fovy=simu_params['fovy'],
                limit_mag=simu_params['limit_mag'],
                sigma_pos=simu_params['sigma_pos'],
                sigma_mag=simu_params['sigma_mag'],
                num_fs=simu_params['num_fs'],
                num_ms=simu_params['num_ms'],
                coords_only=True
            )

            # get the coordinates, ids and catalogue indexs of the stars within the image
            coords = stars[:, 1:3]
            ids = stars[:, 0]
            cata_idxs = np.full_like(ids, cata_idx)
            
            # generate only the patterns for the given star
            cata_idxs[ids != star_id] = -1
            ids[ids != star_id] = -1
        
            # generate the patterns
            pats_dict = gen_pattern(
                meth_params, 
                coords, 
                ids, 
                cata_idxs,
                img_id=str(uuid.uuid1()),
                h=simu_params['h'], 
                w=simu_params['w'], 
                f=f, 
            )
            
            for method in pats_dict:
                assert len(pats_dict[method]) == 1

                # get the pattern for the given star
                pat = pats_dict[method][0]
                assert pat['star_id'] == star_id

                # process pattern to generate a database entry(lookup table)
                pat = pat['pat'].split(' ')
                
                db_entry = {}
                for p in pat:
                    db_entry[int(p)] = star_id
                db_dict[method].append(db_entry)

        return db_dict
    
    if meth_params == {}:
        return

    # read the star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])

    # simulation config
    sim_cfg = f'{simu_params["h"]}_{simu_params["w"]}_{simu_params["fovx"]}_{simu_params["fovy"]}_{simu_params["limit_mag"]}'

    # focus in pixels used to calculate the buffer and pattern radius
    f1 = simu_params['w'] / (2 * tan(radians(simu_params['fovx'] / 2)))
    f2 = simu_params['h'] / (2 * tan(radians(simu_params['fovy'] / 2)))
    assert np.isclose(f1, f2), "The focal length in x and y direction are not equal."
    f = (f1 + f2) / 2

    print('Database Generation')
    print('-------------------')
    print('Method parameters:', meth_params)
    print('Simulation parameters:', simu_params)
    print('Guide star catalogue:', gcata_path)
    
    # use thread pool to generate the database
    pool = ThreadPoolExecutor(max_workers=num_thd)
    tasks = []

    # length of sub database for each thread
    len_thd = len(gcata)//(num_thd-1)
    
    # add task
    for i in range(num_thd):
        beg, end = i*len_thd, min((i+1)*len_thd, len(gcata))
        if beg >= end:
            break
        tasks.append(pool.submit(gen_sub_database, np.arange(beg, end, dtype=int)))
    
    dfs_dict = defaultdict(list)
    # wait all tasks to be done and merge all the results
    for task in tasks:
        db_dicts = task.result()

        for method in db_dicts:
            dfs_dict[method].append(pd.DataFrame(db_dicts[method]))
        
    # merge the results
    for method in dfs_dict:
        # generation config for each method
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))

        # default noise config for each method
        noise_cfg = f'{simu_params["sigma_pos"]}_{simu_params["sigma_mag"]}_{simu_params["num_fs"]}_{simu_params["num_ms"]}'

        # make directory to store the database
        db_dir = os.path.join('database', sim_cfg, method, gen_cfg, noise_cfg)
        os.makedirs(db_dir, exist_ok=True)

        # save the database
        df = pd.concat(dfs_dict[method], ignore_index=True)
        df.to_csv(os.path.join(db_dir, 'db.csv'), index=False)

    pool.shutdown()

    return 


def gen_dataset(meth_params: dict, simu_params: dict, ds_paths: dict, star_id: int, cata_idx: int, star_ra: float, star_de: float, offset :float, num_roll: int):
    '''
        Generate dataset for NN model using the given star catalogue.
    '''
    # skip if the method is already generated
    nmeth_params = {}
    for method in meth_params:
        # only generate the dataset for the given star id
        ds_path = os.path.join(ds_paths[method], f'{star_id}')
        if os.path.exists(ds_path) and any(file.endswith(f'{num_roll}.csv') for file in os.listdir(ds_path)):
            continue
        nmeth_params[method] = meth_params[method]

    if nmeth_params == {}:
        return

    # csv file name
    name = uuid.uuid1()

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    ras = np.clip(np.full(num_roll, star_ra) + np.radians(np.random.normal(0, offset, num_roll)), 0, 2*np.pi)
    des = np.clip(np.full(num_roll, star_de) + np.radians(np.random.normal(0, offset, num_roll)), -np.pi/2, np.pi/2)
    rolls = np.arange(0, 2*np.pi, 2*np.pi/num_roll)

    # the dict to store the patterns(method -> [patterns])
    df_dict = defaultdict(list)

    # focus in pixels used to calculate the buffer and pattern radius
    f1 = simu_params['w'] / (2 * tan(radians(simu_params['fovx'] / 2)))
    f2 = simu_params['h'] / (2 * tan(radians(simu_params['fovy'] / 2)))
    assert np.isclose(f1, f2), f"The focal length in x direction is {f1}, while in y direction is {f2}."
    f = (f1+f2)/2

    # generate the star image
    for ra, de, roll in zip(ras, des, rolls):
        # stars is a np array [[id, row, col, mag]]
        _,  stars = create_star_image(ra, de, roll, 
            h=simu_params['h'], 
            w=simu_params['w'],
            fovx=simu_params['fovx'],
            fovy=simu_params['fovy'],
            # pixel=simu_params['pixel'],
            limit_mag=simu_params['limit_mag'],
            sigma_pos=simu_params['sigma_pos'],
            sigma_mag=simu_params['sigma_mag'],
            num_fs=simu_params['num_fs'],
            num_ms=simu_params['num_ms'], 
            coords_only=False
        )

        # get star ids and coordinates
        ids = stars[:, 0]
        coords = stars[:, 1:3]
        
        # set all the star ids and catalogue indexs to -1 except the given star id in order to make sure only generate the pattern for the given star id
        cata_idxs = np.full_like(ids, cata_idx)
        cata_idxs[id != star_id] = -1
        ids[ids != star_id] = -1
        
        if np.all(ids == -1):
            continue

        # generate a unique img id for later accuracy calculation
        img_id = str(uuid.uuid1())

        # generate the pattern(dict, method->pattern) for the given star id
        pat_dict = gen_pattern(
            nmeth_params, 
            coords, 
            ids,
            cata_idxs, 
            img_id, 
            h=simu_params['h'], 
            w=simu_params['w'], 
            f=f,
            ra=ra,
            de=de,
            roll=roll,
        )

        # store the patterns
        for method, pat in pat_dict.items():
            assert len(pat) == 1, f"Error: {len(pat)} patterns generated for method {method}."
            pat = pat[0]
            df_dict[method].append(pat)

    for method in df_dict:
        if len(df_dict[method]) == 0:
            continue
        # make directory
        ds_path = os.path.join(ds_paths[method], f'{star_id}')
        os.makedirs(ds_path, exist_ok=True)

        # save the dataset
        df_dict[method] = pd.DataFrame(df_dict[method])
        df_dict[method].to_csv(os.path.join(ds_path, f'{name}_{num_roll}.csv'), index=False)

    return


def gen_sample(num_img: int, meth_params: dict, simu_params: dict, gcata: pd.DataFrame, sigma_pos: float=0.0, sigma_mag: float=0.0, num_fs: int=0, num_ms: int=0):
    '''
        Generate test samples.
    Args:
        num_img: number of test images expected to be generated
        meth_params: the parameters for the test sample generation
            'rac_1dcnn':
                r: the radius of the region in degrees
                arr_Nr: the array of ring number
                Ns: the number of sectors
                Nn: the minimum number of neighbor stars in the region
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
        sigma_pos: the standard deviation of the positional noise
        sigma_mag: the standard deviation of the magnitude noise
        num_fs: the number of false stars
        num_ms: the number of missing stars
    Returns:
        dict: method->dataframe
    '''

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    ras = np.random.uniform(0, 2*np.pi, num_img)
    des = np.arcsin(np.random.uniform(-1, 1, num_img))
    rolls = np.random.uniform(0, 2*np.pi, num_img)

    # focus in pixels used to calculate the buffer and pattern radius
    f1 = simu_params['w'] / (2 * tan(radians(simu_params['fovx'] / 2)))
    f2 = simu_params['h'] / (2 * tan(radians(simu_params['fovy'] / 2)))
    assert np.isclose(f1, f2), "The focal length in x and y direction are not equal."
    f = (f1+f2)/2

    # the dict to store the results
    df_dict = defaultdict(list)

    # generate the star image
    for ra, de, roll in zip(ras, des, rolls):
        img, stars = create_star_image(ra, de, roll, 
            h=simu_params['h'], 
            w=simu_params['w'],
            fovx=simu_params['fovx'],
            fovy=simu_params['fovy'],
            # pixel=simu_params['pixel'],
            limit_mag=simu_params['limit_mag'],
            sigma_pos=sigma_pos,
            sigma_mag=sigma_mag,
            num_fs=num_fs,
            num_ms=num_ms, 
            coords_only=True
        )

        # get the centroids of the stars in the image
        if False:
            coords = np.array(get_star_centroids(img))
        else:
            coords = stars[:, 1:3]

        # get star ids
        ids = stars[:, 0]

        # find the catalogue index of all the guide stars in the image
        cata_idxs = np.full_like(ids, -1)

        # intersect the ids with the catalogue, and return the common ids and their indexs
        intersect_ids, ids_idxs, gcata_idxs = np.intersect1d(ids, gcata['Star ID'].to_numpy(), return_indices=True)

        # two few guide stars to identify
        if len(intersect_ids) < min(min_num_star, 3):
            continue

        # set the catalogue indexs
        cata_idxs[ids_idxs] = gcata_idxs

        # generate image id
        img_id = str(uuid.uuid1())

        # patterns for this image
        pats_dict = gen_pattern(
            meth_params, 
            coords, 
            ids,
            cata_idxs,
            img_id, 
            h=simu_params['h'], 
            w=simu_params['w'],
            f=f, 
            ra=ra,
            de=de,
            roll=roll,
        )
        for method in pats_dict:
            df_dict[method].extend(pats_dict[method])

    # convert the results into dataframe
    for method in df_dict:
        df_dict[method] = pd.DataFrame(df_dict[method])
    
    return df_dict


def gen_real_sample(img_paths: list[str], meth_params: dict, f: float):
    '''
        Generate pattern match test case using real star image.
    '''

    df_dict = defaultdict(list)
    for img_path in img_paths:
        # read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # get the image size
        h, w = img.shape

        # get the centroids of the stars in the image
        coords = np.array(get_star_centroids(img, 'MEDIAN', 'Liebe', 'CCL', 'CoG', pixel_limit=3))

        # set all star ids and catalogue indexs to -1, since unknown for real image
        ids = np.full(len(coords), -1)
        cata_idxs = np.full(len(coords), -1)

        print(len(coords), 'stars in the image')

        # generate pattern for each img
        pats_dict = gen_pattern(
            meth_params,
            coords, 
            ids,
            cata_idxs,
            img_id=os.path.basename(img_path),
            h=h,
            w=w,
            f=f,
            max_num_samp=20,
            realshot=True
        )
        
        # merge the patterns
        for method in pats_dict:
            df_dict[method].extend(pats_dict[method])
    
    # convert the results into dataframe
    for method in df_dict:
        df_dict[method] = pd.DataFrame(df_dict[method])

    return df_dict


if __name__ == '__main__':
    if True:
        gen_database(
            {
                'grid': [0.5, 6, 75], 
                # 'lpt': [0.3, 6, 25, 36]
            },
            {
                'h': 512,
                'w': 512,
                'fovx': 12,
                'fovy': 12,
                'limit_mag': 6,
                'sigma_pos': 0,
                'sigma_mag': 0,
                'num_fs': 0,
                'num_ms': 0,
            },
            './catalogue/sao6.0_d0.03_12_15.csv',
        )

    