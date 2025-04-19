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


def gen_pattern(meth_params: dict, coords: np.ndarray, ids: np.ndarray, img_id: str, h: int, w: int, f: float, gcata: pd.DataFrame, max_num_samp: int=20, realshot: bool=False):
    '''
        Generate the pattern with the given star coordinates for one star image.
    Args:
        meth_params: the parameters for the method
        coords: the coordinates of the stars
        ids: the ids of the stars
        scale: the scale used for region radius calculation
        img_id: the id of the image
        h: the height of the image
        w: the width of the image
        f: the focal length of the camera(in pixel)
        max_num_samp: the maximum number of stars to be used as reference star
    Return:
        patterns: the pattern generated
    '''
    # the dict to store the patterns
    pats_dict = defaultdict(list)

    if len(coords) < min_num_star:
        return pats_dict

    assert len(coords) == len(ids) and coords.shape[1] == 2, "The coordinates and ids are not matched."

    # distances between each stars
    # np.linalg.norm(stars[:,None,:] - stars[None,:,:], axis=-1)
    distances = cdist(coords, coords, 'euclidean')
    
    # angles with respect to the reference star
    angles = np.arctan2(coords[:, 1] - coords[:, 1][:, None], coords[:, 0] - coords[:, 0][:, None])
    
    # angular distances with respect to the origin
    n = len(coords)
    points = np.hstack([
        coords-np.array([h/2, w/2]),
        np.full((n, 1), f)
    ])
    norm = np.linalg.norm(points, axis=1)
    ang_dists = np.dot(points, points.T) / np.outer(norm, norm)

    # choose top max_num_samp brightest star as the reference star
    n = min(n, max_num_samp+1)
    for star_id, coord, ds, ags, ads in zip(ids[:n], coords[:n, :], distances[:n, :], angles[:n, :], ang_dists[:n, :]):
        # get catalogue index of the guide star
        if not realshot and star_id not in gcata['Star ID'].to_numpy():
            continue
        elif star_id in gcata['Star ID'].to_numpy():
            cata_idx = gcata[gcata['Star ID'] == star_id].index.to_list()[0]
        else:
            cata_idx = -1

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
            exc_cs, exc_ds, exc_ags = cs[(ads >= Rp) & (ads <= Rb)], ds[(ads >= Rp) & (ads <= Rb)], ags[(ads >= Rp) & (ads <= Rb)]
            assert len(exc_cs) == len(exc_ds) == len(exc_ags), "The number of stars in the region is not matched."
            if len(exc_cs) < min_num_star:
                continue

            # initialize the pattern
            pat = {
                'star_id': star_id,
                'cata_idx': cata_idx,
                'img_id': img_id
            }

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
                if exc_ds[-1] > border_d:
                    # print(exc_ds[0], border_d)
                    continue

                # rotate stars
                v1, v2 = exc_cs[0], np.array([0, 1])
                M = get_rotation_matrix(v1, v2)
                rot_cs = exc_cs @ M.T
                assert np.allclose(rot_cs[0], [0, np.linalg.norm(v1)], atol=1e-3), "Rotation matrix is not correct."
                
                # calculate the pattern
                grid = np.zeros((Lg, Lg), dtype=int)
                for c in rot_cs:
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
                grid = [int((d-Rb)/(Rp-Rb)*Nd)*Nt+int((t+np.pi)/(2*np.pi)*Nt) for d, t in zip(exc_ds, rot_ags)]
                pat['pat'] = ' '.join(map(str, grid))
            
            else:
                print('Invalid method')
    
            pats_dict[method].append(pat)

    return pats_dict


def gen_database(meth_params: dict, simu_params: dict, gcata_path: str, num_thd: int = 10):
    '''
        Generate the pattern database for the given star catalogue.
    '''
    
    def gen_sub_database(idxs: pd.Index):
        '''
            Generate the pattern database for the given star catalogue.
        '''

        db_dict = defaultdict(list)
        for star_id, ra, de in gcata.loc[idxs, ['Star ID', 'Ra', 'De']].to_numpy():
            star_id = int(star_id)

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

            # get the coordinates and ids of the stars
            ids = stars[:, 0]
            coords = stars[:, 1:3]

            # generate only the patterns for the given star
            ids[ids != star_id] = -1
        
            # generate the patterns
            pats_dict = gen_pattern(
                meth_params, 
                coords, 
                ids, 
                img_id=str(uuid.uuid1()),
                h=simu_params['h'], 
                w=simu_params['w'], 
                f=f, 
                gcata=gcata
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

    # number of round used for this method
    num_round = num_thd//len(meth_params)
    len_td = len(gcata)//num_round
    
    # add task
    for i in range(num_round):
        beg, end = i*len_td, min((i+1)*len_td, len(gcata))
        if beg >= end:
            continue
        tasks.append(pool.submit(gen_sub_database, gcata.index[beg:end]))
    
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
        max_num_samp: the maximum number of samples for each test image
            In other words, we can use the top num_samp brightest to construct pattern at most.
    Returns:
        dict: method->dataframe
    '''

    # generate right ascension[-pi, pi] and declination[-pi/2, pi/2]
    ras = np.random.uniform(0, 2*np.pi, num_img)
    des = np.arcsin(np.random.uniform(-1, 1, num_img))
    # ras, des = gcata[['Ra', 'De']].sample(num_img).to_numpy().T
    # 183987
    ras = np.full(num_img, 4.1837814)
    des = np.full(num_img, -0.4557763)
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
            coords_only=False
        )
        # get star ids
        ids = stars[:, 0]

        # two few guide stars to identify
        if len(np.intersect1d(ids, gcata['Star ID'].to_numpy())) <= min(min_num_star, 3):
            continue
    
        ids[ids != 183987] = -1

        # get the centroids of the stars in the image
        if False:
            coords = np.array(get_star_centroids(img))
        else:
            coords = stars[:, 1:3]

        # generate image id
        img_id = str(uuid.uuid1())

        # patterns for this image
        img_pats = gen_pattern(meth_params, coords, ids, img_id, simu_params['h'], simu_params['w'], f, gcata, max_num_samp=20)
        for method in img_pats:
            df_dict[method].extend(img_pats[method])

    # convert the results into dataframe
    for method in df_dict:
        df_dict[method] = pd.DataFrame(df_dict[method])
    
    return df_dict


def gen_real_sample(img_paths: list[str], meth_params: dict):
    '''
        Generate pattern match test case using real star image.
    '''

    patterns = defaultdict(list)
    for img_path in img_paths:
        # read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # get the centroids of the stars in the image
        coords = np.array(get_star_centroids(img, 'MEDIAN', 'Liebe', 'CCL', 'CoG', pixel_limit=3))
        # print(len(coords))
        # get star ids
        ids = np.full(len(coords), -1)
        # generate pattern
        img_patterns = gen_pattern(meth_params, coords, ids, img_id=os.path.basename(img_path), max_num_samp=20, realshot=True)
        for method in img_patterns:
            patterns[method].extend(img_patterns[method])
    
    # convert the results into dataframe
    for method in patterns:
        patterns[method] = pd.DataFrame(patterns[method])

    return patterns


def agg_sample(num_img: int, meth_params: dict, simu_params: dict, test_params: dict, gcata_path: str, num_thd: int = 20):
    '''
    Aggregate the test samples. 
    Args:
        num_img: number of test images expected to be generated
        meth_params: the parameters of methods, possible methods include:
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
        test_params: the parameters for the test sample generation
            'pos': the standard deviation of the positional noise
            'mag': the standard deviation of the magnitude noise
            'fs': the number of false stars
            'ms': the number of missing stars
        simu_params: the parameters for the simulation
            'h': the height of the image
            'w': the width of the image
            'fovx': the field of view in x direction
            'fovy': the field of view in y direction
            'limit_mag': the limit magnitude
            'pixel': the pixel size
        num_thd: the number of threads to generate the test samples
    '''

    if num_img == 0:
        return

    print('Test Image Generation')
    print('----------------------')
    print('Method Parameters:', meth_params)
    print('Simulation Parameters:', simu_params)
    print('Number of test images expected to be generated:', num_img)
    print('----------------------')

    # simulation config
    sim_cfg = f'{simu_params["h"]}_{simu_params["w"]}_{simu_params["fovx"]}_{simu_params["fovy"]}_{simu_params["limit_mag"]}'

    # read the guide star catalogue
    gcata_name = os.path.basename(gcata_path).rsplit('.', 1)[0]
    gcata = pd.read_csv(gcata_path, usecols=['Star ID', 'Ra', 'De', 'Magnitude'])

    # use thread pool
    pool = ThreadPoolExecutor(max_workers=num_thd)
    # tasks for later aggregation
    tasks = defaultdict(list)

    # add tasks to the thread pool
    for pos in test_params.get('pos', []):
        tasks[f'pos{pos}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, sigma_pos=pos))
    for mag in test_params.get('mag', []):
        tasks[f'mag{mag}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, sigma_mag=mag))
    for fs in test_params.get('fs', []):
        tasks[f'fs{fs}'].append(pool.submit(gen_sample, num_img, meth_params, simu_params, gcata, num_fs=fs))

    # sub test name
    for st_name in tasks:
        for task in tasks[st_name]:
            # get the async task result and store the returned dataframe
            df_dict = task.result()
            for method in df_dict:
                gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
                st_path = os.path.join('test', sim_cfg, method, gen_cfg, st_name)
                if not os.path.exists(st_path):
                    os.makedirs(st_path)
                df = df_dict[method]
                df.to_csv(os.path.join(st_path, str(uuid.uuid1())), index=False)

    # aggregate all the test patterns
    for method in meth_params:
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        path = os.path.join('test', sim_cfg, method, gen_cfg)
        if not os.path.exists(path):
            continue
        # sub test dir names
        test_names = list(tasks.keys())
        for tn in test_names:
            p = os.path.join(path, tn)
            dfs = [pd.read_csv(os.path.join(p, f)) for f in os.listdir(p) if f != 'labels.csv']
            if len(dfs) > 0:        
                df = pd.concat(dfs, ignore_index=True)
                df.to_csv(os.path.join(p, 'labels.csv'), index=False)
                # count the number of samples for each class
                print('Method and test name:', method, tn, '\nTotal number of images for this sub test', len(df['img_id'].unique()))

    pool.shutdown()

    return


if __name__ == '__main__':
    if False:
        gen_database(
            {
                'grid': [0.3, 6, 90], 
                # 'lpt': [0.3, 6, 50, 50]
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
            './catalogue/sao6.0_d0.2_12_12.csv',
        )

    if True:
        agg_sample(
            400, 
            {
                'grid': [0.3, 6, 90],
                # 'lpt': [0.3, 6, 50, 50],
                # 'lpt_nn': [6, 50],
                # 'rac_1dcnn': [6, [20, 50, 80], 16, 3]
            }, 
            {
                'h': 512,
                'w': 512,
                'fovx': 12,
                'fovy': 12,
                'limit_mag': 6
            },
            {
                'pos': [0, 1, 2, 3, 4], 
                # 'mag': [0, 0.1, 0.2, 0.3, 0.4], 
                # 'fs': [0, 1, 2, 3, 4]
            },
            './catalogue/sao6.0_d0.2_12_12.csv',
        )
    