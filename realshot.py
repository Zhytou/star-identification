import os, cv2, h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from generate import gen_real_sample, setup
from model import create_model
from dataset import create_dataset
from test import predict, verify, postprocess
from utils import label_star_image, traid



def load_h5data(dir: str, name: str):
    '''
        Load h5 data.
    '''
    with h5py.File(os.path.join(dir, name), 'r') as f:
        coords = f['/coords'][:].T
        # remove the 1 demension
        img_names = np.squeeze(f['/names'][:].T)
        cnts = np.squeeze(f['/cnts'][:].astype(np.int32).T)
        ids = np.squeeze(f['/ids'][:].astype(np.int32).T)

    start = 0
    data = []
    for i, cnt in enumerate(cnts):
        end = start + cnt
        data.append({
            'path': os.path.join(dir, str(img_names[i], 'UTF-8')),
            'coords': coords[start:end],
            'ids': ids[start:end]
        })
        start = end
    
    return data


def cal_attitude(cata: pd.DataFrame, coords: np.ndarray, ids: np.ndarray, h: int, w: int, f: int, grays: np.ndarray=None):
    '''
        Calculate the attitude matrix of star sensor.
    '''
    assert len(coords) == len(ids)
    n = len(ids)
    # get the view vectors
    vvs = np.full((n, 3), f)
    vvs[:, 0] = coords[:, 1]-w/2
    vvs[:, 1] = coords[:, 0]-h/2

    # get the refernce vectors
    stars = cata[cata['Star ID'].isin(ids)].copy()
    stars['X'] = np.cos(stars['Ra'])*np.cos(stars['De'])
    stars['Y'] = np.sin(stars['Ra'])*np.cos(stars['De'])
    stars['Z'] = np.sin(stars['De'])
    #! careful! use index to sort the stars, because the output of isin is not sorted by the given list
    stars = stars.set_index('Star ID').loc[ids]
    rvs = stars[['X', 'Y', 'Z']].to_numpy()

    return traid(vvs.T, rvs.T)


def identify_realshot_by_nn(img_paths: list[str], simu_params: dict, meth_params: dict, extr_params: dict, model_types: dict, gcata_path: str, device: str=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), eps1: float=1e-4, eps2: float=1e-3, output_dir: str=None):
    '''
        Identify realshot by nn method.
    '''

    # generate test samples for realshot
    df_dict = gen_real_sample(
        img_paths,
        meth_params,
        extr_params,
        f=simu_params['f'],
    )

    for method in meth_params:
        # setup all the configs
        sim_cfg, _, gcata_name, gcata = setup(simu_params, gcata_path)
        gen_cfg = f'{gcata_name}_'+'_'.join(map(str, meth_params[method]))
        ext_cfg = '_'.join(map(str, extr_params.values()))

        # print info
        print(
            'Realshot Test',
            '\n-----------------------',
            '\nMETHOD INFO',
            '\nMethod:', method,
            '\nSimulation config:', sim_cfg,
            '\nGeneration config:', gen_cfg,
            '\nExtraction config:', ext_cfg,
            '\n-----------------------',
            '\nMODEL INFO',
            '\nModel type:', model_types[method],
            '\nDevice:', device,
            '\n-----------------------',
        )

        # get the pattern radius for later fov restriction
        rp = np.radians(meth_params[method][1])

        # test data
        df = df_dict[method]

        # dataset
        dataset = create_dataset(
            method,
            df,
            meth_params[method],
        )
        loader = DataLoader(dataset, batch_size=20, shuffle=False)

        # best model
        model = create_model(
            method,
            model_types[method],
            meth_params[method],
            len(gcata),
        )
        model_path = os.path.join('model', sim_cfg, method, gen_cfg, model_types[method], 'best_model.pth')
        model.load_state_dict(torch.load(model_path))

        # predict the star catalogue index
        cata_idxs, valid_probs = predict(model, loader, 0, device)
        
        # initilize the predicted ids and probs
        ids, probs = np.full(len(df), -1), np.full(len(df), 0.0)
        
        # some stars may not have valid patterns, because of insufficient stars in their neighboring regions
        mask = df.isna().any(axis=1).to_numpy()
        ids[~mask] = gcata.loc[cata_idxs, 'Star ID'].to_numpy()
        probs[~mask] = valid_probs

        # get all the coordinates
        coords = df[['row', 'col']].to_numpy()

        # get the image id(same as image path)
        img_ids = df['img_id'].to_numpy()

        # add flags
        df['valid'] = ~mask
        df['verified'] = False

        # do verification and postprocess for each image
        for img_id in np.unique(img_ids):
            # get the predications for each image
            esti_ids = ids[img_ids == img_id]
            esti_coords = coords[img_ids == img_id]
            esti_probs = probs[img_ids == img_id]

            # verify the results
            esti_ids, esti_atti = verify(
                gcata,
                coords=esti_coords, 
                ids=esti_ids, 
                probs=esti_probs,
                fov=2*rp, 
                h=simu_params['h'], 
                w=simu_params['w'], 
                f=simu_params['f'],
            )

            if esti_atti is not None:
                df.loc[img_ids==img_id, 'verified'] = esti_ids != -1

            # postprocess
            esti_ids = postprocess(
                gcata,
                esti_coords, 
                esti_ids, 
                esti_atti,
                h=simu_params['h'], 
                w=simu_params['w'], 
                f=simu_params['f'],
                eps1=eps1,
                eps2=eps2
            )

            # store the estimation results
            df.loc[img_ids==img_id, 'star_id'] = esti_ids

            # label the realshot and save it to res dir
            if output_dir is not None:                
                # xxxx.bmp
                img_name = os.path.basename(img_id)

                # read image
                img = cv2.imread(img_id, cv2.IMREAD_GRAYSCALE)

                # label and save the image
                label_star_image(
                    img, 
                    esti_coords, 
                    esti_ids, 
                    circle=True,
                    axis_on=False,
                    show=False,
                    output_path=os.path.join(output_dir, img_name)
                )

        # update dataframe
        df_dict[method] = df[['img_id', 'star_id', 'row', 'col', 'gray', 'valid', 'verified']]

    return df_dict

