# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import argparse
import random
import numpy as np
import scipy.io as sio
from pprint import pprint

def gen_BMs(world_len, mesh_len, n_BM, save_dir, seed=2021):
    """
    Generate a world with building meshes.
    """
    random.seed(seed)
    np.random.seed(seed)

    fname = f'terrain-{n_BM}.mat'
    assert world_len % mesh_len == 0, f'world_len={world_len}, mesh_len={mesh_len}'
    M = world_len//mesh_len
    assert M*M - n_BM >= 0


    raw_grids = np.ones(M*M, dtype=np.float32) * 90
    zero_idcs = sorted(random.sample(range(0, M*M), M*M - n_BM))
    raw_grids[zero_idcs] = 0.
    grids = raw_grids.reshape((M, M)).astype(np.float32)

    mat = {
        'world_len': world_len,
        'mesh_len': mesh_len,
        'N': n_BM,
        'grids': grids,
    }

    save_path = os.path.join(save_dir, fname)
    sio.savemat(save_path, mat)
    return mat, save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_len', type=int, default=1000)
    parser.add_argument('--mesh_len', type=int, default=50)
    parser.add_argument('--n_BM', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    mat, save_path = gen_BMs(args.world_len, args.mesh_len, args.n_BM, args.save_dir, args.seed)
    pprint(mat)
    pprint(save_path)