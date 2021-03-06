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
    assert int(world_len % mesh_len) == 0, f'world_len={world_len}, mesh_len={mesh_len}'
    M = int(world_len / mesh_len)
    assert M*M - n_BM >= 0

    if args.random_h:
        raw_grids = np.random.randint(args.h_min, args.h_max, size=M*M).astype(np.float32)
    else:
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

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, fname)
    sio.savemat(save_path, mat)
    return mat, save_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_len', type=float, default=1000)
    parser.add_argument('--mesh_len', type=float, default=31.25)
    parser.add_argument('--n_BM', type=int, required=True)
    parser.add_argument('--save_dir', type=str, default='./terrains')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--random_h', action='store_false', default=True,
                        help="by default true, using random height.")
    parser.add_argument('--h_min', type=int, default=30)
    parser.add_argument('--h_max', type=int, default=90)
    args = parser.parse_args()

    assert args.h_min <= args.h_max

    mat, save_path = gen_BMs(args.world_len, args.mesh_len, args.n_BM, args.save_dir, args.seed)
    pprint(mat)
    pprint(save_path)