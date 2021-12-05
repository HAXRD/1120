# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random
import csv

from pprint import pprint
from pathlib import Path
from glob import glob
from tqdm import tqdm

from envs.sse.SSE_env import SSEEnv
from config import get_config



def make_env(args, TYPE):
    assert TYPE in ["base", "train", "eval"]
    if TYPE == "base":
        a, b, is_base = 1, 0, True
    elif TYPE == "train":
        a, b, is_base = 10, 1, False
    elif TYPE == "eval":
        a, b, is_base = 1000, 13, False

    seed = a * args.seed + b
    env = SSEEnv(args, is_base, seed)
    env.seed(seed)
    print(f"[env | seed] processed seed for env is {seed}")

    return env

def sync(fh):
    """This make sure data is writter to disk."""
    fh.flush()
    os.fsync(fh.fileno())

def npz_save(data, fpath):
    with open(fpath, 'wb+') as fh:
        np.savez_compressed(fh, data=data)
        sync(fh)

def npz_load(fpath):
    data = np.load(fpath)
    return data['data']

def load_n_copy(replay, replay_dir, prefix):
    from replays.pattern.replay import UniformReplay
    assert isinstance(replay, UniformReplay)

    P_GUs_fnames, P_ABSs_fnames, P_CGUs_fnames = [
        sorted(glob(os.path.join(replay_dir, f"{prefix}_{p}_*.npz")))
        for p in ["GUs", "ABSs", "CGUs"]
    ]

    for _GU_fname, _ABS_fname, _CGU_fname in tqdm(zip(P_GUs_fnames, P_ABSs_fnames, P_CGUs_fnames)):
        P_GUs, P_ABSs, P_CGUs = [npz_load(fpath) for fpath in [_GU_fname, _ABS_fname, _CGU_fname]]

        P_GUs = np.expand_dims(P_GUs, axis=1)
        P_ABSs = np.expand_dims(P_ABSs, axis=1)
        P_CGUs = np.expand_dims(P_CGUs, axis=1)

        data = P_GUs, P_ABSs, P_CGUs
        replay.paste(data)
        del data

def run_preparation():
    """
    Shared code snippet across all simulation related code.
    Call this function before do further ops.

    :return: (
        args,
        run_dir
    )
    """

    # get specs
    parser = get_config()
    args = parser.parse_args()
    pprint(vars(args))

    # run dir
    run_dir = args.run_dir
    assert isinstance(run_dir, Path)
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    return (
        args,
        run_dir
    )

def get_replay_fpaths(replay_dir, prefix, SHUFFLE_FILE_ORDER=False):
    """
    Get all pattern files in given directory.
    """

    GUs_fpaths, ABSs_fpaths, CGUs_fpaths = [
        sorted(glob(os.path.join(replay_dir, f"{prefix}_{pname}_*.npz")))
        for pname in ["GUs", "ABSs", "CGUs"]
    ]

    if SHUFFLE_FILE_ORDER:
        n_files = len(GUs_fpaths)
        perm = np.arange(n_files)
        np.random.shuffle(perm)

        GUs_fpaths  = [GUs_fpaths[i]  for i in perm]
        ABSs_fpaths = [ABSs_fpaths[i] for i in perm]
        CGUs_fpaths = [CGUs_fpaths[i] for i in perm]
        print(f"[pretrain | {prefix}] replay fpaths shuffled!")
    print(f"[pretrain | {prefix}] get all {prefix} file paths.")
    return (
        GUs_fpaths, ABSs_fpaths, CGUs_fpaths
    )

def to_csv(header, data, fpath, mode="w"):
    """
    Write data to csv.
        e.g. header = ["id", "name", "gender"]
             data = {
                 "id":   [0, 1, 2],
                 "name": ["Jack", "Peter", "May"],
                 "gener": ["male", "male", "female"]
             }
    :param header: (list), a list of header names
    :param data  : (dict), a dictionary whose keys are header names, values
    are a list of values.
    """

    with open(fpath, mode) as f:
        w = csv.writer(f)
        w.writerow(header)
        content = [data[hn] for hn in header]
        for _row in zip(*content):
            w.writerow(_row)

    print(f"data is writen to '{fpath}'")

def dict2csv(output_dict, fpath):
    with open(fpath, "w") as f:
        writer = csv.writer(f, delimiter=',')
        for k, v in output_dict.items():
            v = [k] + v
            writer.writerow(v)

    print(f"data is written to '{fpath}'")