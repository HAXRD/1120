# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import make_env, npz_save, run_preparation

def collect(args, env_type, RENDER="non-display"):
    """
    Collect samples by interacting with site-specific environment,
    then save them as npz files.

    :param args  : (namespace), specs;
    :param RENDER: (str), either 'human' or 'non-display';
    """

    def _save(dir, prefix, pname, idx, arr):
        fpath = os.path.join(dir, f"{prefix}_{pname}_s{args.seed}_{idx}.npz")
        npz_save(arr, fpath)
        print(f"[pretrain | collect] saved to '{fpath}' (size: {len(arr)})")

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K
    n_ABS = args.n_ABS
    splits = args.splits
    file_episode_limit = args.file_episode_limit
    collect_strategy = args.collect_strategy
    num_episodes_per_trial = args.num_episodes_per_trial

    # replay saving dir
    replay_dir = os.path.join(run_dir, "emulator_replays")
    if not os.path.isdir(replay_dir):
        os.makedirs(replay_dir)

    # env
    env = make_env(args, TYPE=env_type)

    """Collect data"""
    prefixs = ["train", "val", "test"]
    for _split, _prefix in zip(splits, prefixs):
        idx = 0
        cur_episodes = _split

        if _prefix in prefixs[-2:]:
            n = 2           # (1 random + 1 kmeans)
        else:
            if collect_strategy == "default":
                n = 2       # (1 random + 1 kmeans)
            elif collect_strategy == "half":
                n = 4       # (1 random + 1 random subset + 1 kmeans + 1 kmeans subset)
                m = 1       # m random subsets
            elif collect_strategy == "third":
                n = 6       # (1 random + 2 random subset + 1 kmeans + 2 kmeans subset)
                m = 2

        while cur_episodes > 0:
            episodes = min(cur_episodes, file_episode_limit)
            P_GUs  = np.zeros((episodes * n, K, K), dtype=np.float32)
            P_ABSs = np.zeros((episodes * n, K, K), dtype=np.float32)
            P_CGUs = np.zeros((episodes * n, K, K), dtype=np.float32)

            for _episode in tqdm(range(episodes)):

                if _prefix in prefixs[-2:] or \
                    _prefix == prefixs[0] and collect_strategy == "default":

                    # totally random
                    if _episode % num_episodes_per_trial == 0:
                        env.reset()
                    else:
                        env.walk()
                    env.render(RENDER)

                    P_GU, P_ABS, P_CGU = env.get_all_Ps()
                    P_GUs[n * _episode] = P_GU
                    P_ABSs[n * _episode] = P_ABS
                    P_CGUs[n * _episode] = P_CGU

                    # kmeans
                    kmeans_P_ABS = env.find_KMEANS_P_ABS()
                    env.step(kmeans_P_ABS)
                    env.render(RENDER)

                    P_GU, P_ABS, P_CGU = env.get_all_Ps()
                    P_GUs[n * _episode + 1] = P_GU
                    P_ABSs[n * _episode + 1] = P_ABS
                    P_CGUs[n * _episode + 1] = P_CGU

                elif _prefix == prefixs[0] and collect_strategy in ["half", "third"]:

                    # totally random
                    if _episode % num_episodes_per_trial == 0:
                        env.reset()
                    else:
                        env.walk()
                    env.render(RENDER)

                    P_GU, P_ABS, P_CGU = env.get_all_Ps()
                    P_GUs[n * _episode] = P_GU
                    P_ABSs[n * _episode] = P_ABS
                    P_CGUs[n * _episode] = P_CGU

                    # sample unique abs ids
                    sampled = random.sample(range(n_ABS), m)
                    for j, _abs_id in enumerate(sampled):
                        abs_ids = [_abs_id]
                        P_GU_aug, P_ABS_aug, P_CGU_aug = env.get_all_Ps_with_augmentation(abs_ids)
                        P_GUs[n * _episode + j + 1] = P_GU_aug
                        P_ABSs[n * _episode + j + 1] = P_ABS_aug
                        P_CGUs[n * _episode + j + 1] = P_CGU_aug

                    # kmeans
                    kmeans_P_ABS = env.find_KMEANS_P_ABS()
                    env.step(kmeans_P_ABS)
                    env.render(RENDER)

                    P_GU, P_ABS, P_CGU = env.get_all_Ps()
                    P_GUs[n * _episode + m + 1] = P_GU
                    P_ABSs[n * _episode + m + 1] = P_ABS
                    P_CGUs[n * _episode + m + 1] = P_CGU

                    # sample unique abs ids
                    sampled = random.sample(range(n_ABS), m)
                    for j, _abs_id in enumerate(sampled):
                        abs_ids = [_abs_id]
                        P_GU_aug, P_ABS_aug, P_CGU_aug = env.get_all_Ps_with_augmentation(abs_ids)
                        P_GUs[n * _episode + m + 1 + j + 1] = P_GU_aug
                        P_ABSs[n * _episode + m + 1 + j + 1] = P_ABS_aug
                        P_CGUs[n * _episode + m + 1 + j + 1] = P_CGU_aug

            for pname, p in zip(["GUs", "ABSs", "CGUs"], [P_GUs, P_ABSs, P_CGUs]):
                _save(replay_dir, _prefix, pname, idx, p)
            del P_GUs, P_ABSs, P_CGUs

            # update counters
            cur_episodes -= episodes
            idx += 1

    env.close()

if __name__ == "__main__":

    # get specs
    args, run_dir = run_preparation()
    print(f"[pretrain | collect] running dir is {str(run_dir)}")

    # cuda
    torch.set_num_threads(1)
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("chosse to use cpu...")
        device = torch.device("cpu")

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "pretrain_tb"))

    collect(args, "train", args.render)

    print(f"[pretrain | collect] seed={args.seed} done!")