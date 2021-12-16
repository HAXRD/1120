# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import make_env, npz_save, run_preparation

def _save(dir, prefix, pname, seed, idx, arr):
    fpath = os.path.join(dir, f"{prefix}_{pname}_s{seed}_{idx}.npz")
    npz_save(arr, fpath)
    print(f"[pretrain | collect] saved to '{fpath}' (size: {len(arr)})")


def collect(args, env_type, RENDER="non-display"):
    """
    Collect samples by interacting with site-specific environment,
    then save them as npz files.

    :param args  : (namespace), specs;
    :param RENDER: (str), either 'human' or 'non-display';
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K
    n_ABS = args.n_ABS
    splits = args.splits
    file_episode_limit = args.file_episode_limit
    collect_strategy = args.collect_strategy
    num_episodes_per_trial = args.num_episodes_per_trial
    seed = args.seed

    # replay saving dir
    replay_dir = os.path.join(run_dir, "emulator_replays")
    if not os.path.isdir(replay_dir):
        os.makedirs(replay_dir)

    # env
    env = make_env(args, TYPE=env_type)

    """Collect data"""
    prefixs = ["test", "val", "train"]
    for _split, _prefix in zip(splits, prefixs):
        idx = 0
        cur_episodes = _split

        if _prefix in prefixs[:2]:
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

                if _prefix in prefixs[:2] or \
                    _prefix == prefixs[-1] and collect_strategy == "default":

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

                elif _prefix == prefixs[-1] and collect_strategy in ["half", "third"]:

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
                _save(replay_dir, _prefix, pname, seed, idx, p)
            del P_GUs, P_ABSs, P_CGUs

            # update counters
            cur_episodes -= episodes
            idx += 1

    env.close()

def collect_3_adaptive_to_variable_entities(args, env_type, RENDER="non-display"):
    """
    Collect samples with a range of config (n_ABS, n_GU) with site-specific
    environment,
    then save them as npz files.

    :param args  : (namespace), specs;
    :param RENDER: (str), either 'human' or 'non-display';
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K
    n_ABS = args.n_ABS
    splits = args.splits
    file_episode_limit = args.file_episode_limit
    num_episodes_per_trial = args.num_episodes_per_trial
    seed = args.seed
    variable_n_ABS = args.variable_n_ABS
    variable_n_GU = args.variable_n_GU

    # replay saving dir
    replay_dir = os.path.join(run_dir, "emulator_replays")
    if not os.path.isdir(replay_dir):
        os.makedirs(replay_dir)

    n = 6

    # envs
    envs = []
    for i in range(n):
        copyed_args = deepcopy(args)
        if variable_n_ABS:
            copyed_args.n_ABS = args.n_ABS - i
        if variable_n_GU:
            copyed_args.n_GU = args.n_GU - i * 25
        envs.append(make_env(copyed_args, TYPE=env_type))

    """Collect data"""
    prefixs = ["test", "val", "train"]
    for _split, _prefix in zip(splits, prefixs):
        idx = 0
        cur_episodes = _split

        while cur_episodes > 0:
            episodes = min(cur_episodes, file_episode_limit)
            P_GUs  = np.zeros((episodes * n * 2, K, K), dtype=np.float32)
            P_ABSs = np.zeros((episodes * n * 2, K, K), dtype=np.float32)
            P_CGUs = np.zeros((episodes * n * 2, K, K), dtype=np.float32)

            for _episode in tqdm(range(episodes)):

                # totally random
                for i in range(n):
                    if _episode % num_episodes_per_trial == 0:
                        envs[i].reset()
                    else:
                        envs[i].walk()
                    envs[i].render(RENDER)

                    P_GU, P_ABS, P_CGU = envs[i].get_all_Ps()
                    j = 2* n * _episode + i
                    P_GUs[j] = P_GU
                    P_ABSs[j] = P_ABS
                    P_CGUs[j] = P_CGU

                    kmeans_P_ABS = envs[i].find_KMEANS_P_ABS()
                    envs[i].step(kmeans_P_ABS)
                    envs[i].render(RENDER)

                    P_GU, P_ABS, P_CGU = envs[i].get_all_Ps()
                    j = 2 * n * _episode + i + n
                    P_GUs[j] = P_GU
                    P_ABSs[j] = P_ABS
                    P_CGUs[j] = P_CGU

            for pname, p in zip(["GUs", "ABSs", "CGUs"], [P_GUs, P_ABSs, P_CGUs]):
                _save(replay_dir, _prefix, pname, seed, idx, p)
            del P_GUs, P_ABSs, P_CGUs

            # update counters
            cur_episodes -= episodes
            idx += 1

    for i in range(n):
        envs[i].close()

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

    if args.collect_strategy == "variable":
        assert not (args.variable_n_GU & args.variable_n_ABS)
        assert args.variable_n_GU | args.variable_n_ABS
        collect_3_adaptive_to_variable_entities(args, "train", args.render)
    else:
        collect(args, "train", args.render)

    print(f"[pretrain | collect] seed={args.seed} done!")