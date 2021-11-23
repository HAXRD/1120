# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import time
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
from config import get_config

from common import make_env, npz_save, load_n_copy

def collect(args, RENDER="non-display"):
    """
    Use computer simulation to quickly train a base emulator without
    site-specific information.

    :param args  : (namespace), specs;
    :param RENDER: (str), whether to render;
    """
    assert RENDER in ["human", "non-display"]

    def _save(dir, prefix, pname, idx, arr):
        fpath = os.path.join(dir, f"{prefix}_{pname}_{idx}.npz")
        npz_save(arr, fpath)
        print(f"[pretrain | collect] saved to '{fpath}' (size: {len(arr)})")

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K
    splits = args.splits
    file_episode_limit = args.file_episode_limit

    # replay saving dir
    replay_dir = os.path.join(run_dir, "base_emulator_replays")
    if not os.path.isdir(replay_dir):
        os.makedirs(replay_dir)

    # env
    env = make_env(args, TYPE="base")

    """Collect data"""
    prefixs = ['train', 'val', 'test']
    for _split, _prefix in zip(splits, prefixs):
        idx = 0
        cur_episodes = _split
        while cur_episodes > 0:
            episodes = min(cur_episodes, file_episode_limit)
            P_GUs  = np.zeros((episodes * 2, K, K), dtype=np.float32)
            P_ABSs = np.zeros((episodes * 2, K, K), dtype=np.float32)
            P_CGUs = np.zeros((episodes * 2, K, K), dtype=np.float32)

            for _episode in tqdm(range(episodes)):
                # totally random
                env.reset()
                env.render(RENDER)
                P_GU, P_ABS, P_CGU = env.get_all_Ps()
                P_GUs[2 * _episode] = P_GU
                P_ABSs[2 * _episode] = P_ABS
                P_CGUs[2 * _episode] = P_CGU

                # kmeans
                kmeans_P_ABS = env.find_KMEANS_P_ABS()
                env.step(kmeans_P_ABS)
                env.render(RENDER)
                P_GU, P_ABS, P_CGU = env.get_all_Ps()
                P_GUs[2 * _episode + 1] = P_GU
                P_ABSs[2 * _episode + 1] = P_ABS
                P_CGUs[2 * _episode + 1] = P_CGU

            for pname, p in zip(["GUs", "ABSs", "CGUs"], [P_GUs, P_ABSs, P_CGUs]):
                _save(replay_dir, _prefix, pname, idx, p)
            del P_GUs, P_ABSs, P_CGUs
            # update counters
            cur_episodes -= episodes
            idx += 1

def train_base(args, writer, device):
    """
    Use collected experience to train a base emulator model;
    in the meanwhile, to use validation set to compute validation
    loss as the metric to save the best ckpt.
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K
    splits = args.splits

    # dirs
    replay_dir = os.path.join(run_dir, "base_emulator_replays")
    ckpt_dir = os.path.join(run_dir, "base_emulator_ckpts")
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # replays
    from replays.pattern.emulator import UniformReplay as Replay
    train_replay = Replay(K, 2 * splits[0])
    val_replay = Replay(K, 2 * splits[1])

    load_n_copy(train_replay, replay_dir, 'train')
    load_n_copy(val_replay, replay_dir, 'val')

    """Train base emulator"""
    from algorithms.emulator import Emulator
    base_emulator = Emulator(args, device)
    min_val_loss = float('inf')
    epochs = args.num_base_emulator_epochs
    best_base_emulator_epoch_counter = 0

    for _epoch in range(epochs):
        start = time.time()
        # train
        train_replay.shuffle()
        train_loss = base_emulator.SGD_compute(train_replay, True)

        # validate
        val_loss = base_emulator.SGD_compute(val_replay)

        # log info
        print(f"[pretrain | Epoch {_epoch + 1} | {time.time() - start:.2f}s] \t loss: {train_loss} \t val_loss: {val_loss}, \t previous val loss: {min_val_loss}")
        writer.add_scalar("train_loss", train_loss, _epoch)
        writer.add_scalar("val_loss", val_loss, _epoch)

        # update ckpt
        if val_loss < min_val_loss:
            best_base_emulator_epoch_counter = 0
            min_val_loss = val_loss
            torch.save(base_emulator.model.state_dict(), os.path.join(ckpt_dir, f"best_base_emulator.pt"))
            print(f"[pretrain] updated best ckpt file.")
        else:
            best_base_emulator_epoch_counter += 1
            print(f"[pretrain] not updating.")
            if best_base_emulator_epoch_counter >= args.num_base_emulator_tolerance_epochs:
                break
        torch.save(base_emulator.model.state_dict(), os.path.join(ckpt_dir, f"base_emulator.pt"))


def test(args, device=torch.device("cpu")):
    """
    Use test set to manual check accuracy
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K
    splits = args.splits
    n_GU = args.n_GU

    # dirs
    replay_dir = os.path.join(run_dir, "base_emulator_replays")
    ckpt_dir = os.path.join(run_dir, "base_emulator_ckpts")

    # replays
    from replays.pattern.emulator import UniformReplay as Replay
    test_replay = Replay(K, 2 * splits[2])

    load_n_copy(test_replay, replay_dir, 'test')

    """Load emulator"""
    from algorithms.emulator import Emulator
    base_emulator = Emulator(args, device)
    base_emulator_state_dict = torch.load(os.path.join(ckpt_dir, f"best_base_emulator.pt"))
    base_emulator.model.load_state_dict(base_emulator_state_dict)

    bz = 1
    cnter = 0
    with torch.no_grad():
        for sample in test_replay.data_loader(bz):
            if cnter > 10:
                break
            cnter += 1
            P_GUs = torch.FloatTensor(sample["P_GUs"]).to(device)
            P_ABSs = torch.FloatTensor(sample["P_ABSs"]).to(device)
            P_CGUs = torch.FloatTensor(sample["P_CGUs"]).to(device)

            P_rec_CGUs = base_emulator.model.predict(P_GUs, P_ABSs)

            # print grid-wise prediction difference
            pprint(f"{torch.sum(torch.abs(P_rec_CGUs - P_CGUs))}")
            # overall CR difference
            CR = torch.sum(P_CGUs) / n_GU
            pCR = torch.sum(P_rec_CGUs) / n_GU

            pprint('[CGU] - [recons] == [diff]')
            pprint(f"{CR} - {pCR} == {CR - pCR}")

            print("")

if __name__ == "__main__":

    # get specs
    parser = get_config()
    args = parser.parse_args()
    pprint(args)

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

    # run dir
    run_dir = args.run_dir
    assert isinstance(run_dir, Path)
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"[pretrain] running dir is '{str(run_dir)}'")

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "pretrain_tb"))

    collect(args, 'human')

    train_base(args, writer, device)

    test(args, device)