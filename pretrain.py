# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random
import time
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from glob import glob
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from config import get_config
from common import make_env, npz_save
from replays.pattern.emulator import UniformReplay as Replay

def collect(args, ENV_TYPE="base", RENDER="non-display"):
    """
    Collect samples by interacting with site-specific environment,
    then save them as npz files.

    :param args  : (namespace), specs;
    :param RENDER: (str), either 'human' or 'non-display';
    """
    assert RENDER in ["human", "non-display"]

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

    # replay saving dir
    replay_dir = os.path.join(run_dir, "emulator_replays")
    if not os.path.isdir(replay_dir):
        os.makedirs(replay_dir)

    # env
    env = make_env(args, TYPE=ENV_TYPE)

    """Collect data"""
    prefixs = ['train', 'val', 'test']
    for _split, _prefix in zip(splits, prefixs):
        idx = 0
        cur_episodes = _split

        if _prefix in ['val', 'test']:
            n = 2
        else:
            if collect_strategy == "default":
                n = 4
            elif collect_strategy == "subset":
                n = n_ABS * 2

        while cur_episodes > 0:
            episodes = min(cur_episodes, file_episode_limit)
            P_GUs  = np.zeros((episodes * n, K, K), dtype=np.float32)
            P_ABSs = np.zeros((episodes * n, K, K), dtype=np.float32)
            P_CGUs = np.zeros((episodes * n, K, K), dtype=np.float32)

            for _episode in tqdm(range(episodes)):
                
                if _prefix == "train" and collect_strategy == "subset":
                    # totally random
                    env.reset()
                    env.render(RENDER)

                    for _abs_id in range(n_ABS):
                        P_GU_aug, P_ABS_aug, P_CGU_aug = env.get_all_Ps_with_augmentation(_abs_id)
                        P_GUs[n * _episode + _abs_id] = P_GU_aug
                        P_ABSs[n * _episode + _abs_id] = P_ABS_aug
                        P_CGUs[n * _episode + _abs_id] = P_CGU_aug

                    # kmeans
                    kmeans_P_ABS = env.find_KMEANS_P_ABS()
                    env.step(kmeans_P_ABS)
                    env.render(RENDER)

                    for _abs_id in range(n_ABS):
                        P_GU_aug, P_ABS_aug, P_CGU_aug = env.get_all_Ps_with_augmentation(_abs_id)
                        P_GUs[n * _episode + _abs_id + n_ABS] = P_GU_aug
                        P_ABSs[n * _episode + _abs_id + n_ABS] = P_ABS_aug
                        P_CGUs[n * _episode + _abs_id + n_ABS] = P_CGU_aug

                    pass
                elif (_prefix == "train" and collect_strategy == "default") \
                    or _prefix in ["val", "test"]:
                    # totally random
                    env.reset()
                    env.render(RENDER)

                    P_GU, P_ABS, P_CGU = env.get_all_Ps()
                    P_GUs[n * _episode] = P_GU
                    P_ABSs[n * _episode] = P_ABS
                    P_CGUs[n * _episode] = P_CGU

                    if _prefix == "train":
                        abs_id = random.randrange(env.world.n_ABS)
                        P_GU_aug, P_ABS_aug, P_CGU_aug = env.get_all_Ps_with_augmentation(abs_id)
                        P_GUs[n * _episode + 2] = P_GU_aug
                        P_ABSs[n * _episode + 2] = P_ABS_aug
                        P_CGUs[n * _episode + 2] = P_CGU_aug

                    # kmeans
                    kmeans_P_ABS = env.find_KMEANS_P_ABS()
                    env.step(kmeans_P_ABS)
                    env.render(RENDER)

                    P_GU, P_ABS, P_CGU = env.get_all_Ps()
                    P_GUs[n * _episode + 1] = P_GU
                    P_ABSs[n * _episode + 1] = P_ABS
                    P_CGUs[n * _episode + 1] = P_CGU

                    if _prefix == 'train':
                        abs_id = random.randrange(env.world.n_ABS)
                        P_GU_aug, P_ABS_aug, P_CGU_aug = env.get_all_Ps_with_augmentation(abs_id)
                        P_GUs[n * _episode + 3] = P_GU_aug
                        P_ABSs[n * _episode + 3] = P_ABS_aug
                        P_CGUs[n * _episode + 3] = P_CGU_aug

            for pname, p in zip(["GUs", "ABSs", "CGUs"], [P_GUs, P_ABSs, P_CGUs]):
                _save(replay_dir, _prefix, pname, idx, p)
            del P_GUs, P_ABSs, P_CGUs
            # update counters
            cur_episodes -= episodes
            idx += 1

    env.close()

def _get_replay_fpaths(replay_dir, prefix, SHUFFLE_FILE_ORDER=False):

    GUs_fpaths, ABSs_fpaths, CGUs_fpaths  = [
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

def train(args, writer, device):
    """
    Use collected experience to train an emulator model;
    in the meanwhile, to use validation set to compute validation
    loss as the metric to save the best ckpt.
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir

    # dirs
    replay_dir = os.path.join(run_dir, "emulator_replays")
    ckpt_dir = os.path.join(run_dir, "emulator_ckpts")
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)


    """Train emulator"""
    from algorithms.emulator import Emulator
    emulator = Emulator(args, device)
    min_val_loss = float('inf')
    epochs = args.num_emulator_epochs
    best_emulator_epoch_counter = 0

    # get val paths since it does not need to be shuffled every epoch
    val_list_of_fpaths = _get_replay_fpaths(replay_dir, "val")

    for _epoch in range(epochs):

        # get train paths with file order shuffled
        train_list_of_fpaths = _get_replay_fpaths(replay_dir, "train", SHUFFLE_FILE_ORDER=True)

        start = time.time()

        # train
        total_train_loss = 0.
        total_train_size = 0
        for _fpaths in tqdm(zip(*train_list_of_fpaths), desc="train"):
            train_replay = Replay(_fpaths)

            train_loss = emulator.SGD_compute(train_replay, True)

            total_train_loss += train_loss
            total_train_size += len(train_replay)
        mean_train_loss = total_train_loss / total_train_size

        # validate
        total_val_loss = 0.
        total_val_size = 0
        for _fpaths in tqdm(zip(*val_list_of_fpaths), desc="val"):
            val_replay = Replay(_fpaths)

            val_loss = emulator.SGD_compute(val_replay, False)

            total_val_loss += val_loss
            total_val_size += len(val_replay)
        mean_val_loss = total_val_loss / total_val_size

        # log info
        print(f"[pretrain | Epoch {_epoch + 1} | {time.time() - start:.2f}s] \t loss: {mean_train_loss} \t val_loss: {mean_val_loss}, \t previous val loss: {min_val_loss}")
        writer.add_scalar("train_loss", mean_train_loss, _epoch)
        writer.add_scalar("val_loss", mean_val_loss, _epoch)

        # update ckpt
        if mean_val_loss < min_val_loss:
            best_emulator_epoch_counter = 0
            min_val_loss = mean_val_loss
            torch.save(emulator.model.state_dict(), os.path.join(ckpt_dir, "best_emulator.pt"))
            print(f"[pretrain] updated best ckpt file.")
        else:
            best_emulator_epoch_counter += 1
            print(f"[pretrain] not updating, {args.num_emulator_tolerance_epochs - best_emulator_epoch_counter}")
            if best_emulator_epoch_counter >= args.num_emulator_tolerance_epochs:
                break
        torch.save(emulator.model.state_dict(), os.path.join(ckpt_dir, f"emulator-{_epoch}.pt"))

def test(args, device=torch.device("cpu")):
    """
    Use test set to manual check accuracy.
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K

    # dirs
    replay_dir = os.path.join(run_dir, "emulator_replays")
    ckpt_dir = os.path.join(run_dir, "emulator_ckpts")

    # replays
    test_list_of_fpaths = _get_replay_fpaths(replay_dir, "test")


    """Load emulator"""
    from algorithms.emulator import Emulator
    emulator = Emulator(args, device)
    emulator_state_dict = torch.load(os.path.join(ckpt_dir, "best_emulator.pt"))
    emulator.model.load_state_dict(emulator_state_dict)

    bz = 1
    total = 0
    total_test_size = 0
    pin_memory = not (device == torch.device("cpu"))
    for _fpaths in zip(*test_list_of_fpaths):
        test_replay = Replay(_fpaths)

        dataloader = DataLoader(test_replay, batch_size=1, pin_memory=pin_memory)
        total_test_size += len(test_replay)

        for P_GUs, P_ABSs, P_CGUs in dataloader:

            P_GUs  = P_GUs.to(device)
            P_ABSs = P_ABSs.to(device)
            P_CGUs = P_CGUs.to(device)

            P_rec_CGUs = emulator.model.predict(P_GUs, P_ABSs)

            pred_error = torch.sum(torch.abs(P_rec_CGUs - P_CGUs)).cpu().numpy()
            total += pred_error
            pprint(f"{pred_error}")

            CR = torch.sum(P_CGUs) / torch.sum(P_GUs)
            pCR = torch.sum(P_rec_CGUs) / torch.sum(P_GUs)

            print(f"{CR} - {pCR} == {CR - pCR}")

    print(total / total_test_size)


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

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # run dir
    run_dir = args.run_dir
    assert isinstance(run_dir, Path)
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"[pretrain] running dir is '{str(run_dir)}'")

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "pretrain_tb"))

    collect(args, "train", args.render)

    train(args, writer, device)

    test(args, device)