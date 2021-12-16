# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import time
import torch
import numpy as np
import random

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from common import get_replay_fpaths, run_preparation
from algorithms.emulator import Emulator
from replays.pattern.replay import UniformReplay

def train(args, writer, device):
    """
    Use collected experience to train an emulator model;
    in the meanwhile, to use validation set to compute validation
    loss as the metric to save the best ckpt.
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    epochs = args.num_emulator_epochs

    # dirs
    replay_dir = os.path.join(run_dir, "emulator_replays")
    ckpt_dir = os.path.join(run_dir, "emulator_ckpts")
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    """Train emulator"""
    emulator = Emulator(args, device)
    best_emulator_epoch_counter = 0
    min_val_loss = float("inf")

    # get val paths since it does not need to be shuffled every epoch
    val_list_of_fpaths = get_replay_fpaths(replay_dir, "val")

    for _epoch in range(epochs):

        # get train paths with file order shuffled
        train_list_of_fpaths = get_replay_fpaths(replay_dir, "train", True)

        start = time.time()

        # train
        total_train_loss = 0.
        total_train_size = 0
        for _fpaths in tqdm(zip(*train_list_of_fpaths), desc="train"):

            train_replay = UniformReplay(_fpaths)

            train_loss = emulator.SGD_compute(train_replay, True)

            total_train_loss += train_loss
            total_train_size += len(train_replay)
        mean_train_loss = total_train_loss / total_train_size

        # validate
        total_val_loss = 0.
        total_val_size = 0
        for _fpaths in tqdm(zip(*val_list_of_fpaths), desc="val"):

            val_replay = UniformReplay(_fpaths)

            val_loss = emulator.SGD_compute(val_replay, False)

            total_val_loss += val_loss
            total_val_size += len(val_replay)
        mean_val_loss = total_val_loss / total_val_size

        # log info
        end = time.time()
        print(f"[pretrain | Epoch {_epoch + 1} | {end - start:.2f}s] loss: {mean_train_loss}, val_loss: {mean_val_loss}, previous val_loss: {min_val_loss}")
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
            print(f"[pretrain] not updating, counter {args.num_emulator_tolerance_epochs - best_emulator_epoch_counter} left.")
            if best_emulator_epoch_counter >= args.num_emulator_tolerance_epochs:
                break
        torch.save(emulator.model.state_dict(), os.path.join(ckpt_dir, f"emulator-{_epoch}.pt"))

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

    train(args, writer, device)
