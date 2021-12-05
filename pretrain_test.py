# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random
import time

from torch.utils.data import DataLoader

from common import get_replay_fpaths, run_preparation, to_csv
from algorithms.emulator import Emulator
from replays.pattern.replay import UniformReplay


def test(args, device):
    """
    Use test set to manual check accuracy.

    :return: (
        mean_abs_elem_error: (float),
        mean_abs_CR_error:   (float)
    )
    """

    """Preparation"""
    # useful params
    run_dir = args.run_dir
    K = args.K

    # dirs
    replay_dir = os.path.join(run_dir, "emulator_replays")
    ckpt_dir = os.path.join(run_dir, "emulator_ckpts")

    # replays
    test_list_of_fpaths = get_replay_fpaths(replay_dir, "test")

    """Load emulator"""
    emulator = Emulator(args, device)
    best_emulator_fpath = os.path.join(ckpt_dir, "best_emulator.pt")
    emulator_state_dict = torch.load(best_emulator_fpath)
    emulator.model.load_state_dict(emulator_state_dict)
    print(f"[pretrain | test] loaded ckpt from '{best_emulator_fpath}'.")

    total_test_size = 0
    total_abs_elem_error = 0
    total_abs_CR_error = 0
    pin_memory = not (device == torch.device("cpu"))
    for _fpaths in zip(*test_list_of_fpaths):

        test_replay = UniformReplay(_fpaths)

        dataloader = DataLoader(test_replay, batch_size=1, pin_memory=pin_memory)
        total_test_size += len(test_replay)

        for P_GUs, P_ABSs, P_CGUs in dataloader:

            P_GUs  = P_GUs.to(device)
            P_ABSs = P_ABSs.to(device)
            P_CGUs = P_CGUs.to(device)

            P_rec_CGUs = emulator.model.predict(P_GUs, P_ABSs)

            pred_error = torch.sum(torch.abs(P_rec_CGUs - P_CGUs)).cpu().numpy()
            total_abs_elem_error += pred_error
            print(f"\Sum|P_rec_CGUs - P_CGUs| = {pred_error}")

            CR = torch.sum(P_CGUs) / torch.sum(P_GUs)
            pCR = torch.sum(P_rec_CGUs) / torch.sum(P_GUs)

            pred_CR_error = torch.sum(torch.abs(pCR - CR)).cpu().numpy()
            total_abs_CR_error += pred_CR_error
            print(f"|{pCR} - {CR}| == {pred_CR_error}")

    mean_abs_elem_error = total_abs_elem_error / total_test_size
    mean_abs_CR_error = total_abs_CR_error / total_test_size

    print(f"mean_abs_elem_error: {mean_abs_elem_error}, mean_abs_CR_error: {mean_abs_CR_error}")
    return (
        mean_abs_elem_error, mean_abs_CR_error
    )

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

    mean_abs_elem_error, mean_abs_CR_error = test(args, device)

    testset_result_fpath = os.path.join(run_dir, f"test_error_{time.strftime('%m%d-%H%M%S')}.csv")

    # write to csv
    header = ["mean_abs_elem_error", "mean_abs_CR_error"]
    data = {
        "mean_abs_elem_error": [mean_abs_elem_error],
        "mean_abs_CR_error": [mean_abs_CR_error]
    }
    to_csv(header, data, testset_result_fpath)