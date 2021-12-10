# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""
Create eval env to evaluate emulator accuracy by
comparing the ground truth `P_CGU` with emulator
predicted `P_rec_CGU`.
"""

import os
import time
import torch
import numpy as np
import random

from common import run_preparation, make_env
from eval_shared import test_emulator

if __name__ == "__main__":

    # get specs
    args, run_dir = run_preparation()
    print(f"[eval emulator] running dir is {str(run_dir)}")

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

    # env
    eval_env = make_env(args, "eval")

    df, mean_df = test_emulator(args, device)

    timestamp = time.strftime('%m%d-%H%M%S')

    # store dataframe
    raw_fpath = os.path.join(run_dir, f"raw_emulator_metrics_{timestamp}.csv")
    df.to_csv(raw_fpath, index=False)
    print(f"dataframe saved to '{raw_fpath}'")

    # store mean dataframe
    mean_fpath = os.path.join(run_dir, f"mean_emulator_metrics_{timestamp}.csv")
    mean_df.to_csv(mean_fpath, index=False)
    print(f"dataframe saved to '{mean_fpath}'")
