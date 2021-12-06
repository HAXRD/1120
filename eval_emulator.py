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

from common import run_preparation, make_env, to_csv
from eval_shared import test_emulator

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

    # env
    eval_env = make_env(args, "eval")

    mean_abs_elem_error, mean_abs_CR_error = test_emulator(args, device)

    test_result_fpath = os.path.join(run_dir, f"test_emulator_error_{time.strftime('%m%d-%H%M%S')}.csv")

    # write to csv
    header = ["mean_abs_elem_error", "mean_abs_CR_error"]
    data = {
        "mean_abs_elem_error": [mean_abs_elem_error],
        "mean_abs_CR_error": [mean_abs_CR_error]
    }
    to_csv(header, data, test_result_fpath)