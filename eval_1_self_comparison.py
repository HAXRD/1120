# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random
import time

from pathlib import Path

from eval_shared import eval_procedure
from common import run_preparation, make_env, dict2csv, to_csv
from runners.pattern import Runner

if __name__ == "__main__":
    """
    Load the best emulator ckpt to evaluate with different methods.
    """

    # get specs
    args, run_dir = run_preparation()

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

    # method dir
    method_dir = Path(os.path.join(run_dir, args.method))
    if not method_dir.exists():
        os.makedirs(str(method_dir))
    print(f"[eval | {args.method}] running dir is {str(run_dir)}")
    print(f"[eval | {args.method}] method dir is {str(method_dir)}")

    eval_env = make_env(args, "eval")

    config = {
        "args": args,
        "run_dir": run_dir,
        "method_dir": method_dir,
        "env": eval_env,
        "device": device
    }

    eval_runner = Runner(config)

    # load best emulator ckpt
    if args.method in ["mutation-kmeans", "map-elites"]:
        eval_runner.emulator_load()

    # start eval
    episodes_CRs, episodes_mean_CRs, overall_mean_CR = eval_procedure(args, eval_runner, args.render)

    timestamp = time.strftime('%m%d-%H%M%S')

    # store overall_mean_CR
    raw_dict = {
        "overall_mean_CR": [overall_mean_CR]
    }
    raw_fpath = os.path.join(method_dir, f"overall_mean_CR_{timestamp}.csv")
    dict2csv(raw_dict, raw_fpath)

    # store episodes_CRs
    raw_dict = {
        "episodes_CRs": episodes_CRs
    }
    raw_fpath = os.path.join(method_dir, f"episodes_CRs_{timestamp}.csv")
    dict2csv(raw_dict, raw_fpath)

    # store expisodes_means_CRs
    header = ["episode", "mean_CR"]
    data = {
        "episode": range(len(episodes_mean_CRs)),
        "mean_CR": episodes_mean_CRs
    }
    episodes_mean_CRs_fpath = os.path.join(method_dir, f"episodes_mean_CRs_{timestamp}.csv")
    to_csv(header, data, episodes_mean_CRs_fpath)
