# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random
import time

from pathlib import Path

from eval_shared import eval_procedure
from common import run_preparation, make_env
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
    df, mean_df = eval_procedure(args, eval_runner, args.render)

    timestamp = time.strftime('%m%d-%H%M%S')

    # store dataframe
    raw_fpath = os.path.join(method_dir, f"raw_CRs_{timestamp}.csv")
    df.to_csv(raw_fpath, index=False)
    print(f"dataframe saved to '{raw_fpath}'")

    # store mean dataframe
    mean_fpath = os.path.join(method_dir, f"mean_CR_{timestamp}.csv")
    mean_df.to_csv(mean_fpath, index=False)
    print(f"dataframe saved to '{mean_fpath}'")
