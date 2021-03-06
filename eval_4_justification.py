# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""
Load the best emulator ckpt with given emulator ckpt fpath to
evaluate the emulator's planning method and get all these planning
`P_ABS` patterns to actually interact with environment.
Then compute how much the percentage that planning method yielded
patterns actually contains the patterns that have the top ?% (e.g. 5%)
coverage.
"""

import os
import time
import torch
import numpy as np
import random

from pathlib import Path

from common import run_preparation, make_env
from runners.pattern import Runner
from eval_shared import justification

if __name__ == "__main__":

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

    # env
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
        eval_runner.emulator_load(args.eval_emulator_fpath)

    # start eval
    df, df_processed, mean_df_processed = justification(args, eval_runner)

    timestamp = time.strftime('%m%d-%H%M%S')

    # store raw dataframe
    raw_fpath = os.path.join(method_dir, f"raw_top_x_percentages_{timestamp}.csv")
    df.to_csv(raw_fpath, index=False)
    print(f"dataframe saved to '{raw_fpath}'")

    # store processed dataframe
    processed_fpath = os.path.join(method_dir, f"processed_top_x_percentages_{timestamp}.csv")
    df_processed.to_csv(processed_fpath, index=False)
    print(f"dataframe saved to '{processed_fpath}'")

    # store mean dataframe
    mean_processed_fpath = os.path.join(method_dir, f"mean_processed_top_x_percentages_{timestamp}.csv")
    mean_df_processed.to_csv(mean_processed_fpath, index=False)
    print(f"dataframe saved to '{mean_processed_fpath}'")
