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

from common import run_preparation, make_env, to_csv
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
    raw_percentage_dict, percentage_dict = justification(args, eval_runner)

    timestamp = time.strftime('%m%d-%H%M%S')

    # store raw to csv
    header = [k for k, _ in raw_percentage_dict.items()]
    data = raw_percentage_dict
    top_x_raw_percentages_fpath = os.path.join(method_dir, f"top_x_raw_percentages_{timestamp}.csv")
    to_csv(header, data, top_x_raw_percentages_fpath)

    # store processed to csv
    header = [k for k, _ in percentage_dict.items()]
    data = percentage_dict
    top_x_percentages_fpath = os.path.join(method_dir, f"top_x_percentages_{timestamp}.csv")
    to_csv(header, data, top_x_percentages_fpath)