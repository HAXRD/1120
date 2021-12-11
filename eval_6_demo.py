# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""
Create an eval env to perform a certain method for 2 episodes,
store only the 2 collections of information of entities
    1. before the 2nd episode start, i,e, GUs walked, but the algorithm
    hasn't tracked;
    2. after the algorithm found its best solution and dispatched
    ABSs;
"""

import os
import time
import torch
import numpy as np
from pathlib import Path
import random

from common import run_preparation, make_env, dict2pkl
from eval_shared import demo

if __name__ == "__main__":

    # get specs
    args, run_dir = run_preparation()
    print(f"[demo] running dir is {str(run_dir)}")

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

    if args.scenario == "pattern":
        from runners.pattern import Runner
        eval_runner = Runner(config)

        # load best emulator ckpt
        if args.method in ["mutation-kmeans", "map-elites"]:
            eval_emulator_fpath = args.eval_emulator_fpath
            eval_runner.emulator_load(eval_emulator_fpath)
    elif args.scenario == "precise": # TODO:
        from runners.precise import Runner
        eval_runner = Runner(config)
        # load best policy ckpt
        # TODO:
        pass

    entities_statuses = demo(args, eval_runner)

    # store to pickle
    pkl_fpath = os.path.join(method_dir, f"entities_statuses.pkl")
    dict2pkl(entities_statuses, pkl_fpath)
    print(f"entities statuses saved to '{pkl_fpath}'")
