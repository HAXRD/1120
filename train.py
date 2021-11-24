# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import torch
import numpy as np
import random
from pathlib import Path
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value

from config import get_config
from common import make_env
from eval import evaluation

if __name__ == "__main__":

    # get specs
    parser = get_config()
    args = parser.parse_args()
    pprint(args)

    # cuda
    torch.set_num_threads(1)
    if args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        if torch.cuda.device_count() == 1:
            device1 = torch.device("cuda:0")
            device2 = torch.device("cpu")
        elif torch.cuda.device_count() == 2:
            device1 = torch.device("cuda:0")
            device2 = torch.device("cuda:1")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("chosse to use cpu...")
        device1 = torch.device("cpu")
        device2 = torch.device("cpu")

    # dirs
    run_dir = args.run_dir
    assert isinstance(run_dir, Path)
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    print(f"[train] run_dir is '{str(run_dir)}'.")

    method_dir = Path(os.path.join(run_dir, args.method))
    assert isinstance(method_dir, Path)
    if not method_dir.exists():
        os.makedirs(str(method_dir))
    print(f"[train] method_dir is '{str(method_dir)}'.")

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(method_dir, "train_tb"))

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # env
    env = make_env(args, "train")
    eval_env = make_env(args, "eval") if args.use_eval else None

    config = {
        "args": args,
        "run_dir": run_dir,
        "method_dir": method_dir,
        "env": env,
        "device": device1,
        "writer": writer,
    }

    eval_config = {
        "args": args,
        "run_dir": run_dir,
        "method_dir": method_dir,
        "env": eval_env,
        "device": device2,
        "writer": writer
    }

    if args.scenario == "pattern":
        from runners.pattern import Runner
    elif args.scenario == "precise":
        from runners.precise import Runner
    runner = Runner("TrainRunner", config)
    eval_runner = Runner("EvalRunner", eval_config)

    # eval subprocess
    test_q = Queue()
    done_training = Value('i', False)
    p = mp.Process(target=evaluation, args=(args, eval_runner, test_q, done_training))
    p.start()

    runner.run(test_q)

    # close envs
    env.close()
    if args.use_eval and eval_env is not env:
        eval_env.close()