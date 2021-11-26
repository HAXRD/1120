# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
import time
import numpy as np
import torch
import random
import contextlib
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

from config import get_config
from common import make_env, dict2csv

@contextlib.contextmanager
def temp_seed(seed):
    npstate = np.random.get_state()
    ranstate = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(npstate)
        random.setstate(ranstate)


def pattern_procedure(args, runner, RENDER):

    # specs
    episodes = args.num_eval_episodes
    num_episodes_per_trial = args.num_episodes_per_trial
    num_mutation_seeds = args.num_mutation_seeds
    num_mutations_per_seed = args.num_mutations_per_seed
    iterations = args.iterations
    n_sample_individuals = args.num_sample_individuals
    K = args.K
    top_k = runner.top_k

    best_CRs = []

    # start eval for pattern-style
    for _episode in tqdm(range(episodes)):

        start = time.time()

        # reset or walk
        if _episode % num_episodes_per_trial == 0:
            runner.env.reset()
        else:
            runner.env.walk()
        runner.env.render(RENDER)
        P_GU = runner.env.get_P_GU()

        # plan with different methods
        top_CR_info = ""
        if runner.method == "naive-kmeans":
            batch_size, top_k_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, top_k, 0)
        elif runner.method == "mutation-kmeans":
            batch_size, top_k_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, num_mutation_seeds, num_mutations_per_seed)
        elif runner.method == "map-elites":
            batch_size, top_k_P_ABSs = runner.map_elites(top_k, P_GU, iterations, n_sample_individuals)

        # interact with env
        top_k_P_CGUs = np.zeros((batch_size, K, K), dtype=np.float32)
        for _idx, _P_ABS in enumerate(top_k_P_ABSs):
            runner.env.step(_P_ABS)
            runner.env.render(RENDER)
            top_k_P_CGUs[_idx] = runner.env.get_P_CGU()

        top_k_CRs = np.sum(top_k_P_CGUs.reshape(batch_size, -1), axis=1) / runner.env.world.n_ON_GU
        top_CR = np.max(top_k_CRs)

        best_CRs.append(top_CR)
        top_CR_info += f" {top_CR}"

        end = time.time()
        print(f"[eval | top {top_CR_info} | {end - start}s] finished!")

    return best_CRs

def eval_procedure(args, runner, RENDER="humnan"):
    """
    Evaluation procedure.
    """

    print(f"[Eval] start.")

    with temp_seed(args.seed + 20212021):

        if args.scenario == "pattern":
            best_CRs = pattern_procedure(args, runner, RENDER)
        elif args.scenario == "precise":
            pass

        mean_CR = np.mean(best_CRs)
        print(f"-------------")
        print(f"[eval | mean_CR] {mean_CR}.")
        print(f"-------------")

    print(f"[eval] end.")
    return best_CRs


if __name__ == "__main__":
    """
    Load the best emualtor ckpt to evaluate.
    """

    # get specs
    parser = get_config()
    args = parser.parse_args()
    pprint(args)

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

    # dirs
    run_dir = args.run_dir
    assert isinstance(run_dir, Path)
    assert run_dir.exists()
    print(f"[train] run_dir is '{str(run_dir)}'.")

    method_dir = Path(os.path.join(run_dir, args.method))
    assert isinstance(method_dir, Path)
    if not method_dir.exists():
        os.makedirs(str(method_dir))
    print(f"[train] method_dir is '{str(method_dir)}'.")

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # eval env
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
    elif args.scenario == "precise":
        from runners.pattern import Runner
    eval_runner = Runner(config)

    # load ckpt
    if args.method in ["mutation-kmeans", "map-elites"]:
        eval_runner.emulator_load()

    best_CRs = eval_procedure(args, eval_runner)

    result_fpath = os.path.join(method_dir, "result.csv")
    result = {
        "best_CRs": best_CRs
    }
    dict2csv(result, result_fpath)