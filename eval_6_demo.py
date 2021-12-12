# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""
Create an eval env to perform a certain method for 1 episodes,
store only 2 collections of the entities statuses:
    1. GU location randomly generated, ABSs location random
    2. after algorithm computation, the 'optimal' ABS deployment.
"""

import os
import time
import torch
import numpy as np
import random

from pathlib import Path
from tqdm import tqdm

from common import run_preparation, make_env, dict2pkl
from eval_shared import temp_seed

def _demo_pattern_procedure(args, runner):

    # specs
    episodes = args.num_eval_episodes
    num_episodes_per_trial = args.num_episodes_per_trial
    num_mutation_seeds = args.num_mutation_seeds
    num_mutations_per_seed = args.num_mutations_per_seed
    iterations = args.iterations
    n_sample_individuals = args.num_sample_individuals
    K = args.K
    top_k = runner.top_k
    render = args.render

    entities_statuses = []

    # start eval for pattern-style
    for _episode in tqdm(range(episodes)):

        # reset or walk
        if _episode % num_episodes_per_trial == 0:
            runner.env.reset()
        else:
            runner.env.walk()
        runner.env.render(render)
        entities_statuses.append(runner.env.get_entities_statuses())
        P_GU = runner.env.get_P_GU()

        # plan with different methods
        if runner.method == "naive-kmeans":
            batch_size, all_planning_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, top_k, 0)
        elif runner.method == "mutation-kmeans":
            batch_size, all_planning_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, num_mutation_seeds, num_mutations_per_seed)
        elif runner.method == "map-elites":
            batch_size, all_planning_P_ABSs = runner.map_elites(top_k, P_GU, iterations, n_sample_individuals)
        top_k_P_ABSs = all_planning_P_ABSs[:batch_size]

        # interact with env
        top_k_P_CGUs = np.zeros((batch_size, K, K), dtype=np.float32)
        for _idx, _P_ABS in enumerate(top_k_P_ABSs):
            runner.env.step(_P_ABS)
            runner.env.render(render)
            top_k_P_CGUs[_idx] = runner.env.get_P_CGU()

        top_k_CRs = np.sum(top_k_P_CGUs.reshape(batch_size, -1), axis=1) / runner.env.world.n_ON_GU
        sorted_idcs = np.argsort(-top_k_CRs, axis=-1)
        top_k_P_ABSs = top_k_P_ABSs[sorted_idcs]

        top_P_ABS = top_k_P_ABSs[0]

        # perform the best P_ABS
        runner.env.step(top_P_ABS)
        runner.env.render(render)
        entities_statuses.append(runner.env.get_entities_statuses())
        print(f"performed best P_ABS for episode {_episode}")

    return entities_statuses

def _demo_precise_procedure(args, runner):
    return None
    pass

def demo(args, runner):
    """
    Collect necessary information to draw a demo.
    """

    print(f"[eval | demo] start")

    with temp_seed(args.seed + 20212021):

        # TODO:
        if args.scenario == "pattern":
            entities_statuses = _demo_pattern_procedure(args, runner)
        elif args.scenario == "precise":
            entities_statuses = _demo_precise_procedure(args, runner)
            pass

    print(f"[eval | demo] end")

    return entities_statuses

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
