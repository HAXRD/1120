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
            planning_size, planning_P_ABSs = runner.naive_kmeans(top_k)
        elif runner.method == "mutation-kmeans":
            planning_size, planning_P_ABSs = runner.mutation_kmeans(num_mutation_seeds, num_mutations_per_seed)
            top_CR_info += "M-K"
        elif runner.method == "map-elite":
            pass
        assert planning_P_ABSs.shape == (planning_size, K, K)

        # only do planning for "mutation-kmeans" or "map-elite"
        if runner.method in ["mutation-kmeans", "map-elite"]:

            # preparation for planning
            repeated_P_GUs = np.repeat(np.expand_dims(P_GU, 0), planning_size, axis=0)
            assert repeated_P_GUs.shape == (planning_size, K, K)

            # use emulator to select top_k transitions
            _, top_k_P_ABSs, _ = runner.plan(repeated_P_GUs, planning_P_ABSs) # (top_k, K, K)
        else:
            top_k_P_ABSs = planning_P_ABSs

        # interact with env
        top_k_P_CGUs = np.zeros((top_k, K, K), dtype=np.float32)
        for _idx, _P_ABS in enumerate(top_k_P_ABSs):
            runner.env.step(_P_ABS)
            runner.env.render(RENDER)
            top_k_P_CGUs[_idx] = runner.env.get_P_CGU()

        top_k_CRs = np.sum(top_k_P_CGUs.reshape(top_k, -1), axis=1) / runner.env.world.n_ON_GU
        top_CR = np.max(top_k_CRs)

        best_CRs.append(top_CR)
        top_CR_info += f" {top_CR}"

        end = time.time()
        print(f"[eval | top {top_CR_info} | |{end - start}s] {_episode + 1}/{episodes}.")

    return best_CRs


def eval_procedure(args, runner, RENDER="human", TENSORBOARD=False, curr_episode=None):
    """
    Evaluation procedure.
    """

    print(f"[eval{' | ' + str(curr_episode) if not curr_episode == None else ''}] start.")
    with temp_seed(args.seed + 20212021):

        if args.scenario == "pattern":
            best_CRs = pattern_procedure(args, runner, RENDER)
        elif args.scenario == "precise":
            pass

        mean_CR = np.mean(best_CRs)
        print(f"-------------")
        print(f"[eval | mean_CR] {mean_CR}.")
        print(f"-------------")

        if TENSORBOARD:
            assert curr_episode is not None
            runner.writer.add_scalar("mean_CR", mean_CR, curr_episode)

    print(f"[eval{' | ' + str(curr_episode) if not curr_episode == None else ''}] end.")
    return best_CRs

def evaluation(args, eval_runner, test_q, done_training):
    """
    Target function for subprocess evaluation.
    """

    plot = {
        "best_CRs_list": [],
        "mean_CRs": [],
        "episode_i": [],
    }

    best_eval_avg_CR = 0.
    best_episode_i = -1

    while True:
        if not test_q.empty():
            curr_episode, emulator_fpath = test_q.get()
            eval_runner.emulator_load(emulator_fpath)

            print(f"{curr_episode}, {emulator_fpath}")

            best_CRs = eval_procedure(args, eval_runner, RENDER="non-display", TENSORBOARD=True, curr_episode=curr_episode)

            # record to plot
            plot["best_CRs_list"].append(best_CRs)
            plot["mean_CRs"].append(np.mean(best_CRs))
            plot["episode_i"].append(curr_episode)

            dict2csv(plot, os.path.join(args.method_dir, "curve.csv"))

            # update best
            avg_CR = np.mean(best_CRs)
            if avg_CR > best_eval_avg_CR:
                best_eval_avg_CR = avg_CR
                best_episode_i = curr_episode

            if done_training.value and test_q.empty():
                with open(os.path.join(args.method_dir, "eval_think_best.txt"), "wb") as f:
                    f.write(f"best_episode_i: {best_episode_i}")
                break


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
    assert method_dir.exists()
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
        "device": device,
        "writer": None
    }

    if args.scenario == "pattern":
        from runners.pattern import Runner
    elif args.scenario == "precise":
        from runners.pattern import Runner
    eval_runner = Runner("EvalRunner", config)

    # load ckpt
    if args.method in ["mutation-kmeans", "map-elite"]:
        eval_runner.emulator_load(os.path.join(method_dir, "emulator_ckpts", "best_emulator.pt"))

    best_CRs = eval_procedure(args, eval_runner)