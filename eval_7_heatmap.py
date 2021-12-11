# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

"""
Create an eval env to perform a certain method for 1 episodes,
store only the planning P_ABS patterns in map form along with
its actual CR map (by interacting with the eval env).
"""

import os
import torch
import numpy as np
import random

from pathlib import Path
from tqdm import tqdm

from common import run_preparation, make_env, dict2pkl, binary_search
from runners.pattern import Runner
from scipy.spatial import distance
from eval_shared import temp_seed


def _heatmap_procedure(args, runner):
    """
    :return: (
        list_solutions,
        list_true_performances
    )
    """

    # specs
    episodes = args.num_eval_episodes
    num_episodes_per_trial = args.num_episodes_per_trial
    num_mutation_seeds = args.num_mutation_seeds
    num_mutations_per_seed = args.num_mutations_per_seed
    iterations = args.iterations
    n_sample_individuals = args.num_sample_individuals
    n_step_explore = args.n_step_explore
    n_step_serve = args.n_step_serve
    n_step = n_step_explore + n_step_serve
    K = args.K
    n_ABS = args.n_ABS
    top_k = runner.top_k
    render = args.render
    assert n_step_explore == top_k

    """get map related features"""
    bin_means, bin_stds = runner.bin_means, runner.bin_stds
    ft_bins = runner.ft_bins

    def _map_x_to_b(x):
        """
        Map x coordinates to feature space dimensions.
        :param x: (nparray) genotype of a solution.
        :return: (tuple) phenotype of the solution
        """

        # get P_ABSs' indices
        ABS_2D_coords = []
        for i in range(K):
            for j in range(K):
                for _ in range(int(x[i, j])):
                    ABS_2D_coords.append([float(i), float(j)])

        dist_matrix = distance.cdist(ABS_2D_coords, ABS_2D_coords, "euclidean").astype(np.float32)
        # process matrix
        tri_upper_no_diag = np.triu(dist_matrix, k=1)
        tri_upper_no_diag = tri_upper_no_diag.reshape(-1)
        dists = tri_upper_no_diag[np.abs(tri_upper_no_diag) > 1e-5]
        assert len(ABS_2D_coords) == n_ABS
        assert len(dists) == n_ABS * (n_ABS - 1) / 2

        mean = np.mean(dists)
        std = np.std(dists)

        i = binary_search(bin_means, mean)
        j = binary_search(bin_stds, std)

        # do check
        assert bin_means[i] <= mean
        if i < len(bin_means) - 1:
            assert mean < bin_means[i + 1]
        assert bin_stds[j] <= std
        if j < len(bin_stds) - 1:
            assert std < bin_stds[j + 1]
        return (
            i, j
        )

    def _init_map():
        solutions = np.empty(ft_bins, dtype=object)
        performances = np.full(ft_bins, -np.inf, dtype=np.float32)
        return (
            solutions, performances
        )

    # return variables
    list_solutions = []
    list_true_performances = []

    # start eval for pattern-style
    for _episode in tqdm(range(episodes)):

        solutions, true_performances = _init_map()

        # reset or walk
        if _episode % num_episodes_per_trial == 0:
            runner.env.reset()
        else:
            runner.env.walk()
        runner.env.render(render)
        P_GU = runner.env.get_P_GU()

        # plan with different methods
        if runner.method == "naive-kmeans":
            batch_size, all_planning_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, top_k, 0)
        elif runner.method == "mutation-kmeans":
            batch_size, all_planning_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, num_mutation_seeds, num_mutations_per_seed)
        elif runner.method == "map-elites":
            batch_size, all_planning_P_ABSs = runner.map_elites(top_k, P_GU, iterations, n_sample_individuals)

        # interact with env with `all_planning_P_ABSs`
        for _P_ABS in all_planning_P_ABSs:
            runner.env.step(_P_ABS)
            runner.env.render(render)
            _P_CGU = runner.env.get_P_CGU()
            _CR = np.sum(_P_CGU) / np.sum(P_GU)

            i, j = _map_x_to_b(_P_ABS)
            true_performances[i, j] = _CR
            solutions[i, j] = _P_ABS

        print(true_performances.reshape(-1)[true_performances.reshape(-1) > -float('inf')])
        list_true_performances.append(true_performances)
        list_solutions.append(solutions)

    return (
        list_solutions,
        list_true_performances
    )


def collect_heatmap(args, runner):
    """
    Collect necessary information to draw heatmap.
    """

    print(f"[eval | heatmap collecting] start")

    with temp_seed(args.seed + 20212021):

        list_solutions, list_true_performances = _heatmap_procedure(args, runner)

    print(f"[eval | heatmap collecting] end")

    return (
        list_solutions,
        list_true_performances
    )

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

    assert args.scenario == "pattern"
    eval_runner = Runner(config)

    # load best emulator ckpt
    if args.method in ["mutation-kmeans", "map-elites"]:
        eval_emulator_fpath = args.eval_emulator_fpath
        eval_runner.emulator_load(eval_emulator_fpath)

    list_solutions, list_true_performances = collect_heatmap(args, eval_runner)

    heatmaps_data = {
        "list_solutions": list_solutions,
        "list_true_performances": list_true_performances
    }

    # store to pkl
    pkl_fpath = os.path.join(method_dir, f"heatmaps_data.pkl")
    dict2pkl(heatmaps_data, pkl_fpath)
    print(f"heatmaps saved to '{pkl_fpath}'")
