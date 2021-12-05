# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import time
import numpy as np

from tqdm import tqdm

def pattern_procedure(args, runner, RENDER):
    """
    :return episodes_CRs: shape=(episodes, n_step), contains each step's
    CR for every episode.
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
    top_k = runner.top_k
    assert n_step_explore == top_k

    # return variable
    episodes_CRs = np.zeros((episodes, n_step))

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
        top_CR_info += f" {top_CR}"

        # overwrite current episode with top CR
        episodes_CRs[_episode, :] = top_CR
        # overwrite previous top_k steps
        episodes_CRs[_episode, :batch_size] = top_k_CRs

        end = time.time()
        print(f"[eval | top {top_CR_info} | {end - start}s] finished!")

    return episodes_CRs
