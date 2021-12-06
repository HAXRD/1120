# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import time
import torch
import numpy as np
import random
import contextlib

from tqdm import tqdm

from common import make_env

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

def eval_procedure(args, runner, RENDER):
    """
    Evaluation procedure.
    """

    print(f"[eval] start")

    with temp_seed(args.seed + 20212021):

        if args.scenario == "pattern":
            episodes_CRs = _pattern_procedure(args, runner, RENDER)
        elif args.scenario == "precise":
            pass

        episodes_mean_CRs = np.mean(episodes_CRs, axis=1)
        overall_mean_CR = np.mean(episodes_CRs)
        print(f"----------------")
        print(f"[eval | overall_mean_CR] {overall_mean_CR}")
        print(f"----------------")

    # convert to list
    episodes_CRs = episodes_CRs.tolist()
    episodes_mean_CRs = episodes_mean_CRs.tolist()

    print(f"[eval] end")
    return (
        episodes_CRs, episodes_mean_CRs, overall_mean_CR
    )

def _pattern_procedure(args, runner, RENDER):
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


def test_emulator(args, device):
    """
    Load emulator ckpt from `args.eval_emulator_fpath`, then
    compare `P_CGU` and `P_rec_CGU` to get errors.

    :return: (
        mean_abs_elem_error: (float),
        mean_abs_CR_error:   (float)
    )
    """

    """Preparation"""
    # useful params
    eval_emulator_fpath = args.eval_emulator_fpath
    episodes = args.num_eval_episodes
    num_episodes_per_trial = args.num_episodes_per_trial
    render = args.render

    """Load emulator"""
    from algorithms.emulator import Emulator
    emulator = Emulator(args, device)
    best_emulator_fpath = eval_emulator_fpath
    emulator_state_dict = torch.load(best_emulator_fpath)
    emulator.model.load_state_dict(emulator_state_dict)
    print(f"[test_emulator] loaded ckpt from '{best_emulator_fpath}'.")

    """Env"""
    env = make_env(args, "eval")

    def _compute_metrics(P_GU, P_ABS, P_CGU):
        P_GUs  = torch.FloatTensor(P_GU).unsqueeze(0).unsqueeze(0).to(device)
        P_ABSs = torch.FloatTensor(P_ABS).unsqueeze(0).unsqueeze(0).to(device)
        P_CGUs = torch.FloatTensor(P_CGU).unsqueeze(0).unsqueeze(0).to(device)

        P_rec_CGUs = emulator.model.predict(P_GUs, P_ABSs)

        pred_error = torch.sum(torch.abs(P_rec_CGUs - P_CGUs)).cpu().numpy()
        print(f"\Sum|P_rec_CGUs - P_CGUs| = {pred_error}")

        CR = torch.sum(P_CGUs) / torch.sum(P_GUs)
        pCR = torch.sum(P_rec_CGUs) / torch.sum(P_GUs)

        pred_CR_error = torch.sum(torch.abs(pCR - CR)).cpu().numpy()
        print(f"|{pCR} - {CR}| == {pred_CR_error}")

        return (
            pred_error, pred_CR_error
        )

    """Start interaction to get transitions to compare on live"""
    total_size = 0
    total_abs_elem_error = 0
    total_abs_CR_error = 0
    for _episode in tqdm(range(episodes)):

        # totally random
        if _episode % num_episodes_per_trial == 0:
            env.reset()
        else:
            env.walk()
        env.render(render)

        P_GU, P_ABS, P_CGU = env.get_all_Ps()
        pred_error, pred_CR_error = _compute_metrics(P_GU, P_ABS, P_CGU)
        total_abs_elem_error += pred_error
        total_abs_CR_error += pred_CR_error
        total_size += 1

        # kmeans
        kmeans_P_ABS = env.find_KMEANS_P_ABS()
        env.step(kmeans_P_ABS)
        env.render(render)

        P_GU, P_ABS, P_CGU = env.get_all_Ps()
        pred_error, pred_CR_error = _compute_metrics(P_GU, P_ABS, P_CGU)
        total_abs_elem_error += pred_error
        total_abs_CR_error += pred_CR_error
        total_size += 1

    mean_abs_elem_error = total_abs_elem_error / total_size
    mean_abs_CR_error = total_abs_CR_error / total_size

    print(f"mean_abs_elem_error: {mean_abs_elem_error}, mean_abs_CR_error: {mean_abs_CR_error}")
    return (
        mean_abs_elem_error, mean_abs_CR_error
    )

def _justification_procedure(args, runner):
    """
    :return percentage_dict: {
        "episode": episodes
        "top_1": top_1_percentages,
        "top_3": top_3_percentages,
        "top_5": top_5_percentages,
        "top_10": top_10_percentages,
        "top_k": top_k_percentages
    }
    """

    assert args.method in ["mutation-kmeans", "map-elites"]

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

    def _compute_containing_top_x_percentage(sorted_idcs, top_x):
        return np.sum((sorted_idcs < top_x).astype(np.float32)) / top_x

    top_1_percentages = []
    top_3_percentages = []
    top_5_percentages = []
    top_10_percentages = []
    top_k_percentages = []

    # start eval for pattern-style
    for _episode in tqdm(range(episodes)):

        # reset or walk
        if _episode % num_episodes_per_trial == 0:
            runner.env.reset()
        else:
            runner.env.walk()
        runner.env.render(render)
        P_GU = runner.env.get_P_GU()

        # plan with different methods
        if runner.method == "mutation-kmeans":
            batch_size, all_planning_P_ABSs = runner.mutation_kmeans_planning(top_k, P_GU, num_mutation_seeds, num_mutations_per_seed)
        elif runner.method == "map-elites":
            batch_size, all_planning_P_ABSs = runner.map_elites(top_k, P_GU, iterations, n_sample_individuals)

        # interact with env with `all_planning_P_ABSs`
        bz = all_planning_P_ABSs.shape[0]
        all_P_CGUs = np.zeros((bz, K, K), dtype=np.float32)
        for _idx, _P_ABS in enumerate(all_planning_P_ABSs):
            runner.env.step(_P_ABS)
            runner.env.render(render)
            all_P_CGUs[_idx] = runner.env.get_P_CGU()

        all_CRs = np.sum(all_P_CGUs.reshape(bz, -1), axis=1) / runner.env.world.n_ON_GU
        sorted_idcs = np.argsort(-all_CRs, axis=-1)
        sorted_top_k_idcs = sorted_idcs[:top_k]

        top_1_percentages.append(_compute_containing_top_x_percentage(sorted_top_k_idcs, 1))
        top_3_percentages.append(_compute_containing_top_x_percentage(sorted_top_k_idcs, 3))
        top_5_percentages.append(_compute_containing_top_x_percentage(sorted_top_k_idcs, 5))
        top_10_percentages.append(_compute_containing_top_x_percentage(sorted_top_k_idcs, 10))
        top_k_percentages.append(_compute_containing_top_x_percentage(sorted_top_k_idcs, top_k))

    percentage_dict = {
        "episode": list(range(episodes)),
        "top_1": top_1_percentages,
        "top_3": top_3_percentages,
        "top_5": top_5_percentages,
        "top_10": top_10_percentages,
        "top_k": top_k_percentages
    }

    return percentage_dict


def justification(args, runner):
    """
    Justification about how much percentage that the emulator predicted
    `P_ABS`s are actually the top_x patterns.
    """

    print(f"[eval | justification] start")

    with temp_seed(args.seed + 20212021):

        assert args.scenario == "pattern"

        percentage_dict = _justification_procedure(args, runner)

    return percentage_dict