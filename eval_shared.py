# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import time
import torch
import numpy as np
import pandas as pd
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

        n_episode, n_step = episodes_CRs.shape

        df = pd.DataFrame({
            "Episode": np.arange(n_episode).reshape(-1, 1).repeat(n_step, axis=1).reshape(-1).tolist(),
            "Step":    np.arange(n_step).reshape(1, -1).repeat(n_episode, axis=0).reshape(-1).tolist(),
            "CR":      episodes_CRs.reshape(-1).tolist(),
        })
        # We use string to store integer to avoid displaying issue on x-axis
        df["n_BM"]  = str(args.n_BM)
        df["n_ABS"] = str(args.n_ABS)
        df["n_GU"]  = str(args.n_GU)
        df["Collect Strategy"] = args.collect_strategy
        df["Method"] = args.method
        df["Explore:Serve"] = f"{args.n_step_explore:2d}:{args.n_step_serve:2d}"

        mean_df = pd.DataFrame({
            "CR": df["CR"].mean()
        }, index=["mean"])

        print(f"----------------")
        print(mean_df)
        print(f"----------------")

    print(f"[eval] end")
    return (
        df, mean_df
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

    return episodes_CRs


def test_emulator(args, device):
    """
    Load emulator ckpt from `args.eval_emulator_fpath`, then
    compare `P_CGU` and `P_rec_CGU` to get errors.
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

        abs_elem_wise_error = torch.sum(torch.abs(P_rec_CGUs - P_CGUs)).cpu().numpy()
        print(f"\Sum|P_rec_CGUs - P_CGUs| = {abs_elem_wise_error}")

        CR = torch.sum(P_CGUs) / torch.sum(P_GUs)
        pCR = torch.sum(P_rec_CGUs) / torch.sum(P_GUs)

        abs_CR_error = torch.sum(torch.abs(pCR - CR)).cpu().numpy()
        print(f"|{pCR} - {CR}| == {abs_CR_error}")

        return (
            abs_elem_wise_error, abs_CR_error
        )

    """Start interaction to get transitions to compare on live"""
    abs_elem_wise_errors = []
    abs_CR_errors = []
    for _episode in tqdm(range(episodes)):

        # totally random
        if _episode % num_episodes_per_trial == 0:
            env.reset()
        else:
            env.walk()
        env.render(render)

        P_GU, P_ABS, P_CGU = env.get_all_Ps()
        abs_elem_wise_error, abs_CR_error = _compute_metrics(P_GU, P_ABS, P_CGU)
        abs_elem_wise_errors.append(abs_elem_wise_error)
        abs_CR_errors.append(abs_CR_error)

        # kmeans
        kmeans_P_ABS = env.find_KMEANS_P_ABS()
        env.step(kmeans_P_ABS)
        env.render(render)

        P_GU, P_ABS, P_CGU = env.get_all_Ps()
        abs_elem_wise_error, abs_CR_error = _compute_metrics(P_GU, P_ABS, P_CGU)
        abs_elem_wise_errors.append(abs_elem_wise_error)
        abs_CR_errors.append(abs_CR_error)

    df = pd.DataFrame({
        "Absolute Elem-wise Error": abs_elem_wise_errors,
        "Absolute CR Error": abs_CR_errors
    })
    df["n_BM"] = str(args.n_BM)
    df["n_ABS"] = str(args.n_ABS)
    df["n_GU"] = str(args.n_GU)
    df["Collect Strategy"] = args.collect_strategy
    df["Method"] = args.method
    df["Explore:Serve"] = f"{args.n_step_explore:2d}:{args.n_step_serve:2d}"

    mean_df = pd.DataFrame({
        "Absolute Elem-wise Error": df["Absolute Elem-wise Error"].mean(),
        "Absolute CR Error": df["Absolute CR Error"].mean()
    }, index=["mean"])

    print(f"------------")
    print(f"{mean_df}")
    print(f"------------")

    return (
        df, mean_df
    )

def _justification_procedure(args, runner):
    """
    :return: (
        df, df_processed, mean_df_processed
    )
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

    def _compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_top_x_CRs, top_k):
        cnter = 0.
        for _cr in emulator_believed_top_k_CRs:
            if _cr in ground_truth_top_x_CRs:
                cnter += 1.
        return cnter / top_k

    top_1_percentages = []
    top_2_percentages = []
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

        all_CRs = np.sum(all_P_CGUs.reshape(bz, -1), axis=1) / runner.env.world.n_ON_GU     # sorted by emulator-believed order
        emulator_believed_top_k_CRs = all_CRs[:top_k] # top_k emulator believed CRs
        ground_truth_unique_top_k_CRs = sorted(list(set(all_CRs)), reverse=True)[:top_k] # top_k ground truth unique CRs

        top_1_percentages.append(_compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_unique_top_k_CRs[:1], top_k))
        top_2_percentages.append(_compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_unique_top_k_CRs[:2], top_k))
        top_3_percentages.append(_compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_unique_top_k_CRs[:3], top_k))
        top_5_percentages.append(_compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_unique_top_k_CRs[:5], top_k))
        top_10_percentages.append(_compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_unique_top_k_CRs[:10], top_k))
        top_k_percentages.append(_compute_containing_top_x_percentage(emulator_believed_top_k_CRs, ground_truth_unique_top_k_CRs[:top_k], top_k))

    df = pd.DataFrame({
        "Episode": list(range(episodes)),
        "Top 1": top_1_percentages,
        "Top 2": top_2_percentages,
        "Top 3": top_3_percentages,
        "Top 5": top_5_percentages,
        "Top 10": top_10_percentages,
        "Top k": top_k_percentages
    })
    df["n_BM"] = str(args.n_BM)
    df["n_ABS"] = str(args.n_ABS)
    df["n_GU"] = str(args.n_GU)
    df["Collect Strategy"] = args.collect_strategy
    df["Method"] = args.method
    df["Explore:Serve"] = f"{args.n_step_explore:2d}:{args.n_step_serve:2d}"

    df_processed = df.copy()
    df_processed["Top 1"] = (df_processed["Top 1"] > 0.).astype(float)
    df_processed["Top 2"] = (df_processed["Top 2"] > 0.).astype(float)
    df_processed["Top 3"] = (df_processed["Top 3"] > 0.).astype(float)
    df_processed["Top 5"] = (df_processed["Top 5"] > 0.).astype(float)
    df_processed["Top 10"] = (df_processed["Top 10"] > 0.).astype(float)
    df_processed["Top k"] = (df_processed["Top k"] > 0.).astype(float)

    mean_df_processed = pd.DataFrame({
        "Top 1": df_processed["Top 1"].mean(),
        "Top 2": df_processed["Top 2"].mean(),
        "Top 3": df_processed["Top 3"].mean(),
        "Top 5": df_processed["Top 5"].mean(),
        "Top 10": df_processed["Top 10"].mean(),
        "Top k": df_processed["Top k"].mean()
    }, index=["mean"])

    print(f"------------------")
    print(mean_df_processed)
    print(f"------------------")

    return (
        df, df_processed, mean_df_processed
    )


def justification(args, runner):
    """
    Justification about how much percentage that the emulator predicted
    `P_ABS`s are actually the top_x patterns.
    """

    print(f"[eval | justification] start")

    with temp_seed(args.seed + 20212021):

        assert args.scenario == "pattern"

        df, df_processed, mean_df_processed = _justification_procedure(args, runner)

    print(f"[eval | justification] end")

    return (
        df, df_processed, mean_df_processed
    )
