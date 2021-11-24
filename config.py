# Copyright (c) 2021, Xu Chen, FUNLab, Xiamen University
# All rights reserved.

import os
from pathlib import Path
import argparse

def get_config():
    """
    The configuration parser for hyperparameters of experiment.
    """
    parser = argparse.ArgumentParser(
        description="Site-specific UAV Tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ####### env params #######
    # shared
    parser.add_argument("--world_len", type=float, default=1000,
                        help="side length of the square shape world.")
    parser.add_argument("--episode_duration", type=float, default=10,
                        help="duration of 1 episode.")
    parser.add_argument("--n_step_explore", type=int, default=32,
                        help="# of steps to explore for each episode.")
    parser.add_argument("--n_step_serve", type=int, default=64,
                        help="# of steps to serve for each episode.")
    parser.add_argument("--n_BM", type=int, required=True,
                        help="# of BMs for Site-specific environment.")
    parser.add_argument("--n_ABS", type=int, default=5,
                        help="# of ABSs.")
    parser.add_argument("--n_GU", type=int, default=100,
                        help="# of GUs.")
    parser.add_argument("--v_ABS", type=float, default=25,
                        help="velocity of ABSs.")
    parser.add_argument("--v_GU", type=float, default=2,
                        help="velocity of GUs.")
    parser.add_argument("--h_ABS", type=float, default=90,
                        help="height of ABSs.")
    parser.add_argument("--h_GU", type=float, default=1,
                        help="height of GUs.")
    parser.add_argument("--render", type=str, default="human",
                        help="either 'non-display' or 'human'.")
    parser.add_argument("--random_on_off", action="store_true", default=False,
                        help="by default false, not use random on&off for GUs.")
    parser.add_argument("--p_on", type=float, default=1.,
                        help="probability of randomly enabling each GU.")

    parser.add_argument("--f_c", type=float, default=2.,
                        help="carrier frequency (GHz).")
    parser.add_argument("--p_t", type=float, required=True,
                        help="maximum transmit power.")
    parser.add_argument("--p_r", type=float, default=-100.,
                        help="minimum receive power.")

    # precise only
    # TODO: add more

    # pattern only
    parser.add_argument("--base_BMs_fname", type=str, default="terrain-0.mat",
                        help="file name of base BMs ('0' means there will be only LoS case) mat file.")
    parser.add_argument("--granularity", type=float, default=15.625,
                        help="pattern side length.")
    parser.add_argument("--normalize_pattern", action="store_true", default=False,
                        help="by default not normalize GU/CGU patterns by # of on-GUs.")

    ####### prepare params #######
    parser.add_argument("--scenario", type=str, required=True,
                        help="either 'precise' or 'pattern'.")
    parser.add_argument("--method", type=str, default="mutation-kmeans",
                        help="among ['', 'naive-kmeans', 'mutation-kmeans', 'map-elite'].")
    parser.add_argument("--seed", type=int, default=2021,
                        help="random seed for numpy&torch")
    parser.add_argument("--cuda", action="store_false", default=True,
                        help="by default 'True', use GPU to train.")
    parser.add_argument("--name_addon", type=str, default="",
                        help="naming addon to differentiate different experiments.")
    parser.add_argument("--use_eval", action="store_false", default=True,
                        help="by default evaluation.")

    ####### pattern only #######
    ## base emulator φ_0 params
    parser.add_argument("--splits", type=int, nargs="+",
                        help="# of episodes for different sets when training base emulator.")
    parser.add_argument("--file_episode_limit", type=int, default=50000,
                        help="# of maximum episode per npz file.")
    parser.add_argument("--base_emulator_batch_size", type=int, default=128,
                        help="batch size for training base emulator.")
    parser.add_argument("--num_base_emulator_epochs", type=int, default=500,
                        help="# of epochs to train base emulator.")
    parser.add_argument("--num_base_emulator_tolerance_epochs", type=int, default=5,
                        help="# of epochs to break the training loop after seeing non-decreasing validation loss (to avoid potential overfitting) for base emulator.")
    parser.add_argument("--base_emulator_grad_clip_norm", action="store_false", default=True,
                        help="by default clip grad norm when training base emulator.")

    ## emulator φ params
    parser.add_argument("--emulator_lr", type=float, default=1.e-4,
                        help="lr for emulator.")
    parser.add_argument("--num_env_episodes", type=int, default=10_000_000,
                        help="# of environment epsiodes to train.")
    parser.add_argument("--num_episodes_per_trial", default=100,
                        help="# of episodes for each trial, between each trial, reset env.")
    parser.add_argument("--emulator_batch_size", type=int, default=64,
                        help="batch size for training emulator.")
    parser.add_argument("--emulator_val_batch_size", type=int, default=512,
                        help="batch size for emulator validation.")
    parser.add_argument("--emulator_train_repeats", type=int, default=2,
                        help="number of repeats to train emulator with samples.")
    parser.add_argument("--num_emulator_tolerance_epochs", type=int, default=5,
                        help="# of epochs to break the training loop after seeing non-decreasing validation loss (this tries to avoid potential overfitting) for emulator.")
    parser.add_argument("--emulator_grad_clip_norm", action="store_false", default=True,
                        help="by default clip norm when training emulator.")

    ## planning methods
    parser.add_argument("--planning_batch_size", type=int, default=512,
                        help="batch size for planning.")

    # naive-kmeans

    # mutation-kmeans
    parser.add_argument("--num_mutation_seeds", type=int, default=32,
                        help="# of different seeds to mutate from.")
    parser.add_argument("--num_mutations_per_seed", type=int, default=256,
                        help="# of mutations per seed to mutate.")
    parser.add_argument("--L", type=int, default=3,
                        help="# of outer rims to mutate (controls mutational variance).")

    # map-elite

    ## replays
    parser.add_argument("--emulator_replay_size", type=int, default=1_000_000,
                        help="replay size for emulator memory.")
    parser.add_argument("--emulator_alpha", type=float, default=0.6,
                        help="alpha for emulator PER.")
    parser.add_argument("--emulator_beta", type=float, default=1.,
                        help="beta for emulator PER.")
    parser.add_argument("--train_2_val_ratio", type=int, default=10,
                        help="training set to validation set ratio in training emulator.")

    ## preload
    parser.add_argument("--use_preload", action="store_true", default=False,
                        help="use preload experience to avoid the long time collecting.")


    ####### interval #######
    parser.add_argument("--log_interval", type=int, default=1,
                        help="every # of episodes to log.")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="every # of episodes to evaluate.")

    ####### evaluation #######
    ## shared
    parser.add_argument("--num_eval_episodes", type=int, default=10_000,
                        help="# of episodes for evaluations.")

    # precise

    # pattern


    ####### additional parsing #######
    args = parser.parse_known_args()[0]

    K = int(args.world_len / args.granularity)

    BMs_fname = f"terrain-{args.n_BM}.mat"
    n_step = args.n_step_explore + args.n_step_serve
    step_duration = args.episode_duration / n_step
    run_dir = Path(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results{'' if args.name_addon == '' else '_' + args.name_addon}",
        f"{args.n_ABS}ABS_{args.n_GU}GU",
        f"{args.scenario}" + f"{'' if args.scenario == 'precise' else '_' + str(K) + 'K'}"
    ))
    method = args.method
    if args.scenario == "pattern":
        assert method in ["naive-kmeans", "mutation-kmeans", "map-elite"]

    parser.add_argument("--K", type=int, default=K,
                        help="K x K pattern.")

    parser.add_argument("--BMs_fname", type=str, default=BMs_fname,
                        help="file name of BMs mat file.")
    parser.add_argument("--n_step", type=int, default=n_step,
                        help="# of steps in total for each episode.")
    parser.add_argument("--step_duration", type=float, default=step_duration,
                        help="duration of a step.")
    parser.add_argument("--run_dir", default=run_dir,
                        help="`Path` object, specifying working directory.")

    return parser

if __name__ == "__main__":
    
    parser = get_config()
    args = parser.parse_args()