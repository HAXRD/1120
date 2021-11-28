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
    parser.add_argument("--n_BM", type=int, default=50,
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
    parser.add_argument("--p_t", type=float, default=0.,
                        help="maximum transmit power.")
    parser.add_argument("--p_r", type=float, default=-87,
                        help="minimum receive power.")

    # precise only
    # TODO: add more

    # pattern only
    parser.add_argument("--granularity", type=float, default=15.625,
                        help="pattern side length.")
    parser.add_argument("--normalize_pattern", action="store_true", default=False,
                        help="by default not normalize GU/CGU patterns by # of on-GUs.")

    ####### prepare params #######
    parser.add_argument("--scenario", type=str, required=True,
                        help="either 'precise' or 'pattern'.")
    parser.add_argument("--method", type=str, default="",
                        help="among ['', 'naive-kmeans', 'mutation-kmeans', 'map-elites'].")
    parser.add_argument("--seed", type=int, default=2021,
                        help="random seed for numpy&torch")
    parser.add_argument("--cuda", action="store_false", default=True,
                        help="by default 'True', use GPU to train.")
    parser.add_argument("--name_addon", type=str, default="",
                        help="naming addon to differentiate different experiments.")
    parser.add_argument("--use_eval", action="store_true", default=False,
                        help="by default not doing evaluation.")

    ####### pattern only #######
    ## emulator φ params
    parser.add_argument("--emulator_net_size", type=str, default="small")
    parser.add_argument("--splits", type=int, nargs="+",
                        help="# of episodes for different sets when training emulator.")
    parser.add_argument("--file_episode_limit", type=int, default=50000,
                        help="# of maximum episode per npz file.")
    parser.add_argument("--emulator_batch_size", type=int, default=128,
                        help="batch size for training emulator.")
    parser.add_argument("--num_emulator_epochs", type=int, default=500,
                        help="# of epochs to train emulator.")
    parser.add_argument("--num_emulator_tolerance_epochs", type=int, default=5,
                        help="# of epochs to break the training loop after seeing non-decreasing validation loss (to avoid potential overfitting) for emulator.")
    parser.add_argument("--emulator_grad_clip_norm", action="store_false", default=True,
                        help="by default clip grad norm when training emulator.")
    # emulator training params
    parser.add_argument("--emulator_lr", type=float, default=1.e-4,
                        help="lr for emulator.")
    ## emulator φ params
    parser.add_argument("--num_episodes_per_trial", default=20,
                        help="# of episodes for each trial, between each trial, reset env.")

    ## planning methods
    parser.add_argument("--planning_batch_size", type=int, default=256,
                        help="batch size for planning.")

    # naive-kmeans
    # None

    # mutation-kmeans
    parser.add_argument("--num_mutation_seeds", type=int, default=32,
                        help="# of different seeds to mutate from.")
    parser.add_argument("--num_mutations_per_seed", type=int, default=256,
                        help="# of mutations per seed to mutate.")
    parser.add_argument("--L", type=int, default=3,
                        help="# of outer rims to mutate (controls mutational variance).")

    # map-elites
    parser.add_argument("--iterations", type=int, default=256,
                        help="# of iterations for MAP-elites")
    parser.add_argument("--num_sample_individuals", type=int, default=32,
                        help="# of individuals to sample for each iteration.")

    ####### evaluation #######
    ## shared
    parser.add_argument("--num_eval_episodes", type=int, default=100,
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
        assert method in ["naive-kmeans", "mutation-kmeans", "map-elites"]

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