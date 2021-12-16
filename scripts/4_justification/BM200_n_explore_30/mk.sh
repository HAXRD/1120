#!/bin/bash

CUDA_VISIBLE_DEVICES=1
seed=0

n_BM=200

scenario=pattern
name_addon=4_justification_n_explore_30
collect_strategy=default

eval_emulator_fpath=results_1_self_comparison/BM200_ABS5_GU100_default/emulator_ckpts/best_emulator.pt
num_eval_episodes=100
num_episodes_per_trial=1
n_step_explore=30
n_step_serve=70

method=mutation-kmeans


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python eval_4_justification.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--seed ${seed} \
--eval_emulator_fpath ${eval_emulator_fpath} \
--num_eval_episodes ${num_eval_episodes} \
--num_episodes_per_trial ${num_episodes_per_trial} \
--n_BM ${n_BM} \
--method ${method} \
--n_step_explore ${n_step_explore} --n_step_serve ${n_step_serve}