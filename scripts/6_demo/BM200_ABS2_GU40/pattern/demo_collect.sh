#!/bin/bash

CUDA_VISIBLE_DEVICES=2
seed=0

n_BM=100
n_ABS=2
n_GU=40

eval_emulator_fpath=results_5_mbp_vs_drl/BM100_ABS2_GU40_default/emulator_ckpts/best_emulator.pt
num_eval_episodes=2

scenario=pattern
name_addon=6_demo
collect_strategy=default

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_demo.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --splits 600 6_000, 120_000 \
    --file_episode_limit 60_000 \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_ABS ${n_ABS} --n_GU ${n_GU} \
    --method ${method}
done