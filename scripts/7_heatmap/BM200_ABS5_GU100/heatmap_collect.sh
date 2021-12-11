#!/bin/bash

CUDA_VISIBLE_DEVICES=2
seed=0

n_BM=200

eval_emulator_fpath=results_1_self_comparison/BM200_ABS5_GU100_default/emulator_ckpts/best_emulator.pt
num_eval_episodes=1

scenario=pattern
name_addon=7_heatmap
collect_strategy=default

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_7_heatmap.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --splits 600 6_000 120_000 \
    --file_episode_limit 60_000 \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} \
    --method ${method}
done