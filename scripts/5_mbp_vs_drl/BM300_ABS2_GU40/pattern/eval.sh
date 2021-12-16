#!/bin/bash

CUDA_VISIBLE_DEVICES=1
seed=0

n_BM=300
n_ABS=2
n_GU=40



scenario=pattern
name_addon=5_mbp_vs_drl
collect_strategy=default

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_1_self_comparison.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --splits 600 6_000 120_000 \
    --file_episode_limit 60_000 \
    --seed ${seed} \
    --n_BM ${n_BM} --n_ABS ${n_ABS} --n_GU ${n_GU} \
    --method ${method}
done