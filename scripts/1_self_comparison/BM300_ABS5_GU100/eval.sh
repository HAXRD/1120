#!/bin/bash

CUDA_VISIBLE_DEVICES=0
seed=0

n_BM=300



scenario=pattern
name_addon=1_self_comparison
collect_strategy=default

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_1_self_comparison.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --splits 500 5_000 120_000 \
    --file_episode_limit 60_000 \
    --seed ${seed} \
    --n_BM ${n_BM} \
    --method ${method}
done