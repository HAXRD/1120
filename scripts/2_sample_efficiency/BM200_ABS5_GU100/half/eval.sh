#!/bin/bash

CUDA_VISIBLE_DEVICES=2
seed=0

n_BM=200



scenario=pattern
name_addon=2_sample_efficiency
collect_strategy=half

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_1_self_comparison.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --splits 500 5_000 60_000 \
    --file_episode_limit 30_000 \
    --seed ${seed} \
    --n_BM ${n_BM} \
    --method ${method}
done