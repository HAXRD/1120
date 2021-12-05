#!/bin/bash

CUDA_VISIBLE_DEVICES=0
seed=0

n_BM=300



scenario=pattern
name_addon=1_self_comparison

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_1_self_comparison.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --seed ${seed} \
    --n_BM ${n_BM} \
    --method ${method}
done