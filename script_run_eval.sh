#!/bin/bash

CUDA_VISIBLE_DEVICES=0  # TODO: change CUDA for different server
seed=0

name_addon=run
scenario=pattern

# bundle

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval.py \
    --name_addon ${name_addon} --scenario ${scenario} \
    --seed ${seed} \
    --method ${method}
done