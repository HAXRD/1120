#!/bin/bash

#### hyperparams ###
CUDA_VISIBLE_DEVICES=1          # TODO: change CUDA for different server
seed=0
### env params ###

# pattern only
granularity=15.625

### prepare params ###
name_addon=run
scenario=pattern

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval.py \
    --granularity ${granularity} \
    --name_addon ${name_addon} --scenario ${scenario} \
    --splits 500_000 50_000 500 \
    --seed ${seed}
    --method ${method} \
done
