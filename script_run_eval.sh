#!/bin/bash

#### hyperparams ###
CUDA_VISIBLE_DEVICES=0
seed=0
### env params ###
# shared
n_BM=50
p_t=0
p_r=-85 # path_loss==85, ==> R_2D_NLoS=35.5, R_2D_LoS=187.6

# pattern only
granularity=15.625

### prepare params ###
name_addon=run
scenario=pattern
method=mutation-kmeans

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval.py \
    --n_BM ${n_BM} --p_t ${p_t} --p_r ${p_r} \
    --granularity ${granularity} \
    --name_addon ${name_addon} --scenario ${scenario} --method ${method} \
    --splits 300_000 30_000 300 \
    --seed ${seed} \
    --num_eval_episodes 100
done
