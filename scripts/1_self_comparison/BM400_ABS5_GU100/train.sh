#!/bin/bash

CUDA_VISIBLE_DEVICES=1
seed=0

n_BM=400



scenario=pattern
name_addon=1_self_comparison
collect_strategy=default

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain_train.py \
--scenario ${scenario} --name_addon ${name_addon} \
--seed ${seed} \
--n_BM ${n_BM}
