#!/bin/bash

CUDA_VISIBLE_DEVICES=0
seed=0

n_BM=300



scenario=pattern
name_addon=1_self_comparison
collect_strategy=default

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain_train.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--splits 600 6_000 120_000 \
--file_episode_limit 60_000 \
--seed ${seed} \
--n_BM ${n_BM}

# bak files
cp -r results_${name_addon} results_${name_addon}_bak
echo "back up original files"