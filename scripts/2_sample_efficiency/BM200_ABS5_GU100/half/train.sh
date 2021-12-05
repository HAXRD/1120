#!/bin/bash

CUDA_VISIBLE_DEVICES=2
seed=0

n_BM=200



scenario=pattern
name_addon=2_sample_efficiency
collect_strategy=half

# bak files
cp -r results_${name_addon} results_${name_addon}_bak
echo "back up original files"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain_train.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--splits 500 5_000 60_000 \
--file_episode_limit 30_000 \
--seed ${seed} \
--n_BM ${n_BM}
