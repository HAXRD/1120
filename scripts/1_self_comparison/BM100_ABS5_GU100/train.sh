#!/bin/bash

CUDA_VISIBLE_DEVICES=0
seed=0

n_BM=100



scenario=pattern
name_addon=1_self_comparison

# bak files
cp -r results_${name_addon} results_${name_addon}_bak
echo "back up original files"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain_train.py \
--scenario ${scenario} --name_addon ${name_addon} \
--seed ${seed} \
--n_BM ${n_BM}
