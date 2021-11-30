#!/bin/bash
# TODO: go to pretrain to comment out collect

CUDA_VISIBLE_DEVICES=0  # TODO: change CUDA for different server
seed=0

name_addon=run
scenario=pattern

# pretrain train & test
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain.py \
--name_addon ${name_addon} --scenario ${scenario} \
--seed ${seed}