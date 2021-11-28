#!/bin/bash
# TODO: go to pretrain to comment out train and test
#### hyperparams ###
CUDA_VISIBLE_DEVICES=1          # TODO: change CUDA for different server
seed=0                          # TODO: change seed for each collect
### env params ###

# pattern only
granularity=15.625

### prepare params ###
name_addon=run
scenario=pattern

# generate terrain file
python gen_terrain.py --n_BM 0
python gen_terrain.py          # TODO: go to gen_terrain.py to change default n_BM

# pretrain
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain.py \
--granularity ${granularity} \
--name_addon ${name_addon} --scenario ${scenario} \
--splits 100_000 10_000 100 \
--seed ${seed}