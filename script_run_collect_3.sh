#!/bin/bash
# TODO: go to pretrain to comment out train and test
#### hyperparams ###
seed=3                          # TODO: change seed for each collect
### env params ###

# pattern only
granularity=15.625

### prepare params ###
name_addon=run
scenario=pattern

# pretrain
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain.py \
--granularity ${granularity} \
--name_addon ${name_addon} --scenario ${scenario} \
--splits 40_000 10_000 100 \
--seed ${seed}