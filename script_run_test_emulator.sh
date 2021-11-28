#!/bin/bash

#### hyperparams ####
CUDA_VISIBLE_DEVICES=0
seed=0
### env params ###

# pattern only
granularity=15.625

### prepare params ###
name_addon=run
scenario=pattern

# test_emulator
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python test_emulator.py \
--granularity ${granularity} \
--name_addon ${name_addon} --scenario ${scenario} \
--seed ${seed} \
--random_on_off --p_on 0.8