#!/bin/bash

CUDA_VISIBLE_DEVICES=0
seed=0

name_addon=run
scenario=pattern

# test emulator accuracy

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python test_emulator.py \
--name_addon ${name_addon} --scenario ${scenario} \
--seed ${seed}


CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python test_emulator.py \
--name_addon ${name_addon} --scenario ${scenario} \
--random_on_off --p_on=0.8 \
--seed ${seed}
