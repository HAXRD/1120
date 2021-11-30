#!/bin/bash
# TODO: go to pretrain to comment out train and test
# TODO: go to config to change collect_strategy

seed=4          # TODO: change seed for each collect

name_addon=run
scenario=pattern

collect_strategy=default

# pretrain collect
python pretrain.py \
--name_addon ${name_addon} --scenario ${scenario} \
--splits 50_000 5_000 500 \
--seed ${seed}