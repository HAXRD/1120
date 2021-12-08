#!/bin/bash

seed=2

n_BM=200



scenario=pattern
name_addon=3_adaptive_to_variable_entities
collect_strategy=variable
n_ABS=10

python pretrain_collect.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--splits 100 1_000 20_000 \
--file_episode_limit 10_000 \
--seed ${seed} \
--n_BM ${n_BM} \
--variable_n_ABS --n_ABS ${n_ABS}

