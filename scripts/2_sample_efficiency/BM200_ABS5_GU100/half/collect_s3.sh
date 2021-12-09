#!/bin/bash

seed=3

n_BM=200



scenario=pattern
name_addon=2_sample_efficiency
collect_strategy=half

python pretrain_collect.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--splits 600 6_000 60_000 \
--file_episode_limit 30_000 \
--seed ${seed} \
--n_BM ${n_BM}