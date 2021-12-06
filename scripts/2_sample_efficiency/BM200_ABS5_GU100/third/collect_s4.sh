#!/bin/bash

seed=4

n_BM=200



scenario=pattern
name_addon=2_sample_efficiency
collect_strategy=third

python pretrain_collect.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--splits 500 5_000 40_000 \
--file_episode_limit 20_000 \
--seed ${seed} \
--n_BM ${n_BM}



