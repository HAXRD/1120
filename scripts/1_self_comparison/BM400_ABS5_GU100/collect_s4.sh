#!/bin/bash

seed=4

n_BM=400



scenario=pattern
name_addon=1_self_comparison
collect_strategy=default

python pretrain_collect.py \
--scenario ${scenario} --name_addon ${name_addon} \
--collect_strategy ${collect_strategy} \
--splits 500 5_000 120_000 \
--file_episode_limit 60_000 \
--seed ${seed} \
--n_BM ${n_BM}
