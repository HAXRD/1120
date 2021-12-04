#!/bin/bash

seed=5

n_BM=200



scenario=pattern
name_addon=1_self_comparison
collect_strategy=default

python pretrain_collect.py \
--scenario ${scenario} --name_addon ${name_addon} \
--splits 100_000 5_000 500 \
--seed ${seed} \
--n_BM ${n_BM}
