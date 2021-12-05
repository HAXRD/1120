#!/bin/bash

seed=0

n_BM=100



scenario=pattern
name_addon=1_self_comparison
collect_strategy=default

python pretrain_train.py \
--scenario ${scenario} --name_addon ${name_addon} \
--seed ${seed} \
--n_BM ${n_BM}
