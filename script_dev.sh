#!/bin/bash

rm -rf results_dev

CUDA_VISIBLE_DEVICES=0

n_BM=100
n_GU=10
n_ABS=3

granularity=15.625

scenario=pattern
name_addon=dev

collect_strategy=default

emulator_batch_size=32
num_emulator_epochs=5

planning_batch_size=32
num_eval_episodes=20

# pretrain
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain.py \
--n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
--granularity ${granularity} \
--scenario ${scenario} \
--name_addon ${name_addon} \
--splits 10 10 10 \
--emulator_batch_size ${emulator_batch_size} \
--num_emulator_epochs ${num_emulator_epochs}

# test emulator
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python test_emulator.py \
--n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
--granularity ${granularity} \
--scenario ${scenario} \
--name_addon ${name_addon} \
--emulator_batch_size ${emulator_batch_size} \
--random_on_off --p_on 0.8

# bundle
for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval.py \
    --n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
    --granularity ${granularity} \
    --scenario ${scenario} --method ${method} \
    --name_addon ${name_addon} \
    --emulator_batch_size ${emulator_batch_size} \
    --planning_batch_size ${planning_batch_size} \
    --num_eval_episodes ${num_eval_episodes}
done