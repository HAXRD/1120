#!/bin/bash

rm -rf results_dev

#### hyperparams ###
CUDA_VISIBLE_DEVICES=0

n_BM=50
n_GU=100
n_ABS=5
p_t=0
p_r=-85

granularity=15.625

scenario=pattern    # TODO: change this to switch to 'precise'
method=naive-kmeans # TODO: change this to switch to 'mutation-kmeans' or 'map-elites'
name_addon=dev


#### emulator training ####
emulator_net_size=small
emulator_batch_size=32
num_emulator_epochs=5

planning_batch_size=32
num_eval_episodes=20

# generate terrain file
for n in 0 ${n_BM}
do
    python gen_terrain.py --n_BM $n
done


# pretrain
for str in pretrain
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python ${str}.py \
    --n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
    --p_t ${p_t} --p_r ${p_r} \
    --granularity ${granularity} \
    --scenario ${scenario} --method ${method} \
    --name_addon ${name_addon} \
    --splits 50 50 50 \
    --emulator_net_size ${emulator_net_size} \
    --emulator_batch_size ${emulator_batch_size} \
    --num_emulator_epochs ${num_emulator_epochs} \
    --planning_batch_size ${planning_batch_size} \
    --num_eval_episodes ${num_emulator_epochs}
done

for method in naive-kmeans mutation-kmeans map-elites
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval.py \
    --n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
    --p_t ${p_t} --p_r ${p_r} \
    --granularity ${granularity} \
    --scenario ${scenario} --method ${method} \
    --name_addon ${name_addon} \
    --splits 50 50 50 \
    --emulator_net_size ${emulator_net_size} \
    --emulator_batch_size ${emulator_batch_size} \
    --num_emulator_epochs ${num_emulator_epochs} \
    --planning_batch_size ${planning_batch_size} \
    --num_eval_episodes ${num_emulator_epochs}
done