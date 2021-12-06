#!/bin/bash

CUDA_VISIBLE_DEVICES=1
seed=0

n_BM=200

scenario=pattern
name_addon=3_adaptive_to_variable_entities
collect_strategy=default

eval_emulator_fpath=results_1_self_comparison/BM200_ABS5_GU100_default/emulator_ckpts/best_emulator.pt


############### emulator accuracy ################
num_eval_episodes=2_000
# run emulator accuracy test
for n in 75 125
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_GU ${n} &
    pids[${n}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
unset pids

# run emulator accuracy test
for n in 150 200
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_GU ${n} &
    pids[${n}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
unset pids

############### method performance ################
num_eval_episodes=100
for n in 75 125 150 200
do
    for method in naive-kmeans mutation-kmeans map-elites
    do
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
        python eval_3_adaptive_to_variable_entities.py \
        --scenario ${scenario} --name_addon ${name_addon} \
        --collect_strategy ${collect_strategy} \
        --seed ${seed} \
        --eval_emulator_fpath ${eval_emulator_fpath} \
        --num_eval_episodes ${num_eval_episodes} \
        --n_BM ${n_BM} --n_GU ${n} \
        --method ${method}
    done
done


