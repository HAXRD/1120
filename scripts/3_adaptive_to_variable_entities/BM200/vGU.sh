#!/bin/bash

CUDA_VISIBLE_DEVICES=2
seed=0

n_BM=200

scenario=pattern
name_addon=3_adaptive_to_variable_entities_var_GU
collect_strategy=default

eval_emulator_fpath=results_3_adaptive_to_variable_entities/BM200_ABS5_GU200_variable_var_GU/emulator_ckpts/best_emulator.pt


############### emulator accuracy ################
num_eval_episodes=2_000
# run emulator accuracy test
for n in 75 100 125
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
for n in 150 175 200
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
for n in 75 100 125 150 175 200
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


