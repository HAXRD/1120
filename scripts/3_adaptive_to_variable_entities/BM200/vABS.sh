#!/bin/bash

CUDA_VISIBLE_DEVICES=1
seed=0

n_BM=200

scenario=pattern
name_addon=3_adaptive_to_variable_entities_var_ABS
collect_strategy=default

eval_emulator_fpath=results_3_adaptive_to_variable_entities/BM200_ABS10_GU100_variable_var_ABS/emulator_ckpts/best_emulator.pt


############### emulator accuracy ################
num_eval_episodes=2_000
# run emulator accuracy test
for n in 3 4
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_ABS ${n} &
    pids[${n}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
unset pids

# run emulator accuracy test
for n in 5 6
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_ABS ${n} &
    pids[${n}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
unset pids

# run emulator accuracy test
for n in 7 8
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_ABS ${n} &
    pids[${n}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
unset pids

# run emulator accuracy test
for n in 9 10
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_ABS ${n} &
    pids[${n}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
unset pids

# run emulator accuracy test
for n in 11 12
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python eval_emulator.py \
    --scenario ${scenario} --name_addon ${name_addon} \
    --collect_strategy ${collect_strategy} \
    --seed ${seed} \
    --eval_emulator_fpath ${eval_emulator_fpath} \
    --num_eval_episodes ${num_eval_episodes} \
    --n_BM ${n_BM} --n_ABS ${n} &
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
for n in 3 4 5 6 7 8 9 10 11 12
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
        --n_BM ${n_BM} --n_ABS ${n} \
        --method ${method}
    done
done


