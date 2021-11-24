#!/bin/bash

rm -rf results_dev

#### hyperparams ###
CUDA_VISIBLE_DEVICES=0
### env params ###
# shared
n_BM=50
n_GU=5
n_ABS=5
p_t=0
p_r=-85 # path_loss==85, ==> R_2D_NLoS=35.5, R_2D_LoS=187.6

# pattern only
granularity=15.625

### prepare params ###
scenario=pattern
name_addon=dev
file_episode_limit=50
method=mutation-kmeans
num_base_emulator_epochs=5
base_emulator_batch_size=32
emulator_batch_size=32
emulator_val_batch_size=32
planning_batch_size=32
num_env_episodes=1000
emulator_replay_size=1000



# generate terrain file
# for n in 0 ${n_BM}
# do
#     python gen_terrain.py --n_BM $n
# done

# pretrain
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python pretrain.py \
--n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
--p_t ${p_t} --p_r ${p_r} \
--granularity ${granularity} \
--scenario ${scenario} --name_addon ${name_addon} \
--splits 50 50 5 \
--method ${method} \
--file_episode_limit ${file_episode_limit} \
--num_base_emulator_epochs ${num_base_emulator_epochs} \
--base_emulator_batch_size ${base_emulator_batch_size} \
--emulator_batch_size ${emulator_batch_size} \
--emulator_val_batch_size ${emulator_val_batch_size} \
--planning_batch_size ${planning_batch_size} \
--num_env_episodes ${num_env_episodes} \
--emulator_replay_size ${emulator_replay_size}

# train
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python train.py \
--n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
--p_t ${p_t} --p_r ${p_r} \
--granularity ${granularity} \
--scenario ${scenario} --name_addon ${name_addon} \
--splits 50 50 5 \
--method ${method} \
--file_episode_limit ${file_episode_limit} \
--num_base_emulator_epochs ${num_base_emulator_epochs} \
--base_emulator_batch_size ${base_emulator_batch_size} \
--emulator_batch_size ${emulator_batch_size} \
--emulator_val_batch_size ${emulator_val_batch_size} \
--planning_batch_size ${planning_batch_size} \
--num_env_episodes ${num_env_episodes} \
--emulator_replay_size ${emulator_replay_size}

# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python train.py \
# --n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
# --p_t ${p_t} --p_r ${p_r} \
# --granularity ${granularity} \
# --scenario ${scenario} --name_addon ${name_addon} \
# --splits 50 50 5 \
# --method ${method} \
# --file_episode_limit ${file_episode_limit} \
# --num_base_emulator_epochs ${num_base_emulator_epochs} \
# --base_emulator_batch_size ${base_emulator_batch_size} \
# --emulator_batch_size ${emulator_batch_size} \
# --emulator_val_batch_size ${emulator_val_batch_size} \
# --planning_batch_size ${planning_batch_size} \
# --num_env_episodes ${num_env_episodes} \
# --emulator_replay_size ${emulator_replay_size} \
# --use_preload

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python eval.py \
--n_BM ${n_BM} --n_GU ${n_GU} --n_ABS ${n_ABS} \
--p_t ${p_t} --p_r ${p_r} \
--granularity ${granularity} \
--scenario ${scenario} --name_addon ${name_addon} \
--splits 50 50 5 \
--method ${method} \
--file_episode_limit ${file_episode_limit} \
--num_base_emulator_epochs ${num_base_emulator_epochs} \
--base_emulator_batch_size ${base_emulator_batch_size} \
--emulator_batch_size ${emulator_batch_size} \
--emulator_val_batch_size ${emulator_val_batch_size} \
--planning_batch_size ${planning_batch_size} \
--num_env_episodes ${num_env_episodes} \
--emulator_replay_size ${emulator_replay_size} \
--use_preload