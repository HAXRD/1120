#!/bin/bash

CUDA_VISIBLE_DEVICES=0      # TODO: change CUDA for different experiment

n_BM=0                      # TODO: change this for different experiment

scenario=precise
name_addon=run

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python train.py \
--n_BM ${n_BM} \
--scenario ${scenario} \
--name_addon ${name_addon}
