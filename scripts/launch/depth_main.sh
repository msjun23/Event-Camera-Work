#!/bin/bash

CUDA_IDX='0'
NUM_PROC=1
cfg_path=/root/code/configs/depth/mvsec.yaml
save_dir=/root/code/save/neurips
exp_name='test'

CUDA_VISIBLE_DEVICES=${CUDA_IDX} \
torchrun --nproc_per_node=$NUM_PROC --master_port=$RANDOM \
../depth_main.py --cfg_path ${cfg_path} --save_dir ${save_dir}