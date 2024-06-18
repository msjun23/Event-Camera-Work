#!/bin/bash

cfg_path=/root/code/configs/depth/mvsec_s4.yaml
save_dir=/root/code/save
exp_name='test'

python ../stereo_depth_main.py --cfg_path ${cfg_path} --save_dir ${save_dir} --exp_name ${exp_name}