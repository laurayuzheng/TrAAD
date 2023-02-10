#!/bin/bash

bash scripts/path_config.sh

################### MAP MODEL TRAINING ######################################

# map model
export ACCEL_MODEL="./models/phase1_ppo"
# export EXP_ID="test_ppo"

CUDA_VISIBLE_DEVICES=0 \
python -m carla_project.src.traffic_map_model \
--hack \
--dataset_dir=data/tiny \
--accel_model_path=${ACCEL_MODEL} \
--id=${EXP_ID} \


################### IMAGE MODEL TRAINING ######################################

# # img model
# export EXP_ID="ppo_lbc_img"
# export MAP_MODEL="./models/test_ppo/epoch=13.ckpt"

# CUDA_VISIBLE_DEVICES=0 \
# /fs/nexus-scratch/lyzheng/miniconda3/envs/tfuse/bin/python -m carla_project.src.traffic_img_model \
# --id=${EXP_ID} \
# --teacher_path=${MAP_MODEL} \
# --dataset_dir=data/tiny \