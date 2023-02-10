#!/bin/bash

bash scripts/path_config.sh

export EXPERIMENT_ID=test_ppo
export CARLA_PORT=2000
export TRAFFIC_PORT=9000
export SUMO_PORT=8813

pkill Carla 
DISPLAY= ${CARLA_HOME}/CarlaUE4.sh -opengl -carla-port=${CARLA_PORT} -fps=20 -nosound >/dev/null 2>&1 & 
last_pid=$!

sleep 5

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64

export RESUME_MODEL="./logs/rl_model_132000_steps.zip"

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=8 python cosim/cosim_gym_env.py \
--routes="./external/leaderboard/data/routes_training.xml" \
--agent="./external/leaderboard/team_code/accel_agent.py" \
--carla-port=${CARLA_PORT} \
--sumo-port=${SUMO_PORT} \
--trafficManagerPort=${TRAFFIC_PORT} \
--id ${EXPERIMENT_ID} \
--algorithm=ppo \
--checkpoint=./checkpoints/${EXPERIMENT_ID}_simulation.json \
--reward_type=ours \
# --load-accel=${RESUME_MODEL} \

kill last_pid