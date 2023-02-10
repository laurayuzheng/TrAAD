
bash scripts/path_config.sh 

export RESULTS_DIR=/scratch/2020_CARLA_challenge/ablation_driving_models/
export PORT=2000                                                    # change to port that CARLA is running on
# export ROUTES=./leaderboard/data/routes_testing.xml         # change to desired route
export ROUTES=./leaderboard/data/routes_testing/route_01.xml         # change to desired route
# export TEAM_AGENT=traffic_image_agent.py                                    # no need to change
# export TEAM_CONFIG=models/ppo_2/epoch=16.ckpt                        # Method 2 model (traffic_image_agent)
export COSIM=1
export METRICS=1
export TEAM_AGENT=image_agent.py
# export TEAM_CONFIG=models/lbc_baseline/image_model.ckpt              # baseline model (image_agent)
# export TEAM_CONFIG=${RESULTS_DIR}/lbc_baseline/image_model.ckpt # baseline model 
export TEAM_CONFIG=${RESULTS_DIR}/diffppo_lbc_img/epoch=24.ckpt # ours

export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window
export DEBUG_CHALLENGE=0

./run_agent.sh
