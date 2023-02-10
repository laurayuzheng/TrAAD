#!/bin/bash

bash scripts/path_config.sh

export PORT=2000                                                    # change to port that CARLA is running on
export TEAM_AGENT=./auto_pilot.py                                    # no need to change
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window
export DEBUG_CHALLENGE=0

split="$1"

if [ "${split}" == "train" ]; then
   search_dir=./external/leaderboard/data/routes_training
   export TEAM_CONFIG=./data/train
fi

if [ "${split}" == "test" ]; then
   search_dir=./external/leaderboard/data/routes_testing
   export TEAM_CONFIG=./data/test
fi

# set -e

pkill Carla

# Start CARLA Server 
${CARLA_HOME}/CarlaUE4.sh -opengl -benchmark -fps=30 & 

# Store CARLA server PID and wait 10 seconds
CARLASIM_PID=$!
sleep 10

function cleanup {
  echo "Killing CARLA Sim"
  kill ${CARLASIM_PID}
}

# Cleanup server process if script exits
trap cleanup EXIT

# Make necessary directories to save data
mkdir -p ${TEAM_CONFIG}

# Iteratively collect data in each route file
for entry in "$search_dir"/*
do
  export ROUTES=$entry 
  ./scripts/collect_helper.sh
done

# Kill the CARLA server
kill ${CARLASIM_PID}