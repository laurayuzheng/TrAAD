#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:external/leaderboard
export PYTHONPATH=$PYTHONPATH:external/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:external/scenario_runner

if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
fi

python3 cosim/cosim_leaderboard_evaluator.py \
--track=SENSORS \
--scenarios=./external/leaderboard/data/all_towns_traffic_scenarios_public.json  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--carla-port=${PORT}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."
