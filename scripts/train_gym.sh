
bash scripts/path_config.sh

${CARLA_HOME}/CarlaUE4.sh -opengl -carla-port=2000 &

CARLA_PID=$!

sleep 5
python cosim_gym_env.py


kill -KILL $CARLA_PID