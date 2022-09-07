import gym
import os
import time
import turtlebot_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from feng_algorithms.GNN_PPO.GNN_PPO import GNN_PPO

if __name__ == '__main__':
	models_dir = f"GNN_PPO/models/{int(time.time())}/"
	logdir = f"GNN_PPO/logs/{int(time.time())}/"

	if not os.path.exists(models_dir):
		os.makedirs(models_dir)

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	env_id = 'Turtlebot-v2'

	num_cpu = 6
	env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
	model = GNN_PPO('GnnPolicy', env, verbose=1, tensorboard_log=logdir, use_sde=False)
	TIMESTEPS = 10000
	iters = 0
	for i in range(407):
		iters += 1
		
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"GNN_PPO")
		model.save(f"{models_dir}/{TIMESTEPS*iters}")




