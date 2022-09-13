import gym
import os
import time
import turtlebot_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from feng_algorithms.RNN_PPO.RNN_PPO import RNN_PPO

def learn():
	env_id = 'Ant-v3'
	models_dir = f"RNN_PPO/{env_id}/models/{int(time.time())}/"
	logdir = f"RNN_PPO/{env_id}/logs/{int(time.time())}/"
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	num_cpu = 6
	# env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'use_gui': False})
	env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': False})
	model = RNN_PPO('RnnPolicy', env, verbose=1, tensorboard_log=logdir, use_sde=False)
	TIMESTEPS = 10000
	iters = 0
	for i in range(407):
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"RNN_PPO")
		model.save(models_dir, TIMESTEPS*iters)

def test():
	env_id = 'Turtlebot-v2'
	env = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv)
	model = RNN_PPO('RnnPolicy', env, verbose=1,  use_sde=False)
	model.load(f'./RNN_PPO/{env_id}/models/1662931524/1320000')
	model.test(100)

if __name__ == '__main__':
	learn()



