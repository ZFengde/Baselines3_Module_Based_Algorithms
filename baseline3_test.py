import gym
import numpy as np
import turtlebot_env
from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


def learn():
	env_id = 'Turtlebot-v2'
	models_dir = f"Baselines3_PPO/{env_id}/models/{int(time.time())}/"
	logdir = f"Baselines3_PPO/{env_id}/logs/{int(time.time())}/"
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	num_cpu = 1
	env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
	model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, use_sde=False)
	model.load(f'./Baselines3_PPO/{env_id}/models/1662931505/2050000.zip')
	TIMESTEPS = 10000
	iters = 0
	for i in range(407):
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"Baselines3_PPO")
		model.save(f"{models_dir}/{TIMESTEPS*iters}")

def test():
	env_id = 'Turtlebot-v2'
	env = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv)
	model = PPO('MlpPolicy', env, verbose=1, use_sde=False)
	model.load(f'./Baselines3_PPO/{env_id}/models/1662931505/2050000.zip')
	obs = env.reset()
	while True:
		action, _states = model.predict(obs, deterministic=True)  # User-defined policy function
		obs, _, done, _ = env.step(action)
		if done:
			observation = env.reset()

if __name__ == '__main__':
	learn()


