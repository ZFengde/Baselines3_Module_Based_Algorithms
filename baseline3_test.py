import gym
import numpy as np
import turtlebot_env
from stable_baselines3 import PPO, TD3
import os
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def learn():
	env_id = 'Ant-v3'
	models_dir = f"Baselines3_PPO/{env_id}/models/{int(time.time())}/"
	logdir = f"Baselines3_PPO/{env_id}/logs/{int(time.time())}/"

	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	num_cpu = 6
	env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': False})
	# env = gym.make('Ant-v3')
	# n_actions = env.action_space.shape[-1]
	# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

	model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, use_sde=False)
	# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
	TIMESTEPS = 10000
	iters = 0
	# for i in range(407):
	for i in range(1000):
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


