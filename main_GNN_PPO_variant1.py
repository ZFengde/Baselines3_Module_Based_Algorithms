import gym
import os
import time
import turtlebot_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from feng_algorithms.GNN_PPO_variant1.GNN_PPO_variant1 import GNN_PPO_variant1

def learn():
	env_id = 'Turtlebot-v2'
	models_dir = f"GNN_PPO_variant1/{env_id}/models/{int(time.time())}/"
	logdir = f"GNN_PPO_variant1/{env_id}/logs/{int(time.time())}/"
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	num_cpu = 6
	# env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={'exclude_current_positions_from_observation': False})
	env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv,)
	model = GNN_PPO_variant1('GnnPolicy_variant1', env, verbose=1, tensorboard_log=logdir, use_sde=False)
	TIMESTEPS = 10000
	iters = 0
	for i in range(1000):
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"GNN_PPO_variant1")
		model.save(models_dir, TIMESTEPS*iters)

def test():
	env_id = 'Ant-v3'
	env = gym.make(env_id, exclude_current_positions_from_observation=False)
	model = GNN_PPO_variant1('GnnPolicy_variant1', env, verbose=1,  use_sde=False)
	model.load(f'./GNN_PPO_variant1/{env_id}/models/1663674299/3100000')
	model.test(100)


if __name__ == '__main__':
	learn()



