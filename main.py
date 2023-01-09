import gym
import os
import time
import turtlebot_env
import feng_algorithms
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def main(env_id, algo, policy_type, n_envs, iter_num, seed):

	algo = eval('feng_algorithms.'+algo)

	# make experiment directory
	models_dir = f"GNN_PPO/{env_id}/models/{int(time.time())}/"
	logdir = f"GNN_PPO/{env_id}/logs/{int(time.time())}/"

	if not os.path.exists(models_dir):
		os.makedirs(models_dir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)

	env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    
	# TODO, here need to be rectified
	model = algo(policy_type, env, verbose=1, tensorboard_log=logdir, use_sde=False)
	TIMESTEPS = 10000
	iters = 0
	for i in range(iter_num):
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"GNN_PPO")
		model.save(models_dir, TIMESTEPS*iters)

# def test():
# 	env_id = 'Turtlebot-v2'
# 	env = make_vec_env(env_id, n_envs=1, seed=0, vec_env_cls=SubprocVecEnv)
# 	model = GNN_PPO('GnnPolicy', env, verbose=1,  use_sde=False)
# 	model.load(f'./GNN_PPO/{env_id}/models/1662931524/1320000')
# 	model.test(100)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Turtlebot-v1')
    parser.add_argument('--algo', type=str, default='GNN_PPO')
    parser.add_argument('--policy_type', type=str, default='GnnPolicy')
    parser.add_argument('--n_envs', type=int, default=3)
    parser.add_argument('--iter_num', type=str, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args.env_id, 
		args.algo, 
		args.policy_type, 
		args.n_envs, 
		args.iter_num, 
		args.seed)



