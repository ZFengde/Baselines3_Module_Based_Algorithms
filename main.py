import gym
import safety_gym
import os
import time
import turtlebot_env
import feng_algorithms
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def main(env_id, algo, policy_type, n_envs, iter_num, seed):

	algo_name = algo

	algo = eval('feng_algorithms.'+algo)
	env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)

	# make experiment directory
	logdir = f"{algo_name}/{env_id}/logs/{int(time.time())}/"
	modeldir = f"{algo_name}/{env_id}/models/{int(time.time())}/"

	if not os.path.exists(modeldir):
		os.makedirs(modeldir)
	if not os.path.exists(logdir):
		os.makedirs(logdir)
		
	# TODO, policy_type need to be rectified here, but not a big deal
	model = algo(policy_type, env, verbose=1, tensorboard_log=logdir, use_sde=False)
	TIMESTEPS = 10000
	iters = 0
	for i in range(iter_num):
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algo_name}")
		model.save(modeldir+f'{int(time.time())}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--algo', type=str, default='PPO')
    parser.add_argument('--policy_type', type=str, default='MlpPolicy')
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



