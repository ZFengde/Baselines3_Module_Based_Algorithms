import gym
import turtlebot_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from module_based_algorithms.Trial_GNN_PPO.Trial_GNN_PPO import Trial_GNN_PPO

def learn():
    env_id = 'Turtlebot-v1'
    num_cpu = 6
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    model = Trial_GNN_PPO(env, experiment_name='Trial_GNN_PPO', parallel=True)
    model.learn(5000000)

def test():
    env = gym.make('Turtlebot-v1', use_gui=True)
    model = Trial_GNN_PPO(env, experiment_name='Trial_GNN_PPO', parallel=False)
    model.load('./Trial_GNN_PPO_Record/models/2022-07-14-00-10-42/5001216')
    model.test(100)

if __name__ == '__main__':
    learn()



