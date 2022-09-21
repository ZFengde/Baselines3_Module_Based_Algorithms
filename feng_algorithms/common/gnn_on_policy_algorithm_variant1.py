import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from feng_algorithms.common.buffers import TempRolloutBuffer_variant1
from feng_algorithms.common.policies import ActorCriticGnnPolicy

class OnPolicyAlgorithm(BaseAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticGnnPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else TempRolloutBuffer_variant1

        self.rollout_buffer = buffer_cls(
            buffer_size=self.n_steps,
            observation_space=self.observation_space,
            action_space=self.action_space,
            node_num=4,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self.policy.g.to(self.device)
        self.policy.gnn.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: TempRolloutBuffer_variant1,
        n_rollout_steps: int,
    ) -> bool:

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        ep_num_success = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # This is for generate forward target
                target = self.target_generator(self._last_obs)
                temp_info = self.to_tensor_pack(self.t_2_target, self.t_1_target, self.t_2_robot, self.t_1_robot) # node_id = 0, 1, 3, 4 | target_t-2, t-1, robot_t-2, t-2
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor, target, temp_info) 
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            self.temp_info_update(self._last_obs, target.cpu().numpy())

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    self.temp_buffer_reset(idx)
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0].squeeze()
                    terminal_target = np.pad(infos[idx]["terminal_observation"][:2] + (np.random.rand(), 0), (0, self.env.observation_space.shape[0]-2))
                    terminal_target = obs_as_tensor(terminal_target, device=self.device)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_target, temp_info[idx])
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs, target, temp_info)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), self.target_generator(new_obs), temp_info)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

    def to_tensor_pack(self, *args):
        output = []
        for ele in args:
            ele = obs_as_tensor(ele, device=self.device)
            output.append(ele)
        output = th.stack(output, dim=1)
        return output # batch, node_num, dim

    def temp_info_update(self, last_obs, last_target): # this target setting is only for general mujoco envs
        self.t_2_target = self.t_1_target
        self.t_1_target = last_target 

        self.t_2_robot = self.t_1_robot
        self.t_1_robot = last_obs

    def temp_buffer_reset(self, index):
        # TODO, which is wrong, but should be able to work, amend later
        # which won't cause huge difference
        self.t_1_target[index] = np.zeros_like(self.t_1_target[index])
        self.t_2_target[index] = np.zeros_like(self.t_2_target[index])
        self.t_1_robot[index] = np.zeros_like(self.t_1_robot[index])
        self.t_2_robot[index] = np.zeros_like(self.t_2_robot[index])

    def target_generator(self, obs):
        target = obs[:, :2] + (np.random.rand(), 0)
        target = np.pad(target, ((0, 0), (0, self.env.observation_space.shape[0]-2))) # zeros padding so that target and agent have same dimension
        # target = obs[:2] + (np.random.rand(), 0)
        # target = np.pad(target, (0, self.env.observation_space.shape[0]-2)) # zeros padding so that target and agent have same dimension
        return obs_as_tensor(target, device=self.device)

    def test(self, test_episode):
        self.rollout_buffer.reset()
        last_obs = self.env.reset().squeeze()
        for episode_num in range(test_episode):
            ep_reward = 0
            ep_len = 0
            while True:
                self.env.render()
                with th.no_grad():
                    target = self.target_generator(last_obs)
                    temp_info = self.to_tensor_pack(self.t_2_target, self.t_1_target, self.t_2_robot, self.t_1_robot) # node_id = 0, 1, 3, 4 | target_t-2, t-1, robot_t-2, t-2
                    obs_tensor = obs_as_tensor(last_obs, self.device)
                    action, _, _ = self.policy(obs_tensor, target, temp_info.squeeze()) 
                action = action.cpu().numpy()

                clipped_actions = action
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(action, self.action_space.low, self.action_space.high)

                new_ob, reward, done, info = self.env.step(clipped_actions)
                last_obs = new_ob.squeeze()

                # self.temp_info_update(new_ob, target)
                self.temp_buffer_reset(0)
                ep_reward += reward
                ep_len += 1
                if done:
                    print(ep_len, ep_reward)
                    break
        return True