import numpy as np
import torch
from gym import spaces
from typing import NamedTuple
from stable_baselines3.common.buffers import BaseBuffer

class Temp_RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    t_1_robot: torch.Tensor
    t_2_robot: torch.Tensor

class Temp_RolloutBufferSamples_variant1(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    target: torch.Tensor
    temp_info: torch.Tensor

class TempRolloutBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        robot_dim: int = 0,
    ):
        super(TempRolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.robot_dim = robot_dim

        # self.pos refer the current step position in buffer
        self.pos = 0

        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.t_1_robot, self.t_2_robot = None, None
        self.generator_ready = False
        self.reset()

    def reset(self):
        self.pos = 0
        self.full = False

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.t_1_robot = np.zeros((self.buffer_size, self.n_envs) + (self.robot_dim,), dtype=np.float32)
        self.t_2_robot = np.zeros((self.buffer_size, self.n_envs) + (self.robot_dim,), dtype=np.float32)

        self.generator_ready = False

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            # delta = r + gamma * V(s+1) - V(s)
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            # gae(s) = delta + gamma * lambda * gae(s+1)
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # A(s) = R - V(S)
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        t_1_robot: torch.Tensor,
        t_2_robot: torch.Tensor,
    ):
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.t_1_robot[self.pos] = t_1_robot.clone().cpu().numpy()
        self.t_2_robot[self.pos] = t_2_robot.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True  

    def get(self, batch_size):
        # generate random permutation for those data
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        # this is a name holder, for change the corresponding value in self.__dict__ 
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "t_1_robot",
                "t_2_robot"
            ]

            # this is for flat data in buffer
            # e.g., original buffer has buffer_size * n_envs actions,
            # which is buffer_size * n_envs * action_dim 3d tensor
            # here we need to change it into (buffer_size * n_envs) * action_dim 2d tensor
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches, batch_size = 64
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # this is for epoch, since for every epoch we need a minibatch from rollout_buffer
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.t_1_robot[batch_inds],
            self.t_2_robot[batch_inds],
        )
        return Temp_RolloutBufferSamples(*tuple(map(self.to_torch, data)))

class TempRolloutBuffer_variant1(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        node_num,
        device,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(TempRolloutBuffer_variant1, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.node_num = node_num

        # self.pos refer the current step position in buffer
        self.pos = 0

        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0]
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.temp_info, self.target = None, None
        self.generator_ready = False
        self.reset()

    def reset(self):
        self.pos = 0
        self.full = False

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32) # buffer_pos, batch, dim
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.target = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.temp_info = np.zeros((self.buffer_size, self.n_envs, self.node_num) + self.obs_shape, dtype=np.float32) # buffer_pos, batch, dim

        self.generator_ready = False

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            # delta = r + gamma * V(s+1) - V(s)
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            # gae(s) = delta + gamma * lambda * gae(s+1)
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # A(s) = R - V(S)
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        target: torch.Tensor,
        temp_info: torch.Tensor,
    ):
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy() # buffer_pos, batch, dim
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.target[self.pos] = target.clone().cpu().numpy()
        self.temp_info[self.pos] = temp_info.clone().cpu().numpy() # buffer_pos, node, batch, dim

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True  

    def get(self, batch_size):
        # generate random permutation for those data
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        # this is a name holder, for change the corresponding value in self.__dict__ 
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "target",
                "temp_info"
            ]

            # this is for flat data in buffer
            # e.g., original buffer has buffer_size * n_envs actions,
            # which is buffer_size * n_envs * action_dim 3d tensor
            # here we need to change it into (buffer_size * n_envs) * action_dim 2d tensor
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches, batch_size = 64
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # this is for epoch, since for every epoch we need a minibatch from rollout_buffer
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds], # 2048 * 6 -> 12288*113
            self.actions[batch_inds], # 12288 * 8
            self.values[batch_inds].flatten(), 
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.target[batch_inds],
            self.temp_info[batch_inds], # 12288 * 6 * 113
        )
        return Temp_RolloutBufferSamples_variant1(*tuple(map(self.to_torch, data)))
