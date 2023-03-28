import numpy as np
from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from sb3_contrib.common.recurrent.buffers import create_sequencers

from sb3_vpt.types import VPTStates, VPTRolloutBufferSamples


class VPTBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the VPT hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect states
        (n_steps, num_blocks, n_envs, buffer_size, hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int, int],
        state_buffer_size: int = 128,
        state_buffer_idx: int = 3,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.state_buffer_size = state_buffer_size
        self.state_buffer_idx = state_buffer_idx
        self.task_id_shape = (buffer_size, n_envs)
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self):
        super().reset()
        self.hidden_states_masks = np.zeros(self.hidden_state_shape[:-1], dtype=np.float32)
        self.hidden_states_keys = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_values = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.task_id = np.zeros(self.task_id_shape, dtype=np.int8)

    def add(self, *args, vpt_states: VPTStates, task_id: th.Tensor, **kwargs) -> None:
        """
        :param hidden_states
        """
        slc = (slice(None),) * (self.state_buffer_idx - 1) + (-1,)
        self.hidden_states_masks[self.pos] = np.array(vpt_states[0][slc].cpu().numpy())
        self.hidden_states_keys[self.pos] = np.array(vpt_states[1][slc].cpu().numpy())
        self.hidden_states_values[self.pos] = np.array(vpt_states[2][slc].cpu().numpy())
        self.task_id[self.pos] = task_id

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[VPTRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, num_blocks, self.n_envs, hidden_size)
            # swap first to (self.n_steps, self.n_envs, num_blocks, hidden_size)
            for tensor in ["hidden_states_masks", "hidden_states_keys", "hidden_states_values"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_masks",
                "hidden_states_keys",
                "hidden_states_values",
                "task_id",
                "episode_starts"
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> VPTRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length

        # We retrieve the hidden states that will allow proper initialization the at the beginning of each sequence
        eps_start_indices = np.logical_or(self.episode_starts, env_change).flatten()
        eps_start_indices[0] = True

        masks = []
        keys = []
        values = []
        for seq_start in batch_inds[self.seq_start_indices]:
            eps_start = np.where(eps_start_indices[:seq_start])[0]
            eps_start = eps_start[-1] if len(eps_start) > 0 else 0  # If len==0, seq_start also equals 0
            eps_start = max(eps_start, seq_start + 1 - self.state_buffer_size)  # Only need 128 sized buffer
            padding_size = self.state_buffer_size - (seq_start + 1 - eps_start)  # May need some padding
            
            # 1, buffer, n_blocks, dim
            masks.append(np.expand_dims(np.concatenate((
                np.zeros((padding_size, self.hidden_states_masks.shape[-1]), dtype=np.float32), 
                self.hidden_states_masks[eps_start:seq_start+1]
            ), axis=0), axis=0))
            keys.append(np.expand_dims(np.concatenate((
                np.zeros((padding_size,) + self.hidden_states_keys.shape[-2:], dtype=np.float32), 
                self.hidden_states_keys[eps_start:seq_start+1]
            ), axis=0), axis=0))
            values.append(np.expand_dims(np.concatenate((
                np.zeros((padding_size,) + self.hidden_states_values.shape[-2:], dtype=np.float32), 
                self.hidden_states_values[eps_start:seq_start+1]
            ), axis=0), axis=0))
        
        # (n_seq, buffer, n_blocks, dim) -> (n_blocks, n_seq, buffer, dim)
        masks = np.concatenate(masks, axis=0).transpose((2, 0, 1))
        keys = np.concatenate(keys, axis=0).transpose((2, 0, 1, 3))
        values = np.concatenate(values, axis=0).transpose((2, 0, 1, 3))

        vpt_states = (self.to_torch(masks), self.to_torch(keys), self.to_torch(values))

        return VPTRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size,) + self.obs_shape),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            vpt_states=VPTStates(*vpt_states),
            task_id=self.pad_and_flatten(self.task_id[batch_inds]),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )
