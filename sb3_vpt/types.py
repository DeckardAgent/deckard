from typing import NamedTuple, Tuple
import torch as th


class VPTStates(NamedTuple):
    mask: Tuple[th.Tensor, ...]
    keys: Tuple[th.Tensor, ...]
    values: Tuple[th.Tensor, ...]


class VPTRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    vpt_states: VPTStates
    task_id: th.Tensor
    episode_starts: th.Tensor
    mask: th.Tensor
