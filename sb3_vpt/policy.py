from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym3.types import DictType
import gym
import torch as th
from torch import nn
from itertools import chain

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from sb3_vpt.types import VPTStates
from VPT.lib.policy import MinecraftAgentPolicy
from VPT.lib.action_mapping import CameraHierarchicalMapping
from VPT.lib.tree_util import tree_map


class VPTPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs\
    ):
        policy_kwargs = kwargs.pop("policy_kwargs", dict())
        pi_head_kwargs = kwargs.pop("pi_head_kwargs", dict())
        weights_path = kwargs.pop("weights_path", None)
        vpt_action_space = DictType(**CameraHierarchicalMapping(n_camera_bins=11).get_action_space_update())

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        self.model = MinecraftAgentPolicy(
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs, 
            action_space=vpt_action_space
        )
        self.exploration_model = MinecraftAgentPolicy(
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs, 
            action_space=vpt_action_space
        )
        if weights_path:
            self.model.load_state_dict(th.load(weights_path), strict=False)
            self.exploration_model.load_state_dict(th.load(weights_path), strict=False)

        self.exploration_model.requires_grad_(False)

        self.model.requires_grad_(False)
        self.params = {}

        self.model.value_head.reset_parameters()
        self.model.value_head.requires_grad_(True)
        self.params["model.value_head"] = self.model.value_head.parameters()

        for n, x in self.model.named_modules():
            if "img_process" not in n and n.split(".")[-1] == "adapter":
                x.requires_grad_(True)
                self.params["model." + n] = x.parameters()

        self.optimizer = self.optimizer_class(
            chain(*self.params.values()), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )

    def get_param_keys(self) -> List[str]:
        return list(self.params.keys())

    @staticmethod
    def _vpt_states_to_sb3(states):
        st = ([], [], [])
        for block_st in states:
            if block_st[0] is None:
                st[0].append(th.full_like(block_st[1][0], -1)[:, :, 0])
            else:
                assert block_st[0].shape[1] == 1
                st[0].append(block_st[0][:, 0])
            st[1].append(block_st[1][0])
            st[2].append(block_st[1][1])
        st = tuple([
            th.cat([blk.unsqueeze(0) for blk in state], dim=0)
            for state in st
        ])
        return VPTStates(*st)

    @staticmethod
    def _sb3_states_to_vpt(states):
        return tuple([
            (
                None if th.all(states[0][i] == -1) else \
                    states[0][i].unsqueeze(1).bool() if len(states[0][i].shape) == 2 else \
                    states[0][i].bool(), 
                (states[1][i], states[2][i])
            )
            for i in range(states[0].shape[0])
        ])

    def initial_state(self, batch_size):
        return self._vpt_states_to_sb3(self.model.initial_state(batch_size))

    def forward(self, 
        obs: th.Tensor,             # batch x H x W x C
        in_states: VPTStates,       # n_blocks x 1 x buffer, n_blocks x 1 x buffer x hidden
        episode_starts: th.Tensor,  # batch
        task_id: th.Tensor,         # batch
        deterministic: bool = False
    ) -> Tuple[Dict[str, th.Tensor], th.Tensor, th.Tensor, VPTStates, Dict[str, th.Tensor]]:

        # pd: dict: batch x 1 x 1 x 121, batch x 1 x 1 x 8641
        # vpred: batch x 1 x 1
        (pd, vpred, _), state_out = self.model(
            tree_map(lambda x: x.unsqueeze(1), {"img": obs}),
            episode_starts.unsqueeze(1).bool(),
            self._sb3_states_to_vpt(in_states),
            task_id
        )

        ac = self.model.pi_head.sample(pd, deterministic=deterministic) # dict: batch x 1 x 1
        log_prob = self.model.pi_head.logprob(ac, pd)[:, 0]             # batch
        vpred = vpred[:, 0, 0]                                          # batch
        ac = th.cat([x[:, 0] for x in ac.values()], dim=1)              # batch x 2

        return ac, vpred, log_prob, self._vpt_states_to_sb3(state_out)

    def predict_values(
        self,
        obs: th.Tensor,
        in_states: VPTStates,
        episode_starts: th.Tensor,
        task_id: th.Tensor
    ) -> th.Tensor:
        (_, vpred, _), _ = self.model(
            tree_map(lambda x: x.unsqueeze(1), {"img": obs}),
            episode_starts.unsqueeze(-1).bool(),
            self._sb3_states_to_vpt(in_states),
            task_id
        )
        return vpred[:, 0, 0]  # batch x 1
        
    def evaluate_actions(
        self,
        obs: th.Tensor,             # n_seq * max_len x H x W x C
        actions: th.Tensor,         # n_seq * max_len x 2
        in_states: VPTStates,       # n_blocks x n_seq x buffer, n_blocks x n_seq x buffer x hidden
        episode_starts: th.Tensor,  # n_seq * max_len
        task_id: th.Tensor,         # n_seq * max_len
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        n_seq = in_states[0].shape[1]
        obs_sequence = obs.reshape((n_seq, -1) + obs.shape[-3:])    # n_seq x max_len x H x W x C
        max_len = obs_sequence.shape[1]
        starts_sequence = episode_starts.reshape((n_seq, max_len))  # n_seq x max_len
        seq_task = task_id.reshape((n_seq, max_len))[:, 0]          # n_seq
        model_input = {"img": obs_sequence}, starts_sequence.bool(), self._sb3_states_to_vpt(in_states), seq_task

        # pd: dict: n_seq x max_len x 1 x 121, n_seq x max_len x 1 x 8641
        # vpred: n_seq x max_len x 1
        (pd, vpred, _), _ = self.model(*model_input)
        with th.no_grad():
            (exploration_pd, _, _), _ = self.exploration_model(*model_input)

        actions_dict = {
            k: actions[:, i].reshape((n_seq, max_len, 1))
            for i, k in enumerate(self.model.pi_head.keys())
        }
        log_prob = self.model.pi_head.logprob(actions_dict, pd)     # n_seq x max_len
        kl = self.model.get_kl_of_action_dists(pd, exploration_pd)  # n_seq x max_len x 1

        return th.flatten(vpred), th.flatten(log_prob), th.flatten(kl)
