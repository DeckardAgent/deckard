from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod


class SuccessWrapper(Wrapper, ABC):
    def __init__(self, env, terminal: bool = True, reward: int = 0, all: Dict = dict(), any: Dict = dict()):
        super().__init__(env)
        self.terminal = terminal
        self.all_conditions = all
        self.any_conditions = any
        self.success_reward = reward

    def step(self, action):
        obs, reward, done, info = super().step(action)

        info["success"] = info.pop("success", False)

        if len(self.all_conditions) > 0:
            info["success"] = info["success"] or all(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.all_conditions.items()
            )

        if len(self.any_conditions) > 0:
            info["success"] = info["success"] or any(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.any_conditions.items()
            )

        if self.terminal:
            done = done or info["success"]
        if info["success"]:
            reward += self.success_reward

        return obs, reward, done, info

    def _check_condition(self, condition_type, condition_info, obs):
        if condition_type == "item":
            return self._check_item_condition(condition_info, obs)
        elif condition_type == "blocks":
            return self._check_blocks_condition(condition_info, obs)
        else:
            raise NotImplementedError("{} terminal condition not implemented".format(condition_type))

    @abstractstaticmethod
    def _check_item_condition(condition_info, obs):
        raise NotImplementedError()

    @abstractstaticmethod
    def _check_blocks_condition(condition_info, obs):
        raise NotImplementedError()
