from typing import List, Dict
import numpy as np
from gym import Wrapper
from abc import ABC, abstractmethod
from minedojo.sim.mc_meta.mc import *

from tech.technode import TechNode
from tech.techtree import TechTree


class TechTreeWrapper(Wrapper, ABC):
    def __init__(self, env, craft_ticks=3, max_task_steps=1000, max_goal_steps=5000, **kwargs):
        super().__init__(env)
        self.craft_ticks = craft_ticks
        self.max_task_steps = max_task_steps
        self.max_goal_steps = max_goal_steps
        self.techtree = TechTree(self._get_all_items(), **kwargs)

        self.craftables = self._get_craftables()
        self.curr_inventory = {x: 0 for x in self._get_all_items() + ["any"]}
        self.last_inventory = self.curr_inventory
        self.curr_goal = None
        self.curr_task = None
        self.curr_tool = None
        self.task_steps = 0
        self.goal_steps = 0
        self._last_obs = None
        self.is_techtree_active = True
        self._sample_goal()

    def reset(self, **kwargs):
        self.curr_inventory = {x: 0 for x in self._get_all_items() + ["any"]}
        self.last_inventory = self.curr_inventory
        self._last_obs = None
        self._sample_goal()
        return self.env.reset(**kwargs)

    def step(self, action):
        if not self.is_techtree_active:
            return self.env.step(action)

        action["drop"] = 0  # manually disable dropping and placing items
        if self.curr_task is not None and self.curr_task.tool is not None:
            action = self._equip_item(self._last_obs, action, self.curr_task.tool.name)

        obs, reward, done, info = self.env.step(action)
        self.techtree.tick()

        # Gather info
        self.goal_steps += 1
        self.task_steps += 1
        self.curr_inventory = self._get_techtree_inventory(obs)

        ######### Tech Tree Logic #########

        # Reward gathering items for current task
        reward += self._get_reward()

        # Add new items we've collected to the graph
        self.techtree.update_collectables(self.last_inventory, self.curr_inventory)

        if self.curr_task is not None and self.curr_inventory[self.curr_task.name] > self.last_inventory[self.curr_task.name]:
            # print("\tCollected {} ({})".format(self.curr_task.name, self.curr_inventory[self.curr_task.name]))
            info["untrained"] = self.curr_task.name if self.curr_task.name not in ["dirt", "any"] and \
                self.curr_task.name not in self.techtree.tasks and \
                self.techtree.get_node_by_name(self.curr_task.name) is not None else ""

        # If we're working on a tool and have the ingredients, craft it
        if self.curr_tool is not None:
            obs, reward, done, info = self._craft(self.curr_tool, obs, reward, done, info)
            if self.curr_inventory[self.curr_tool.name] > 0:
                self.curr_tool = None

        # If we have ingredients for the current goal, craft it
        if not self.curr_goal.collectable:
            obs, reward, done, info = self._craft(self.curr_goal, obs, reward, done, info)

        # If we successfully obtained the current goal, try crafting something new, then sample a new goal
        if self._check_goal_success():
            if not self.techtree.has_target():
                obs, reward, done, info = self._try_craft_new_item(obs, reward, done, info)
            self._sample_goal(completed=True, success=True)

        # If we're still looking for a collectable and we're not crafting a tool, try crafting new items to expand the graph
        elif self.curr_goal.collectable and self.curr_tool is None:
            if not self.techtree.has_target():
                obs, reward, done, info = self._try_craft_new_item(obs, reward, done, info)
            # If we haven't found the collectable yet, sample a new goal
            if self.task_steps >= self.max_task_steps:
                self._sample_goal(completed=True)

        else:
            # If we've been unable to craft the current goal, sample a new goal
            if self.goal_steps >= self.max_goal_steps:
                obs, reward, done, info = self._try_craft_new_item(obs, reward, done, info)
                self._sample_goal(completed=True)
            # If we haven't found current task yet, try looking for something else for the current goal
            if self.task_steps >= self.max_task_steps:
                self.curr_task = None

        # Update what we're currently looking for, if it requires a tool, that takes priority
        self._update_task()

        ######### End Tech Tree Logic #########

        # Info to pass to algorithm
        info["subgoal"] = self.techtree.get_task_id(self.curr_task)
        info["timestep"] = self.techtree.total_steps
        info["early_stop"] = self.techtree.get_node_by_name(self.techtree.target_item) is not None \
            if self.techtree.target_item is not None else False

        self._last_obs = obs.copy()
        self.last_inventory = self.curr_inventory
        return obs, reward, done, info
    
    def add_task(self, new_task):
        self.techtree.tasks.append(new_task)

    def _sample_goal(self, completed=False, success=False):
        self.curr_goal = self.techtree.sample_goal(last_goal=self.curr_goal, completed=completed, success=success)
        self.curr_task = None
        self.goal_steps = 0
        self.task_steps = 0

    def _update_task(self):
        new_task, self.curr_tool = self.techtree.update_task(
            self.curr_goal, self.curr_task, self.curr_tool, self.curr_inventory
        )
        if new_task != self.curr_task:
            self.task_steps = 0
        self.curr_task = new_task

    def _craft(self, node: TechNode, obs, reward, done, info):
        crafts = node.get_craft_order(self.curr_inventory)

        crafting = True
        while crafting:

            crafting = False
            for to_craft in crafts:

                if not done and self._check_ingredients(to_craft):
                    action = self._get_craft_action(
                        to_craft.name,
                        self.curr_inventory["crafting_table"] > 0,
                        self.curr_inventory["furnace"] > 0
                    )
                    if action is not None:
                        obs, reward, done, info = self.env.step(action)
                        for _ in range(self.craft_ticks):
                            if not done:
                                obs, reward, done, info = self.env.step(self._get_noop_action())
                    self.techtree.tick(1 + self.craft_ticks)

                    inventory = self._get_techtree_inventory(obs)
                    if inventory[to_craft.name] > self.curr_inventory[to_craft.name]:
                        # print("\tCrafted {} ({})".format(to_craft.name, inventory[to_craft.name]))
                        if to_craft not in self.techtree.get_all_nodes():
                            self.techtree.add_craft(to_craft.name, self.curr_inventory, inventory)
                        self.curr_inventory = inventory
                        crafts = node.get_craft_order(self.curr_inventory)
                        crafting = to_craft != node
                        break

                    else:
                        self.curr_inventory = inventory

        return obs, reward, done, info

    def _try_craft_new_item(self, obs, reward, done, info):
        if self.techtree.check_seen_inv(self.curr_inventory):
            return obs, reward, done, info

        success = False
        crafted = [x.name for x in self.techtree.get_all_nodes() if not x.collectable]
        names = [x for x in self.craftables if x in self.curr_inventory and x not in crafted]
        np.random.shuffle(names)
        for name in names:
            if not done:
                action = self._get_craft_action(
                    name,
                    self.curr_inventory["crafting_table"] > 0,
                    self.curr_inventory["furnace"] > 0,
                    [n.name for n in self.techtree.get_all_nodes() if not n.collectable or n.name in self.techtree.tasks]
                )
                if action is not None:
                    obs, reward, done, info = self.env.step(action)
                    for _ in range(self.craft_ticks):
                        if not done:
                            obs, reward, done, info = self.env.step(self._get_noop_action())
                self.techtree.tick(1 + self.craft_ticks)
            inventory = self._get_techtree_inventory(obs)
            if inventory[name] > self.curr_inventory[name]:
                # print("\tCrafted New {} ({})".format(name, inventory[name]))
                self.techtree.add_craft(name, self.curr_inventory, inventory)
                success = True
                break
            if done:
                break

        if not success:
            self.techtree.add_seen_invs(self.curr_inventory)
        
        self.curr_inventory = self._get_techtree_inventory(obs)
        return obs, reward, done, info

    def _check_ingredients(self, node: TechNode) -> bool:
        return sum(x for x in node.get_ingredients(self.curr_inventory).values()) == 0 and \
            (not node.table or self.curr_inventory["crafting_table"] > 0) and \
            (not node.furnace or self.curr_inventory["furnace"] > 0)

    def _get_reward(self) -> float:
        if self.curr_task is not None and self.curr_task.name != "any":
            return int(self.curr_inventory[self.curr_task.name] > self.last_inventory[self.curr_task.name])
        else:
            return 0

    def _check_goal_success(self, goal: TechNode = None) -> bool:
        if goal is None:
            goal = self.curr_goal
        return self.curr_inventory[goal.name] > self.last_inventory[goal.name]

    def _get_techtree_inventory(self, obs: Dict) -> Dict[str, int]:
        inventory = self._get_inventory(obs)
        prev_count = sum(self.last_inventory.values()) if self.last_inventory is not None else 0
        inventory["any"] = max(0, sum(inventory.values()) - prev_count)
        return inventory

    @abstractmethod
    def _get_inventory(obs: Dict) -> Dict[str, int]:
        # retrieves a dict of item names to quantities for every game item
        raise NotImplementedError()

    @abstractmethod
    def _get_all_items(self) -> List[str]:
        # retrieves a list of all item names
        raise NotImplementedError()

    @abstractmethod
    def _get_craftables(self) -> List[str]:
        # retrieves a list of craftable item names
        raise NotImplementedError()

    @abstractmethod
    def _get_noop_action(self) -> Dict:
        # the no op action
        raise NotImplementedError()

    @abstractmethod
    def _get_craft_action(self, item, crafting_table=False, furnace=False, valid=None) -> Dict:
        # retrieves the craft action for the given item with the given tools
        raise NotImplementedError()

    @abstractmethod
    def _equip_item(self, last_obs: Dict, action: Dict, item: str) -> Dict:
        # retrieves a dict of item names to game actions
        raise NotImplementedError()
