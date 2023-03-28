from typing import List, Dict, Tuple
import numpy as np
import json
import os
from copy import deepcopy
from filelock import FileLock

from tech import TECH_ORDER
from tech.technode import TechNode


class TechTree:
    def __init__(self, all_items, log_dir=None, tasks=dict(), guide_path=None, target_item=None, init_graph=None, **kwargs):
        self.techtree_lock = FileLock(os.path.join(log_dir, "techtree.json.lock"), timeout=10)
        self.techtree_path = os.path.join(log_dir, "techtree.json")
        self.target_item = target_item
        self.steps_since_sync = 0
        self.total_steps = 0
        self.iterations = 0
        self.seen_invs = []
        self.node_visits = {x: 0 for x in all_items + ["any"]}
        self.node_success = {x: 1 for x in all_items + ["any"]}

        self.guide_trees = []
        self.trees = []
        self.tasks = list(tasks.keys())

        # Initialize trees if given
        if init_graph is not None:
            with self.techtree_lock:
                self._sync_info()
                self.trees = [TechNode.from_json(x) for x in init_graph]
                self._save_info()

        # Initialize with exploration task
        self._add_node(TechNode("any", collectable=True))

        # If using a guide, retrieve predicted graph
        if guide_path is not None:
            with open(guide_path, "r") as f:
                self.guide_trees = [TechNode.from_json(x) for x in json.load(f)]

            # Remove unknown items from guidance
            unknown_items = [x.name for x in self.get_all_nodes(guide=True) if x.name not in all_items]
            self.guide_trees = [x for x in self.guide_trees if x.name not in unknown_items]
            for name in unknown_items:
                for root in self.guide_trees:
                    root.purge(name)

    def tick(self, n: int = 1) -> None:
        self.steps_since_sync += n

    def get_task_id(self, task: TechNode) -> int:
        if task is not None and task.name in self.tasks and self.get_node_by_name(task.name) is not None:
            return self.tasks.index(task.name)
        return -1

    def has_target(self) -> bool:
        # Check if there is an unexplored boundary
        # Visit count can be 1 if agent is currently exploring it
        return self.target_item is not None and any(self.node_visits[n.name] <= 1 for n in self._get_boundary())

    def get_all_nodes(self, guide: bool = False) -> List[TechNode]:
        trees = self.guide_trees if guide else self.trees
        all_nodes = set()
        for root in trees:
            all_nodes.update([root] + root.get_subnodes())
        return list(all_nodes)

    def get_node_by_name(self, name: str, guide: bool = False) -> TechNode:
        matches = [x for x in self.get_all_nodes(guide=guide) if x.name == name]
        if len(matches) == 0:
            return None
        return matches[0]

    def update_collectables(self, prev_inv: Dict[str, int], curr_inv: Dict[str, int]) -> None:
        collected = set()
        for n, q in curr_inv.items():
            if q > prev_inv[n]:
                collected.add(n)
        for name in collected:
            if self.get_node_by_name(name) is None:
                tool = None
                for n, q in prev_inv.items():
                    # For now we only care if a pickaxe is required
                    if q > 0 and "pickaxe" in n and (tool is None or self._get_tech_rank(n) > self._get_tech_rank(tool.name)):
                        tool = self.get_node_by_name(n)
                new_node = TechNode(
                    name, 
                    collectable=True, 
                    tool=tool,
                    timestep=self.total_steps+self.steps_since_sync,
                    iteration=self.iterations
                )
                self._add_node(new_node)

    def check_seen_inv(self, inv: Dict[str, int]) -> bool:
        return sum(q for q in inv.values()) == 0 or len(self.seen_invs) > 0 and \
            any(all(
                q <= seen[n] if n in seen else False 
                for n, q in inv.items() if q > 0 and n != "any"
            ) for seen in self.seen_invs)

    def add_craft(self, name, prev_inv: Dict[str, int], curr_inv: Dict[str, int]) -> None:
        recipe = []
        for n, q in curr_inv.items():
            if n != "any" and q < prev_inv[n]:
                node = self.get_node_by_name(n)
                assert node is not None, "Crafted {} with item ({}) not in dag".format(name, n)
                recipe.append((node, prev_inv[n] - q))
        new_node = TechNode(
            name, 
            recipe=recipe, 
            table=prev_inv["crafting_table"] > 0,
            furnace=prev_inv["furnace"] > 0, 
            timestep=self.total_steps+self.steps_since_sync,
            iteration=self.iterations
        )
        self._add_node(new_node)

    def add_seen_invs(self, inv: Dict[str, int]) -> None:
        with self.techtree_lock:
            self._sync_info()
            filterd_inv = {n: q for n, q in inv.items() if q > 0 and n != "any"}
            found = False
            to_remove = []
            for i, seen in enumerate(self.seen_invs):
                if set(seen.keys()).issubset(set(filterd_inv.keys())):
                    if not found:
                        for item in filterd_inv:
                            seen[item] = max(seen[item] if item in seen else 0, filterd_inv[item])
                        found = True
                    else:
                        to_remove.append(i)
            for i in reversed(to_remove):
                self.seen_invs.pop(i)
            if not found:
                self.seen_invs.append(filterd_inv)
            self._save_info()

    def sample_goal(self, last_goal=None, completed=False, success=False) -> TechNode:

        nodes = self._get_candidate_nodes()
        with self.techtree_lock:
            self._sync_info()
            if last_goal is not None:
                if completed:
                    self.node_success[last_goal.name] = .9 * self.node_success[last_goal.name] + .1 * success
                else:
                    self.node_visits[last_goal.name] = max(0, self.node_visits[last_goal.name] - 1)
                
            nodes = [n for n in nodes if self.node_success[n.name] > -float("inf")]
            if any(self.node_visits[s.name] <= 0 for s in nodes):
                nodes = [n for n in nodes if self.node_visits[n.name] <= 0]

            goal = np.random.choice(nodes)

            self.iterations += 1
            self.node_visits[goal.name] += 1

            self._save_info()
        
        # print("Sampled goal:", goal.name)
        return goal

    def update_task(self, goal: TechNode, task: TechNode, tool: TechNode, inv: Dict[str, int]) -> Tuple[TechNode, TechNode, Dict[TechNode, int]]:
        assert goal is not None

        ingredients = tool.get_ingredients(inv) if tool is not None else goal.get_ingredients(inv)
        if task is not None and task in ingredients and ingredients[task] > 0:
            return task, tool

        # Get nodes for benches
        crafting_table = self.get_node_by_name("crafting_table")
        if crafting_table is None and len(self.guide_trees) > 0:
            crafting_table = self.get_node_by_name("crafting_table", guide=True)
        furnace = self.get_node_by_name("furnace")
        if furnace is None and len(self.guide_trees) > 0:
            furnace = self.get_node_by_name("furnace", guide=True)

        # Sample new subgoal
        tool = None
        new_task = goal
        while not new_task.collectable or (new_task.tool is not None and inv[new_task.tool.name] == 0):
            ingredients = new_task.get_ingredients(inv)

            subgoals = set([n for n in ingredients.keys() if n.tool is None])
            for n in ingredients:
                if n.tool is None or inv[n.tool.name] > 0:
                    subgoals.add(n)
                else:
                    subgoals.add(n.tool)
            if inv["crafting_table"] == 0 and any(n.table for n in new_task.get_craft_order()):
                subgoals.add(crafting_table)
            if inv["furnace"] == 0 and any(n.furnace for n in new_task.get_craft_order()):
                subgoals.add(furnace)

            if len(subgoals) > 0:
                new_task = np.random.choice(list(subgoals))
            else:
                # If using a guide, we may have the recipe wrong. Sample from the known tree to explore.
                new_task = np.random.choice([n for n in self.get_all_nodes() if self.node_success[n.name] > -float("inf")])
                # print("Failed to find task. Doing {} instead".format(new_task.name))

            if not new_task.collectable:
                tool = new_task

        # print("Current task/tool:", new_task.name if new_task is not None else "None", 
        #     tool.name if tool is not None else "no tool")
        return new_task, tool

    def _get_boundary(self) -> List[TechNode]:
        nodes = set()
        if len(self.guide_trees) > 0:
            if self.target_item is not None:
                table_node = self.get_node_by_name("crafting_table", guide=True)
                furnace_node = self.get_node_by_name("furnace", guide=True)
                guide_nodes = []
                to_add = [self.get_node_by_name(self.target_item, guide=True)]
                while len(to_add) > 0:
                    guide_nodes += to_add
                    last_nodes = to_add.copy()
                    to_add = []
                    for node in last_nodes:
                        if node.tool is not None and node.tool not in guide_nodes:
                            to_add.append(node.tool)
                        if node.table and table_node not in guide_nodes:
                            to_add.append(table_node)
                        if node.furnace and furnace_node not in guide_nodes:
                            to_add.append(furnace_node)
                        for n in node.get_subnodes():
                            if n not in guide_nodes:
                                to_add.append(n)
            else:
                guide_nodes = self.get_all_nodes(guide=True)

            for node in guide_nodes:
                if self.get_node_by_name(node.name) is None and \
                        (node.tool is None or self.get_node_by_name(node.tool.name) is not None) and \
                        (not node.table or self.get_node_by_name("crafting_table") is not None) and \
                        (not node.furnace or self.get_node_by_name("furnace") is not None) and \
                        (node.collectable or all(x.name in self.tasks for x in node.get_ingredients())) and \
                        all(self.get_node_by_name(x.name) is not None for x, _ in node.recipe) and \
                        all(x.get_requirements() != node.get_requirements() for x in nodes):
                    nodes.add(node)

        return list(nodes)

    def _get_candidate_nodes(self) -> List[TechNode]:
        if self.target_item is not None:
            target_node = self.get_node_by_name(self.target_item)
            if target_node is not None:
                return [target_node]
        nodes = self._get_boundary()
        nodes += self.get_all_nodes()
        return nodes

    def _get_tech_rank(self, name: str) -> int:
        return ([0] + [i + 1 for i, x in enumerate(TECH_ORDER) if name is not None and x in name])[-1]

    def _check_tool(self, tool: TechNode, inv: Dict[str, int]) -> bool:
        if "_" in tool.name and tool.name.split("_")[0] in TECH_ORDER:
            tech_name = tool.name.split("_")[0]
            base_name = tool.name.split("_")[1]
            sufficient_tools = [x + "_" + base_name for x in TECH_ORDER \
                if TECH_ORDER.index(x) >= TECH_ORDER.index(tech_name)]
        else:
            sufficient_tools = [tool.name]
        return any(inv[x] > 0 for x in sufficient_tools)

    def _add_node(self, node: TechNode) -> None:
        with self.techtree_lock:
            self._sync_info()

            if any(node == x or node in x.get_subnodes() for x in self.trees):
                return
            for child, _ in node.recipe:
                if child in self.trees:
                    self.trees.remove(child)
            self.trees.append(node)
            
            # Update guide dag with known dag
            for root in self.guide_trees:
                root.update_children_info([node])

            # Update node success with pretrained tasks
            if node.name != "any" and any(x.name not in self.tasks for x in node.get_ingredients().keys()):
                self.node_success[node.name] = -float("inf")

            self._save_info()

    def _sync_info(self) -> None:
        if not os.path.exists(self.techtree_path):
            with open(self.techtree_path, "w") as f:
                json.dump(dict(
                    trees=[], seen_invs=[], total_steps=0, iterations=0, node_success=dict(), node_visits=dict()
                ), f)

        with open(self.techtree_path, "r") as f:
            stored = json.load(f)
            self.trees = [TechNode.from_json(x) for x in stored["trees"]]
            self.seen_invs = stored["seen_invs"]
            self.total_steps = stored["total_steps"]
            self.iterations = stored["iterations"]
            self.node_success = {
                k: stored["node_success"][k] if k in stored["node_success"] else 1
                for k in self.node_success.keys()
            }
            self.node_visits = {
                k: stored["node_visits"][k] if k in stored["node_visits"] else 0
                for k in self.node_visits.keys()
            }

        # Update guide dag with known dag
        all_nodes = self.get_all_nodes()
        for root in self.guide_trees:
            root.update_children_info(all_nodes)

    def _save_info(self) -> None:
        self.total_steps += self.steps_since_sync
        self.steps_since_sync = 0
        with open(self.techtree_path, "w") as f:
            json.dump(dict(
                trees=[x.to_json() for x in self.trees], 
                seen_invs=self.seen_invs,
                total_steps=self.total_steps,
                iterations=self.iterations,
                node_success={k: v for k, v in self.node_success.items() if v != 1},
                node_visits={k: v for k, v in self.node_visits.items() if v > 0},
            ), f, indent=4)
