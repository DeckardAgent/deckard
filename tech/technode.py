from typing import List, Dict, Tuple
from copy import deepcopy
import json


class TechNode:
    def __init__(self, name: str, collectable: bool = False, recipe: List[Tuple["TechNode", int]] = [],
                 tool: "TechNode" = None, table: bool = False, furnace: bool = False, timestep: int = 0, iteration: int = 0):
        self.name = name
        self.collectable = collectable
        self.recipe = recipe
        self.tool = tool
        self.table = table
        self.furnace = furnace
        self.timestep = timestep
        self.iteration = iteration
        assert self not in self.get_subnodes(), "Found cycle at node " + name
        
    def get_depth(self) -> List["TechNode"]:
        depths = [0]
        depths += [x.get_depth() for x, _ in self.recipe]
        if self.table:
            depths.append(3)
        if self.furnace:
            depths.append(6)
        if self.tool is not None:
            depths.append(self.tool.get_depth())
        return 1 + max(depths)
        
    def get_children(self) -> List["TechNode"]:
        return [x for x, _ in self.recipe]

    def get_subnodes(self) -> List["TechNode"]:
        nodes = set()
        for child in self.get_children():
            nodes.update([child] + child.get_subnodes())
        return list(nodes)
    
    def get_requirements(self) -> List[str]:
        requirements = set([x.name for x, _ in self.recipe])
        if self.tool is not None:
            requirements.add(self.tool.name)
        if self.table is not None:
            requirements.add("crafting_table")
        if self.furnace is not None:
            requirements.add("furnace")
        return list(requirements)

    def update_children_info(self, nodes: List["TechNode"]) -> None:
        new_recipe = []
        for n, q in self.recipe:
            new_node = [x for x in nodes if x.name == n.name]
            if len(new_node) > 0:
                new_recipe.append((new_node[0], q))
            else:
                new_recipe.append((n, q))
        self.recipe = new_recipe

        if self.tool is not None and self.tool in nodes:
            self.tool  = [x for x in nodes if x == self.tool][0]

        for n, _ in self.recipe:
            n.update_children_info(nodes)

    def get_ingredients(self, inventory: Dict[str, int] = None) -> Dict["TechNode", int]:
        if inventory is None:
            inventory = dict()
        return self._get_ingredients(deepcopy(inventory))[0]

    def get_craft_order(self, inventory: Dict[str, int] = None) -> List["TechNode"]:
        if inventory is None:
            inventory = dict()
        return self._get_craft_order(deepcopy(inventory))[0]

    def purge(self, name):
        self.recipe = [(n, q) for n, q in self.recipe if n.name != name]
        for n, _ in self.recipe:
            n.purge(name)
        if self.tool is not None and self.tool.name == name:
            self.tool = None
        elif self.tool is not None:
            self.tool.purge(name)

    def to_json(self) -> None:
        return dict(
            name=self.name,
            collectable=self.collectable,
            tool=self.tool.to_json() if self.tool is not None else "",
            table=self.table,
            furnace=self.furnace,
            timestep=self.timestep,
            iteration=self.iteration,
            recipe=[[x[0].to_json(), x[1]] for x in self.recipe]
        )

    @staticmethod
    def from_json(info: dict) -> "TechNode":
        return TechNode(
            info["name"], 
            collectable=info["collectable"] if "collectable" in info else False,
            recipe=[(TechNode.from_json(x[0]), x[1]) for x in info["recipe"]] if "recipe" in info else [],
            tool=TechNode.from_json(info["tool"]) if "tool" in info and info["tool"] != "" else None,
            table=info["table"] if "table" in info else False,
            furnace=info["furnace"] if "furnace" in info else False,
            timestep=info["timestep"] if "timestep" in info else 0,
            iteration=info["iteration"] if "iteration" in info else 0
        )

    def _get_ingredients(self, inventory: Dict[str, int]) -> Tuple[Dict["TechNode", int], Dict[str, int]]:
        ingredients = dict()
        if self.collectable:
            return {self: 1}, inventory
        for node, quantity in self.recipe:
            if node.name in inventory:
                gathered = min(inventory[node.name], quantity)
                quantity -= gathered
                inventory[node.name] -= gathered
            for _ in range(quantity):
                node_ingredients, inventory = node._get_ingredients(inventory)
                for n, q in node_ingredients.items():
                    if n not in ingredients:
                        ingredients[n] = 0
                    ingredients[n] += q
        return ingredients, inventory

    def _get_craft_order(self, inventory: Dict[str, int]) -> Tuple[List["TechNode"], Dict[str, int]]:
        to_craft = []
        for node, quantity in self.recipe:
            if node.name in inventory:
                gathered = min(inventory[node.name], quantity)
                quantity -= gathered
                inventory[node.name] -= gathered
            for _ in range(quantity):
                next_crafts, inventory = node._get_craft_order(inventory)
                to_craft += next_crafts
        if not self.collectable:
            to_craft += [self]
        return to_craft, inventory

    def __eq__(self, __o: object) -> bool:
        return hasattr(__o, "name") and __o.name == self.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self) -> str:
        return json.dumps(self.to_json(), indent=4)
