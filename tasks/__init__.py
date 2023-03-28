from omegaconf import OmegaConf
from tasks.minedojo import make_minedojo


CUTOM_TASK_SPECS = OmegaConf.to_container(OmegaConf.load("tasks/task_specs.yaml"))


def get_specs(task, **kwargs):
    # Get task data and task id
    if task in CUTOM_TASK_SPECS:
        yaml_specs = CUTOM_TASK_SPECS[task].copy()
        task_id = yaml_specs.pop("task_id", task)
        assert "sim" in yaml_specs, "task_specs.yaml must define sim attribute"
    else:
        yaml_specs = dict()
        task_id = task

    if "target_item" in kwargs and task == "base_task":
        yaml_specs["clip_specs"]["prompts"].append("collect " + kwargs["target_item"])
        yaml_specs["reward_specs"]["item_rewards"][kwargs["target_item"]] = dict(reward=1)
        yaml_specs["success_specs"]["all"]["item"]["type"] = kwargs["target_item"]

    # Get minedojo specs
    sim_specs = yaml_specs.pop("sim_specs", dict())

    # Get our task specs
    task_specs = dict(
        clip=False,
        fake_clip=False,
        fake_dreamer=False,
        subgoals=False,
    )
    task_specs.update(**yaml_specs)
    task_specs.update(**kwargs)
    assert not (task_specs["clip"] and task_specs["fake_clip"]), "Can only use one reward shaper"

    return task_id, task_specs, sim_specs


def make(task: str, **kwargs):
    # Get our custom task specs
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict

    # Make minedojo env
    env = make_minedojo(task_id, task_specs, sim_specs)
    
    return env
