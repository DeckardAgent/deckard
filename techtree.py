import numpy as np
import pickle
import argparse
import os
import sys
from datetime import datetime
import shutil
from copy import deepcopy
import torch as th
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.save_util import load_from_zip_file

from sb3_vpt.policy import VPTPolicy
from tasks import make, get_specs


def explore_techtree(env, policy, max_explore_steps=1e6, device="cuda", allow_sample_untrained=True):
    last_episode_starts = np.ones((env.num_envs,), dtype=bool)
    last_task_id = np.zeros((env.num_envs,), dtype=np.int16)
    last_vpt_states = policy.initial_state(env.num_envs)
    vpt_states = deepcopy(last_vpt_states)
    last_obs = env.reset()

    num_timesteps = 0
    while num_timesteps < max_explore_steps:

        with th.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(last_obs, device)
            episode_starts = th.tensor(last_episode_starts).float().to(device)
            actions, _, _, vpt_states = policy.forward(obs_tensor, vpt_states, episode_starts, last_task_id)

        new_obs, _, dones, infos = env.step(actions.cpu().numpy())
        num_timesteps = max(info["timestep"] for info in infos)

        curr_task_id = np.array([
            info["subgoal"] if "subgoal" in info else 0 
            for info in infos
        ], dtype=np.int16)

        if any(info["early_stop"] for info in infos):
            return "success"

        if not allow_sample_untrained and any("untrained" in info and info["untrained"] for info in infos):
            return [info["untrained"] for info in infos if "untrained" in info and info["untrained"]][0]

        last_obs = new_obs
        last_episode_starts = dones
        last_vpt_states = vpt_states
        last_task_id = curr_task_id

    return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test", 
                        help="Name of the experiment, will be used to create a results directory.")
    parser.add_argument("--config", type=str, default="wooden_pickaxe", 
                        help="Name of task. Will check tasks/task_specs for specified name.")
    parser.add_argument("--model", type=str, default="models/3x.model",
                        help="Path to file that stores model parameters for the policy.")
    parser.add_argument("--weights", type=str, default="weights/bc-house-3x.weights",
                        help="Path to the file that stores initial model weights for the policy.")
    parser.add_argument("--load", type=str, default="",
                        help="Path to a zip filed to load from, saved by a previous run.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Path to results dir.")
    parser.add_argument("--steps", type=int, default=1000000, 
                        help="Total number of learner environement steps before learning stops.")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of environment instances to run. Set to 0 to run 1 instance in the learner thread.")
    parser.add_argument("--cpu", action="store_true",
                        help="Use cpus over gpus.")
    args = parser.parse_args()

    _, task_specs, _ = get_specs(args.config)

    log_dir = os.path.join(args.results_dir, args.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir)

    env = make_vec_env(
        lambda task="", kwargs=dict(): make(task, **kwargs), 
        n_envs=max(1, args.num_envs), 
        vec_env_cls=SubprocVecEnv if args.num_envs > 0 else DummyVecEnv, 
        env_kwargs=dict(
            task=args.config, 
            kwargs=dict(
                log_dir=log_dir
            )
        )
    )

    if args.load:
        shutil.copyfile(args.load, os.path.join(log_dir, "techtree.json"))

    agent_parameters = pickle.load(open(args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    policy_kwargs["transformer_adapters"] = True
    policy_kwargs["n_adapters"] = len(task_specs["techtree_specs"]["tasks"]) if "tasks" in task_specs["techtree_specs"] \
        else task_specs["techtree_specs"].pop("max_tasks", 16)
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]

    device = "cpu" if args.cpu else "cuda"

    policy = VPTPolicy(
        env.observation_space,
        env.action_space,
        lambda x: 0,
        policy_kwargs=policy_kwargs, 
        pi_head_kwargs=pi_head_kwargs,
        weights_path=args.weights
    ).to(device)
    if "tasks" in task_specs["techtree_specs"]:
        for task_id, task_weights in enumerate(task_specs["techtree_specs"]["tasks"].values()):
            if not task_weights:
                continue
            _, params, _ = load_from_zip_file(task_weights, device=device)
            for n, x in policy.model.named_modules():
                if "img_process" not in n and n.split(".")[-1] == "adapter":
                    x.task_adapters[task_id].load_state_dict(
                        {".".join(k.split(".")[2:]): v for k, v in params["policy.model." + n].items()}
                    )
    policy.requires_grad_(False)
    policy.set_training_mode(False)

    explore_techtree(env, policy, args.steps, device)
