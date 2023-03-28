import pickle
import argparse
import os
from gym import Wrapper
from datetime import datetime
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sb3_vpt.algorithm import VPTPPO
from sb3_vpt.policy import VPTPolicy
from tasks import make, get_specs
from techtree import explore_techtree
from tasks.base import *
from tasks.minedojo import MinedojoSemifastResetWrapper


class SetterWrapper(Wrapper):

    def recursive_set_attr(self, cls, attr, value):
        env = self.env
        while isinstance(env, Wrapper):
            if isinstance(env, cls):
                setattr(env, attr, value)
                return
            env = env.env
        if isinstance(env, cls):
            setattr(env, attr, value)

    def set_item_task(self, item):
        self.recursive_set_attr(MinedojoSemifastResetWrapper, "reset_freq", 5)
        self.recursive_set_attr(ClipWrapper, "prompt", ["collect " + item])
        self.recursive_set_attr(RewardWrapper, "item_rewards", {item: {"reward":1}})
        self.recursive_set_attr(SuccessWrapper, "all_conditions", {"item": {"type": item, "quantity": 1}})
        self.recursive_set_attr(TerminalWrapper, "max_steps", 1000)
        self.recursive_set_attr(TechTreeWrapper, "is_techtree_active", False)

    def set_techtree_task(self):
        self.recursive_set_attr(MinedojoSemifastResetWrapper, "reset_freq", 0)
        self.recursive_set_attr(ClipWrapper, "prompt", [])
        self.recursive_set_attr(RewardWrapper, "item_rewards", {})
        self.recursive_set_attr(SuccessWrapper, "all_conditions", {})
        self.recursive_set_attr(TerminalWrapper, "max_steps", 10000)
        self.recursive_set_attr(TechTreeWrapper, "is_techtree_active", True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test", 
                        help="Name of the experiment, will be used to create a results directory.")
    parser.add_argument("--config", type=str, default="wooden_pickaxe", 
                        help="Name of task. Will check tasks/task_specs.yaml for specified name.")
    parser.add_argument("--model", type=str, default="models/3x.model",
                        help="Path to file that stores model parameters for the policy.")
    parser.add_argument("--weights", type=str, default="weights/bc-house-3x.weights",
                        help="Path to the file that stores initial model weights for the policy.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Path to results dir.")
    parser.add_argument("--explore_steps", type=int, default=2000000,
                        help="Number of environment steps each iteration until exploration early stops.")
    parser.add_argument("--steps_per_subtask", type=int, default=2000000,
                        help="Number of environment steps to allow for each subtask.")
    parser.add_argument("--steps_per_iter", type=int, default=500,
                        help="Number of steps per environment each iteration.")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size for learning.")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of PPO epochs every iteration.")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of environment instances to run. Set to 0 to run 1 instance in the learner thread.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=.999,
                        help="Discount factor.")
    parser.add_argument("--kl_coef", type=float, default=.1,
                        help="Initial loss coefficient for VPT KL loss.")
    parser.add_argument("--kl_decay", type=float, default=.999,
                        help="How much to decay KL coefficient each iteration.")
    parser.add_argument("--cpu", action="store_true",
                        help="Use cpus over gpus.")
    args = parser.parse_args()

    _, task_specs, _ = get_specs(args.config)
    assert "tasks" not in task_specs["techtree_specs"], "Don't use this script with pretrained task adapters. Use techtree.py instead."

    # Prepare results dir
    log_dir = os.path.join(args.results_dir, args.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir)
    print("Created results directory", log_dir)

    # Get techtree exploration env
    env = make_vec_env(
        lambda task="", kwargs=dict(): SetterWrapper(make(task, **kwargs)), 
        n_envs=max(1, args.num_envs), 
        vec_env_cls=SubprocVecEnv if args.num_envs > 0 else DummyVecEnv, 
        env_kwargs=dict(
            task=args.config, 
            kwargs=dict(
                log_dir=log_dir
            )
        )
    )

    # Prepare policy
    max_tasks = task_specs["techtree_specs"].pop("max_tasks", 16)
    agent_parameters = pickle.load(open(args.model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    policy_kwargs["transformer_adapters"] = True
    policy_kwargs["n_adapters"] = max_tasks
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]

    device = "cpu" if args.cpu else "cuda"

    model = VPTPPO(
        VPTPolicy, 
        env, 
        n_steps=args.steps_per_iter,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=device, 
        policy_kwargs=dict(
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs,
            weights_path=args.weights,
        ),
        tensorboard_log=os.path.join(log_dir, "tb"),
        learning_rate=args.lr,
        gamma=args.gamma,
        vf_coef=1,
        kl_coef=args.kl_coef,
        kl_decay=args.kl_decay,
        n_tasks=1,
    )

    # Get tasks so far
    subtasks = []
    print("Starting training...")

    # Start training
    while True:
        if len(subtasks) >= max_tasks:
            print("Ran out of adapters for subtasks...")
            break

        # Explore to discover new subtask nodes
        print("Exploring subtasks")
        model.env.env_method("set_techtree_task")
        next_task = explore_techtree(model.env, model.policy, allow_sample_untrained=False, max_explore_steps=args.explore_steps)
        
        # If a new subtask is discovered
        if next_task == "success":
            print(args.config, "task success")
            break
        elif next_task == "done":
            print(args.config, "task failed")
            break
        else:
            print("Found new subtask:", next_task)

            # Modify environment for training subtask
            model.env.env_method("set_item_task", next_task)

            # Finetune adapters with RL
            print("Beginning adapter finetuning for subtask", next_task, "with adapter set", len(subtasks))
            model.reset()
            model.set_task_id(len(subtasks))
            model.learn(args.steps_per_subtask)
            model.save(os.path.join(log_dir, "task{}_{}".format(len(subtasks), next_task)))
            print("Subtask finetuning finished, saving model to task{}_{}.zip".format(len(subtasks), next_task))

            # Add new subtask
            subtasks.append(next_task)
            model.env.env_method("add_task", next_task)
            print("Finished adding new subtask", next_task)
    model.env.close()
