import pickle
import argparse
import os
from datetime import datetime
import sys
import shutil

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from sb3_vpt.algorithm import VPTPPO
from sb3_vpt.policy import VPTPolicy
from sb3_vpt.logging import LoggingCallback
from tasks import make, get_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test", 
                        help="Name of the experiment, will be used to create a results directory.")
    parser.add_argument("--config", type=str, default="base_task",
                        help="Minedojo task to run. Should be a minedojo task_id or exist in tasks/task_specs.yaml")
    parser.add_argument("--target_item", type=str, default="log",
                        help="Item to use if using base_task.")
    parser.add_argument("--model", type=str, default="models/3x.model",
                        help="Path to file that stores model parameters for the policy.")
    parser.add_argument("--weights", type=str, default="weights/bc-house-3x.weights",
                        help="Path to the file that stores initial model weights for the policy.")
    parser.add_argument("--load", type=str, default="",
                        help="Path to a zip filed to load from, saved by a previous run.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Path to results dir.")
    parser.add_argument("--steps", type=int, default=10000000, 
                        help="Total number of learner environement steps before learning stops.")
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
    parser.add_argument("--adapter_factor", type=float, default=16,
                        help="What reduction factor to use for adapters.")
    parser.add_argument("--cpu", action="store_true",
                        help="Use cpus over gpus.")
    parser.add_argument("--update_norms", action="store_true",
                        help="Update the layer norms of the network.")
    parser.add_argument("--final_layer", action="store_true",
                        help="Update the layer immediately before the heads.")
    parser.add_argument("--policy_head", action="store_true",
                        help="Update the policy head.")
    parser.add_argument("--no_transformer_adapters", action="store_true",
                        help="Trains adapters in the transformer.")
    parser.add_argument("--finetune_full", action="store_true",
                        help="Finetune the entire network.")
    parser.add_argument("--finetune_transformer", action="store_true",
                        help="Finetune the transformer and heads.")
    args = parser.parse_args()

    _, task_specs, _ = get_specs(args.config)
    vars(args).update(**task_specs)

    log_dir = os.path.join(args.results_dir, args.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(log_dir, "checkpoints"))

    # sys.stderr = open(os.path.join(log_dir, "error.txt"), "w")

    env = make_vec_env(
        lambda task="", kwargs=dict(): make(task, **kwargs), 
        n_envs=max(1, args.num_envs), 
        vec_env_cls=SubprocVecEnv if args.num_envs > 0 else DummyVecEnv, 
        env_kwargs=dict(
            task=args.config, 
            kwargs=dict(
                log_dir=log_dir,
                target_item=args.target_item
            )
        )
    )

    if args.load:
        model = VPTPPO.load(args.load, env)
        prev_log_dir = "/".join(args.load.split("/")[:-2])
        if "techtree_specs" in task_specs:
            shutil.copyfile(os.path.join(prev_log_dir, "techtree.json"), os.path.join(log_dir, "techtree.json"))
    else:

        agent_parameters = pickle.load(open(args.model, "rb"))
        policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
        policy_kwargs["transformer_adapters"] = not args.no_transformer_adapters
        policy_kwargs["adapter_factor"] = args.adapter_factor
        policy_kwargs["n_adapters"] = 1
        pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
        pi_head_kwargs["adapter_factor"] = args.adapter_factor
        pi_head_kwargs["n_adapters"] = 1

        model = VPTPPO(
            VPTPolicy, 
            env, 
            n_steps=args.steps_per_iter,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device="cpu" if args.cpu else "cuda", 
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
    model.learn(
        args.steps, 
        callback=LoggingCallback(model, log_dir)
    )
