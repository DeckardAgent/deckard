import numpy as np
import os
from csv import DictWriter
from datetime import datetime
import psutil
import json

from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
    def __init__(self, model, log_dir, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir
        self.model = model
        self.save_freq = 100

        self.log_out = []
        self.iteration = -1
        self.num_dones = 0
        self.total_steps = 0
        self.rollout_steps = 0
        self.cum_rewards = None
        self.successes = None
        self.success_count = 0
        self.rollout_start_time = None
        self.update_start_time = None
        self.iter_rewards = []
        self.subgoals = []
        self.clip_scores = None

    def _on_training_start(self) -> None:
        self.cum_rewards = np.zeros(self.training_env.num_envs)
        self.successes = np.zeros(self.training_env.num_envs).astype(bool)
        self.clip_scores = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        self.cum_rewards += self.locals["rewards"]
        eps_rewards = self.cum_rewards[np.where(self.locals["dones"])].tolist()
        for r in eps_rewards:
            self.logger.record_mean("custom/reward", r)
        self.iter_rewards += eps_rewards
        self.cum_rewards *= (1 - self.locals["dones"])
        if "subgoal" in self.locals["infos"][0]:
            self.subgoals += [info["subgoal"] for i, info in enumerate(self.locals["infos"]) if self.locals["dones"][i]]
        
        success = np.array([x["success"] if "success" in x else False for x in self.locals["infos"]])
        self.successes = np.bitwise_or(success, self.successes)
        self.success_count += np.sum(np.bitwise_and(self.successes, self.locals["dones"]))
        self.successes = np.where(self.locals["dones"], 0, self.successes)

        # Log exploration
        if self.clip_scores is not None and "clip_scores" in self.locals["infos"][0]:
            for i, info in enumerate(self.locals["infos"]):
                self.clip_scores[i].append(info["clip_scores"])
                if self.locals["dones"][i]:
                    with open(os.path.join(self.log_dir, "clip_scores.txt"), "a") as f:
                        f.write(json.dumps(self.clip_scores[i]) + "\n")
                        self.clip_scores[i] = []

        self.num_dones += np.sum(self.locals["dones"])
        self.rollout_steps += self.training_env.num_envs
        self.total_steps += self.training_env.num_envs
        if "craft_steps" in  self.locals["infos"][0]:
            self.total_steps += sum(info["craft_steps"] for info in self.locals["infos"])

        return True

    def _on_rollout_start(self):
        if self.update_start_time is not None:
            print("Finished updated in", datetime.now() - self.update_start_time)
            self.logger.record("custom/update_secs", (datetime.now() - self.update_start_time).total_seconds())
            print()

        self.iteration += 1
        if self.iteration > 0:
            self.log_out.append(dict(
                timesteps=self.total_steps,
                rollout_secs=self.update_start_time - self.rollout_start_time,
                update_secs=datetime.now() - self.update_start_time,
                dones=self.num_dones,
                success=self.success_count/self.num_dones if self.num_dones > 0 else np.nan,
                reward=np.mean(self.iter_rewards) if self.num_dones > 0 else np.nan,
                max_reward=np.amax(self.iter_rewards) if self.num_dones > 0 else np.nan,
                subgoals=np.mean(self.subgoals) if len(self.subgoals) > 0 else np.nan,
                max_subgoals=np.amax(self.subgoals) if len(self.subgoals) > 0 else np.nan,
                memory=psutil.virtual_memory()[3]/1e9
            ))
            with open(os.path.join(self.log_dir, "stats.csv"), "w") as f:
                writer = DictWriter(f, fieldnames=list(self.log_out[0].keys()))
                writer.writeheader()
                writer.writerows(self.log_out)
            if self.iteration % self.save_freq == 0:
                self.model.save(os.path.join(self.log_dir, "checkpoints", "timestep_{}".format(self.num_timesteps)))

        self.num_dones = 0
        self.rollout_steps = 0
        self.rollout_start_time = datetime.now()
        self.iter_rewards = []
        self.subgoals = []
        self.success_count = 0
        print("Starting rollout")

    def _on_rollout_end(self):
        self.update_start_time = datetime.now()
        print("Finished rollout in", self.update_start_time - self.rollout_start_time)
        print("\tMax reward:", np.amax(self.iter_rewards) if self.num_dones > 0 else np.nan)
        print("\tLast rewards:", np.mean(self.iter_rewards))
        print("\tNum dones:", self.num_dones)
        print("\tMemory:", psutil.virtual_memory()[3]/1e9)
        print("\tFPS:", self.rollout_steps / (self.update_start_time - self.rollout_start_time).total_seconds())

        self.logger.record("custom/rollout_secs", (self.update_start_time - self.rollout_start_time).total_seconds())
        self.logger.record("custom/completed_episodes", self.num_dones)
        self.logger.record("custom/FPS", self.rollout_steps / (self.update_start_time - self.rollout_start_time).total_seconds())
        if len(self.iter_rewards) > 0:
            self.logger.record("custom/max_reward", np.amax(self.iter_rewards))
        if len(self.subgoals) > 0:
            self.logger.record("custom/max_subgoals", np.amax(self.subgoals))
            self.logger.record("custom/subgoals", np.mean(self.subgoals))
        self.logger.record("custom/memory", psutil.virtual_memory()[3]/1e9)

        if self.num_dones > 0:
            print("\tSuccesses: {}/{}={}".format(self.success_count, self.num_dones, self.success_count/self.num_dones))
            self.logger.record("custom/success", self.success_count/self.num_dones)

        print("Starting update")