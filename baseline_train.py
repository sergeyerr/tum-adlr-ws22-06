import argparse
import json
import os
import random
from collections import deque

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch as T
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from EnvironmentUtils import StateInjectorWrapper, LunarEnvRandomFabric, LunarEnvFixedFabric, LunarEnvHypercubeFabric

import wandb
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ZeroNoise
from agents import DDPGAgent, SACAgent, SACAgent2

from utils import print_run_info, validate


class BaselineExperiment(object):

    def __init__(self, cfg, train_tasks, eval_tasks, experiment_path):
        self.cfg = cfg
        self.agent_name = str(cfg["agent"]["name"])
        self.general_training_args = cfg["training"]
        self.training_args = cfg["training"]
        self.agent_args = cfg["agent"]
        self.experiment_name = ""
        self.env_args = cfg["env"]
        self.validation_args = cfg["validation"]

        # normal train_tasks, train_tasks_with_params, train_tasks_ood
        self.train_tasks_array = train_tasks
        self.eval_tasks_array = eval_tasks
        # default tasks are the in distribution training and eval tasks
        self.train_tasks = self.train_tasks_array[0]
        self.eval_tasks = self.eval_tasks_array[0]

        self.ood = False
        self.pass_params = False
        self.experiment_path = experiment_path
        self.agent_experiment_path = ""
        self.solved_tasks = [False] * self.validation_args["n_eval_tasks"]
        self.solved_episodes = [""] * self.validation_args["n_eval_tasks"]
        self.last_avg_reward = -100000
        self.reward_history = deque(maxlen=100)
        self.no_convergence_counter = 0

        if str(self.agent_name) == "ddpg":
            self.algorithm = DDPGAgent
        elif str(self.agent_name) == "sac":
            self.algorithm = SACAgent
        elif str(self.agent_name) == "sac2":
            self.algorithm = SACAgent2

        self.agent = None

        if not self.general_training_args["random"]:
            T.manual_seed(self.general_training_args["seed"])
            T.backends.cudnn.deterministic = True
            T.backends.cudnn.benchmark = False
            np.random.seed(self.general_training_args["seed"])
            random.seed(self.general_training_args["seed"])

    def run(self, **kwargs):

        print("Baseline agent name is: " + self.agent_name)
        # clear reward history, list of solved tasks etc.
        self.reset_variables()

        self.ood = kwargs["ood"]
        self.pass_params = kwargs["pass_params"]
        init_wandb = kwargs["init_wandb"]

        # create folder based on experiment_name
        self.make_experiment_directory()

        if self.pass_params and self.ood:
            self.train_tasks = self.train_tasks_array[3]
            self.eval_tasks = self.eval_tasks_array[3]
        else:
            if self.ood:
                self.train_tasks = self.train_tasks_array[2]
                self.eval_tasks = self.eval_tasks_array[2]
            elif self.pass_params:
                self.train_tasks = self.train_tasks_array[1]
                self.eval_tasks = self.eval_tasks_array[1]
            else:
                self.train_tasks = self.train_tasks_array[0]
                self.eval_tasks = self.eval_tasks_array[0]

        n_actions = self.train_tasks[0].action_space.shape[0] if \
            type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n

        env_info = {"input_dims": int(np.prod(self.train_tasks[0].observation_space.shape)),
                         "n_actions": n_actions, "max_action": self.train_tasks[0].action_space.high}

        self.agent = self.algorithm(**self.agent_args, **self.training_args, **env_info)

        # Weights and biases initialization
        if init_wandb:
            wandb.init(project="ADLR randomized envs", entity="tum-adlr-ws22-06", config=self.cfg)
        else:
            wandb.init(mode="disabled")

        if self.validation_args["log_model_wandb"]:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args["log_model_every_training_batch"])
            temp = self.validation_args["log_model_every_training_batch"]
            print(f"================= {f'Sending weights to W&B every {temp} batch'} =================")

        noise = ZeroNoise(size=n_actions)

        print_run_info(self.train_tasks[0], self.agent, self.agent_args, self.training_args, self.env_args,
                       self.validation_args, noise)

        for episode in range(self.general_training_args["episodes"]):
            episode_reward = 0.0
            actor_loss = 0.0
            critic_loss = 0.0
            initial_steps = 0

            for i in range(self.general_training_args["num_tasks_sample"]):
                idx = np.random.randint(self.general_training_args["n_train_tasks"])
                env = self.train_tasks[idx]
                gravity, enable_wind, wind_power, turbulence_power = env.gravity, env.enable_wind,\
                    env.wind_power, env.turbulence_power

                print(f"Gravity: {round(gravity, 3)}\t Wind: {enable_wind}\t Wind power: {round(wind_power, 3)}\t"
                      f" Turbulence power: {round(turbulence_power, 3)}")

                total_steps_per_task = 0
                while total_steps_per_task < self.training_args["episode_length"]:
                    obs, info = env.reset()
                    # Generate rollout
                    for step in range(self.general_training_args["max_path_length"]):

                        # Get actions
                        if episode == 0 and initial_steps < self.general_training_args["num_initial_steps"]:
                            action = env.action_space.sample()
                        else:
                            action = self.agent.action(obs, addNoise=True, noise=noise)

                        # TODO Ignore the "done" signal if it comes from hitting the time horizon.

                        # Take step in environment
                        new_obs, reward, done, _, _ = env.step(action)
                        episode_reward += reward

                        # Store experience
                        self.agent.experience(obs, action, reward, new_obs, done)

                        # Update obs
                        obs = new_obs
                        total_steps_per_task += 1
                        initial_steps += 1
                        # End episode if done
                        if done:
                            break

            self.reward_history.append(episode_reward)
            loss = 0
            if self.agent.replay_buffer.size() > self.training_args["min_replay_size"]:
                for train_step in range(self.general_training_args["num_train_steps_per_itr"]):
                        loss = self.agent.train()
                        # Loss information kept for monitoring purposes during training
                        actor_loss += loss['actor_loss']
                        critic_loss += loss['critic_loss']
                        wandb.log({"Training episode": episode, "Batch": episode * self.general_training_args["num_train_steps_per_itr"] + train_step,
                                   "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})

            # log average results and episode result
            self.log_episode_reward(loss, episode, episode_reward)

            # mechanism to abort training if bad convergence
            if not self.converging(episode):
                print("breaking due to convergence")
                break

            if episode % self.validation_args["eval_interval"] == 0:
                solved_all_tests = self.run_test_tasks(episode)
                if solved_all_tests:
                    break

        self.log_end()

    def reset_variables(self):
        self.solved_tasks = [False] * self.validation_args["n_eval_tasks"]
        self.solved_episodes = [""] * self.validation_args["n_eval_tasks"]
        self.last_avg_reward = -100000
        self.reward_history = deque(maxlen=100)
        self.no_convergence_counter = 0

    def make_experiment_directory(self):
        self.experiment_name = self.agent_args["experiment_name"]

        if self.ood:
            self.experiment_name = self.experiment_name + "_ood"

        if self.pass_params:
            self.experiment_name = self.experiment_name + "_pass_params"

        self.agent_experiment_path = os.path.join(self.experiment_path, f"{self.experiment_name}")
        os.mkdir(self.agent_experiment_path)
        os.mkdir(os.path.join(self.agent_experiment_path, "saves"))

    def run_test_tasks(self, episode, pearl=False):
        print("starting evaluation")
        for task_id, eval_task in enumerate(self.eval_tasks):
            solved = validate(self.agent, self.validation_args, self.agent_experiment_path, episode, eval_task, task_id, pearl)
            if solved:
                print(f"{self.agent_name} solved task {task_id} in episode {episode}!!")
                self.solved_tasks[task_id] = solved
                self.solved_episodes[task_id] += f", {episode}"

        if all(self.solved_tasks):
            print(f"{self.agent_name} solved all tasks (but not necessarily in a row)!!")
            return True
        print("evaluation over\n")
        return False

    def log_episode_reward(self, loss, episode, episode_reward):
        print(f"episode actor loss is: {loss['actor_loss']} \t episode critic loss is: {loss['critic_loss']}")
        print(f"Training episode: {episode} Episode reward: {episode_reward}"
              f" Average reward: {np.mean(self.reward_history)}")
        wandb.log({"Training episode": episode, "Episode reward": episode_reward,
                   "Average reward": np.mean(self.reward_history)})
        print("wandb logging successful")

    def converging(self, episode):
        # mechanism to abort training if bad convergence
        if self.last_avg_reward >= np.mean(self.reward_history):
            self.no_convergence_counter += 1
            print(f"convergence counter increased to: {self.no_convergence_counter}\n")
            print(f"threshold is: {self.general_training_args['abort_training_after']}")
            print("_______________________________________________________________\n\n\n")
            if self.no_convergence_counter > self.general_training_args["abort_training_after"]:
                print(f"{self.agent_name}{'_ood' if self.ood else ''}{'_wp' if self.pass_params else ''}"
                      f" aborting training in episode {episode} because agent is not converging")
                with open(f"{self.experiment_path}/solved_env.txt", "a") as f:
                    f.write(
                        f"{self.agent_name}{'_ood' if self.ood else ''}{'_wp' if self.pass_params else ''}"
                        f" did not converge. Training aborted in episode {episode}\n")
                return False
            else:
                return True
        else:
            self.no_convergence_counter = 0
            self.last_avg_reward = np.mean(self.reward_history)
            print(f"convergence counter decreased to: {self.no_convergence_counter}")
            print("_______________________________________________________________\n\n\n")
            return True

    def log_end(self):
        print(f"{self.agent_name}{'_ood' if self.ood else ''}{'_wp' if self.pass_params else ''}"
              f" training is over\nfollowing tasks have been solved\n")
        print(f"{['solved task: ' + str(s) for s, i in enumerate(self.solved_tasks) if i]}\n\n")
        with open(f"{self.experiment_path}/solved_env.txt", "a") as f:
            f.write(
                f"{self.agent_name}{'_ood' if self.ood else ''}{'_wp' if self.pass_params else ''}"
                f" has solved the following tasks\n"
                f"{['solved task: ' + str(i) + ' in episodes: ' + self.solved_episodes[i][2:] for i, t in enumerate(self.solved_tasks) if t]}\n")
