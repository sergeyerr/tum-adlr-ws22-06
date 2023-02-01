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

    def __init__(self, cfg, train_tasks, eval_tasks):
        self.cfg = cfg
        self.general_training_args = cfg["training"]
        self.training_args = cfg["training"][str(cfg["agent"]["name"])]
        # print(f"training args {json.dumps(self.training_args, indent=4)}")
        self.agent_args = cfg["agent"][str(cfg["agent"]["name"])]
        # print(f"agent args{json.dumps(self.agent_args, indent=4)}")
        self.env_args = cfg["env"]
        # print(f"env args{json.dumps(self.env_args, indent=4)}")
        self.validation_args = cfg["validation"]
        # print(f"validation args{json.dumps(self.validation_args, indent=4)}")
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks

    def run(self, experiment_path, init_wandb=True, ood=False, pass_params=False):

        experiment_name = self.agent_args["experiment_name"]

        if ood:
            experiment_name = experiment_name + "_ood"

        if pass_params:
            experiment_name = experiment_name + "_pass_params"

        if not self.general_training_args["random"]:
            T.manual_seed(self.general_training_args["seed"])
            T.backends.cudnn.deterministic = True
            T.backends.cudnn.benchmark = False
            np.random.seed(self.general_training_args["seed"])
            random.seed(self.general_training_args["seed"])

        # Weights and biases initialization
        if init_wandb:
            wandb.init(project="ADLR randomized envs", entity="tum-adlr-ws22-06", config=self.cfg)
        else:
            wandb.init(mode="disabled")

        agent_experiment_path = os.path.join(experiment_path, f"{experiment_name}")
        os.mkdir(agent_experiment_path)
        os.mkdir(os.path.join(agent_experiment_path, "saves"))

        n_actions = self.train_tasks[0].action_space.shape[0] if\
            type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n

        env_info = {"input_dims": int(np.prod(self.train_tasks[0].observation_space.shape)), "n_actions": n_actions,
                    "max_action": self.train_tasks[0].action_space.high}

        if str(self.cfg["agent"]["name"]) == "ddpg":
            algorithm = DDPGAgent
        elif str(self.cfg["agent"]["name"]) == "sac":
            algorithm = SACAgent
        elif str(self.cfg["agent"]["name"]) == "sac2":
            algorithm = SACAgent2

        agent = algorithm(**self.agent_args, **self.training_args, **self.general_training_args, **env_info)

        if self.validation_args["log_model_wandb"]:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(agent.pi, log="all", log_freq=self.validation_args["log_model_every_training_batch"])
            temp = self.validation_args["log_model_every_training_batch"]
            print(f"================= {f'Sending weights to W&B every {temp} batch'} =================")

        noise = ZeroNoise(size=n_actions)

        print_run_info(self.train_tasks[0], agent, self.agent_args, self.training_args, self.env_args,
                       self.validation_args, noise)

        reward_history = deque(maxlen=100)
        solved_tasks = [None] * self.validation_args["n_eval_tasks"]
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
                        # it does not make sense to re-collect the initial steps for every task, because in sac
                        # we only have one replay buffer for all tasks!
                        with T.no_grad():
                            if episode == 0 and initial_steps < self.general_training_args["num_initial_steps"]:
                                action = env.action_space.sample()
                            else:
                                action = agent.action(obs, addNoise=True, noise=noise)

                        # TODO Ignore the "done" signal if it comes from hitting the time horizon.

                        # Take step in environment
                        new_obs, reward, done, _, _ = env.step(action)
                        episode_reward += reward

                        # Store experience
                        agent.experience(obs, action, reward, new_obs, done)

                        # Update obs
                        obs = new_obs
                        total_steps_per_task += 1
                        initial_steps += 1
                        # End episode if done
                        if done:
                            break

            if agent.replay_buffer.size() > self.training_args["min_replay_size"]:
                for train_step in range(self.general_training_args["num_train_steps_per_itr"]):
                        loss = agent.train()
                        # Loss information kept for monitoring purposes during training
                        actor_loss += loss['actor_loss']
                        critic_loss += loss['critic_loss']
                        wandb.log({"Training episode": episode, "Batch": episode * self.general_training_args["num_train_steps_per_itr"] + train_step,
                                   "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})
                agent.update_target_network()

            reward_history.append(episode_reward)
            print(f"episode actor loss is: {loss['actor_loss']} \t episode critic loss is: {loss['critic_loss']}")
            print(f"Training episode: {episode} Episode reward: {episode_reward}"
                  f" Average reward: {np.mean(reward_history)}")
            print("_______________________________________________________________\n\n\n")

            wandb.log({"Training episode": episode, "Episode reward": episode_reward,
                       "Average reward": np.mean(reward_history)})

            if episode % self.validation_args["eval_interval"] == 0:
                print("starting evaluation")
                for task_id, eval_task in enumerate(self.eval_tasks):
                    solved = validate(agent, self.validation_args, agent_experiment_path, episode, eval_task, task_id)
                    if solved:
                        print(f"{str(self.cfg['agent']['name'])} solved task {task_id}!!")
                        solved_tasks[task_id] = solved

                if all(solved_tasks):
                    print(f"{str(self.cfg['agent']['name'])} solved all tasks (but not necessarily in a row)!!")
                    break
                print("evaluation over\n")

        # TODO note down the episodes in which the tasks were solved, so we can check the videos afterwards
        print(f"{str(self.cfg['agent']['name']) + '_ood' if ood else ''}{'_wp' if pass_params else ''}"
              f" training is over\n following tasks have been solved\n")
        print(f"{['solved task: ' + str(s) for s, i in enumerate(solved_tasks) if i]}\n\n")
        with open(f"{experiment_path}/solved_env.txt", "a") as f:
            f.write(
                f"{str(self.cfg['agent']['name']) + '_ood' if ood else ''}{'_wp' if pass_params else ''}"
                f" has solved the following tasks\n"
                f"{['solved task: ' + str(s) for s, i in enumerate(solved_tasks) if i]}\n")
