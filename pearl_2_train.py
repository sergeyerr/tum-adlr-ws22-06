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
from EnvironmentUtils import LunarEnvRandomFabric, LunarEnvFixedFabric, LunarEnvHypercubeFabric
from Noise import ZeroNoise

import wandb
from agents import PEARLAgent2

from utils import print_run_info, validate_task
from baseline_train import BaselineExperiment


class PEARLE2xperiment(BaselineExperiment):

    def __init__(self, cfg, train_tasks, eval_tasks, experiment_path):
        super().__init__(cfg, train_tasks, eval_tasks, experiment_path)
        self.agent_name = "pearl2"
        self.training_args = cfg["training"]
        self.agent_args = cfg["agent"]
        self.task_idx = 0

    def run(self, **kwargs):

        print("Pearl experiment agent name is: " + self.agent_name)
        # clear reward history, list of solved tasks etc.
        self.reset_variables()

        self.ood = kwargs["ood"]
        init_wandb = kwargs["init_wandb"]

        # create folder based on experiment_name
        self.make_experiment_directory()

        if self.ood:
            self.train_tasks = self.train_tasks_array[2]
            self.eval_tasks = self.eval_tasks_array[2]
        else:
            self.train_tasks = self.train_tasks_array[0]
            self.eval_tasks = self.eval_tasks_array[0]

        n_actions = self.train_tasks[0].action_space.shape[0] if \
            type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n

        env_info = {"obs_dim": int(np.prod(self.train_tasks[0].observation_space.shape)),
                         "n_actions": n_actions, "max_action": self.train_tasks[0].action_space.high}

        # this is critical so that the q and v functions have the right input size
        env_info["input_dims"] = env_info["obs_dim"] + self.agent_args["latent_size"]

        self.agent = PEARLAgent2(**self.agent_args, **self.training_args, **env_info)

        # Weights and biases initialization
        if init_wandb:
            wandb.init(project="ADLR Unified training", entity="tum-adlr-ws22-06", config=self.cfg)
        else:
            wandb.init(mode="disabled")

        if self.validation_args["log_model_wandb"]:
            # assumes that the model has only one actor, we may also log different models differently
            #wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args["log_model_every_training_batch"])
            temp = self.validation_args["log_model_every_training_batch"]
            print(
                f"================= {f'Sending weights to W&B every {temp} batch'} =================")

        noise = ZeroNoise(n_actions)

        print_run_info(self.train_tasks[0], self.agent, self.agent_args, self.training_args, self.env_args,
                       self.validation_args, noise)

        # meta-training loop
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for episode in range(self.general_training_args["episodes"]):
            self.episode_rewards = []
            if episode == 0:
                print('collecting initial pool of data for train and eval')
                for idx, env in enumerate(self.train_tasks):
                    env.reset()
                    self.task_idx = idx
                    self.roll_out(self.general_training_args["num_initial_steps"], explore=True)

            # Sample data from train tasks.
            print('Sample data from train tasks')
            for i in range(self.general_training_args["num_tasks_sample"]):
                idx = np.random.randint(self.general_training_args["n_train_tasks"])
                self.task_idx = idx
                env = self.train_tasks[idx]
                env.reset()
                #print(env.wind_idx, env.torque_idx)

                gravity, enable_wind, wind_power, turbulence_power = env.gravity, env.enable_wind, \
                    env.wind_power, env.turbulence_power

                print(f"Gravity: {round(gravity,3)}\t Wind: {enable_wind}\t Wind power: {round(wind_power,3)}\t"
                      f" Turbulence power: {round(turbulence_power,3)}")


                self.roll_out(self.training_args["num_steps"])

            self.reward_history.append(self.episode_rewards)

            print("Sample train tasks and compute gradient updates on parameters.")
            loss = 0
            
            actor_loss = 0.0
            critic_loss = 0.0
            for train_step in range(self.general_training_args["num_train_steps_per_itr"]):
                indices = np.random.choice(self.general_training_args["n_train_tasks"], self.training_args["meta_batch"])
                loss = self.agent.optimize(indices)
                # Loss information kept for monitoring purposes during training
                actor_loss += loss['actor_loss']
                critic_loss += loss['critic_loss']

                #wandb.log({"Training episode": episode, "Batch": episode * self.general_training_args["num_train_steps_per_itr"] + train_step,
                       #    "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})

            # averaging per number of batches
            actor_loss /= self.general_training_args["num_train_steps_per_itr"]
            critic_loss /= self.general_training_args["num_train_steps_per_itr"]
            self.log_episode_reward( actor_loss , critic_loss, episode, self.episode_rewards)

            # TURNING IT OFF BECAUSE WE DON'T CARE IN CASE OF CLOUD
            # mechanism to abort training if bad convergence
            # if not self.converging(episode):
            #     print("breaking due to convergence")
            #     break

            if episode % self.validation_args["eval_interval"] == 0:
                solved_all_tests = self.run_test_tasks(episode, pearl=False, pearl_2=True)
                if solved_all_tests:
                    break

        self.log_end()

    # one path length is between 80 - 200 steps.
    def roll_out(self, num_samples, explore=False):

        # start from the prior
        context_size = self.agent_args["context_size"]
        total_num_samples = 0
        num_trajs = 0
        paths = []
        env = self.train_tasks[self.task_idx]
        while total_num_samples < num_samples:
            observations = []
            actions = []
            rewards = []
            terminals = []
            o, _ = env.reset()
            next_o = None

            # inner most loop
            # here we interact with the environment
            rollout_reward = 0.0
            for step in range(self.general_training_args["max_path_length"]):
                if explore:
                    a = env.action_space.sample()
                else:
                    if step != 0:
                        number_of_samples = len(observations[-context_size:])
                        o_tensor = torch.Tensor(np.array(observations[-context_size:])).reshape(number_of_samples, -1)
                        a_tensor = torch.Tensor(np.array(actions[-context_size:])).reshape(number_of_samples, -1)
                        r_tensor = torch.Tensor(np.array(rewards[-context_size:])).reshape(number_of_samples, -1)
                        context = torch.cat([o_tensor, a_tensor, r_tensor], dim=1)
                        a = self.agent.action(o, [context], addNoise=True)
                    else:
                        a = self.agent.action(o, [torch.empty(0, self.agent.encoder_in_size)], addNoise=True)
                next_o, r, d, _, _ = env.step(a)
                observations.append(o)
                rewards.append(r)
                terminals.append(d)
                actions.append(a)
                o = next_o
                rollout_reward += r
                total_num_samples += 1

                if d:
                    break
            self.episode_rewards.append(rollout_reward)

            num_trajs += 1

            actions = np.array(actions)
            if len(actions.shape) == 1:
                actions = np.expand_dims(actions, 1)

            observations = np.array(observations)
            if len(observations.shape) == 1:
                observations = np.expand_dims(observations, 1)
                next_o = np.array([next_o])

            next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
            path = dict(
                o=observations,  # np.array [[1,2,4,5],[1,2,4,5],[1,2,4,5],...]
                a=actions,
                r=np.array(rewards).reshape(-1, 1),  # [[1],[1.4],[34],...]
                o2=next_observations,
                d=np.array(terminals).reshape(-1, 1),  # [[false],[false],[true],...]
            )

            paths.append(path)
            self.agent.replay_buffer.add_paths(self.task_idx, paths)
