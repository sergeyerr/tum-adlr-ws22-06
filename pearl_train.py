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
from agents import PEARLAgent

from utils import print_run_info, validate
from baseline_train import BaselineExperiment


class PEARLExperiment(BaselineExperiment):

    def __init__(self, cfg, train_tasks, eval_tasks, experiment_path):
        super().__init__(cfg, train_tasks, eval_tasks, experiment_path)
        self.agent_name = "pearl"
        self.training_args = cfg["training"]
        self.agent_args = cfg["agent"]

        self.episode_reward = 0.0
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

        self.agent = PEARLAgent(**self.agent_args, **self.training_args, **env_info)

        # Weights and biases initialization
        if init_wandb:
            wandb.init(project="ADLR randomized envs with Meta RL", entity="tum-adlr-ws22-06", config=self.cfg)
        else:
            wandb.init(mode="disabled")

        if self.validation_args["log_model_wandb"]:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args["log_model_every_training_batch"])
            temp = self.validation_args["log_model_every_training_batch"]
            print(
                f"================= {f'Sending weights to W&B every {temp} batch'} =================")

        noise = ZeroNoise(n_actions)

        print_run_info(self.train_tasks[0], self.agent, self.agent_args, self.training_args, self.env_args,
                       self.validation_args, noise)

        # meta-training loop
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for episode in range(self.general_training_args["episodes"]):

            if episode == 0:
                print('collecting initial pool of data for train and eval')
                for idx, env in enumerate(self.train_tasks):
                    env.reset()
                    self.task_idx = idx
                    self.roll_out(self.general_training_args["num_initial_steps"], 1, np.inf, explore=True)

            # Sample data from train tasks.
            actor_loss = 0.0
            critic_loss = 0.0
            self.episode_reward = 0.0
            print('Sample data from train tasks')
            for i in range(self.general_training_args["num_tasks_sample"]):
                idx = np.random.randint(self.general_training_args["n_train_tasks"])
                self.task_idx = idx
                env = self.train_tasks[idx]
                env.reset()

                gravity, enable_wind, wind_power, turbulence_power = env.gravity, env.enable_wind, \
                    env.wind_power, env.turbulence_power

                print(f"Gravity: {round(gravity,3)}\t Wind: {enable_wind}\t Wind power: {round(wind_power,3)}\t"
                      f" Turbulence power: {round(turbulence_power,3)}")

                # collect some trajectories with z ~ prior
                self.agent.encoder_replay_buffer.clear_buffer(idx)
                if self.training_args["num_steps_prior"] > 0:
                    self.roll_out(self.training_args["num_steps_prior"], 1, np.inf)
                # even if encoder is trained only on samples from the prior,
                # the policy needs to learn to handle z ~ posterior
                if self.training_args["num_extra_rl_steps_posterior"] > 0:
                    self.roll_out(self.training_args["num_extra_rl_steps_posterior"],
                                  1, self.training_args["update_post_train"], add_to_enc_buffer=False)
                # collect some trajectories with z ~ posterior
                # self.agent.encoder_replay_buffer.clear_buffer(idx) # do we need to clear here??
                if self.training_args["num_steps_posterior"] > 0:
                    self.roll_out(self.training_args["num_steps_posterior"], 1, self.training_args["update_post_train"])

            self.reward_history.append(self.episode_reward)

            print("Sample train tasks and compute gradient updates on parameters.")
            loss = 0
            for train_step in range(self.general_training_args["num_train_steps_per_itr"]):
                indices = np.random.choice(self.general_training_args["n_train_tasks"], self.training_args["meta_batch"])
                loss = self.agent.optimize(indices)
                # Loss information kept for monitoring purposes during training
                actor_loss += loss['actor_loss']
                critic_loss += loss['critic_loss']

                wandb.log({"Training episode": episode, "Batch": episode * self.general_training_args["num_train_steps_per_itr"] + train_step,
                           "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})

            self.log_episode_reward(loss, episode, self.episode_reward)

            # mechanism to abort training if bad convergence
            if not self.converging(episode):
                print("breaking due to convergence")
                break

            if episode % self.validation_args["eval_interval"] == 0:
                solved_all_tests = self.run_test_tasks(episode, pearl=True)
                if solved_all_tests:
                    break

        self.log_end()

    # one path length is between 80 - 200 steps.
    def roll_out(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True, explore=False):

        # start from the prior
        self.agent.clear_z()
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

            if num_trajs % resample_z_rate == 0:
                self.agent.sample_z()
            # inner most loop
            # here we interact with the environment
            for step in range(self.general_training_args["max_path_length"]):
                if explore:
                    a = env.action_space.sample()
                else:
                    a = self.agent.action(o, addNoise=True)
                next_o, r, d, _, _ = env.step(a)
                observations.append(o)
                rewards.append(r)
                terminals.append(d)
                actions.append(a)
                o = next_o
                self.episode_reward += r
                total_num_samples += 1

                if d:
                    break

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

            if add_to_enc_buffer:
                self.agent.encoder_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.agent.encoder_replay_buffer.sample_random_batch(self.task_idx,
                                                                               self.training_args[
                                                                                   "embedding_batch_size"],
                                                                               sample_context=True,
                                                                               use_next_obs_in_context=
                                                                               self.agent_args[
                                                                                   "use_next_obs_in_context"])
                self.agent.infer_posterior(context)
