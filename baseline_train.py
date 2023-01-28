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

    def run(self):

        experiment_name = self.agent_args["experiment_name"]

        if not self.general_training_args["random"]:
            T.manual_seed(self.general_training_args["seed"])
            T.backends.cudnn.deterministic = True
            T.backends.cudnn.benchmark = False
            np.random.seed(self.general_training_args["seed"])
            random.seed(self.general_training_args["seed"])

        # Weights and biases initialization
        wandb.init(project="ADLR randomized envs", entity="tum-adlr-ws22-06", config=self.cfg)
        wandb.init(mode="disabled")

        # Experiment directory storage
        env_path = os.path.join("experiments", "LunarLanderContinuous-v2")
        if not os.path.exists(env_path):
            os.makedirs(env_path)

        #ugly, but acceptable
        experiment_counter = 0
        while True:
            try:
                experiment_path = os.path.join(env_path, f"{experiment_name}_{experiment_counter}")
                os.mkdir(experiment_path)
                os.mkdir(os.path.join(experiment_path, "saves"))
                break
            except FileExistsError as e:
                experiment_counter += 1

        with open(os.path.join(experiment_path, 'parameters.json'), 'w') as f:
            OmegaConf.save(self.cfg, f)

        n_actions = self.train_tasks[0].action_space.shape[0] if\
            type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n

        env_info = {"input_dims": int(np.prod(self.train_tasks[0].observation_space.shape)), "n_actions": n_actions,
                    "max_action": self.train_tasks[0].action_space.high}

        # TODO: Modify this to call any other algorithm
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
        t = 0
        for episode in range(self.general_training_args["episodes"]):
            for i in range(self.general_training_args["num_tasks_sample"]):
                idx = np.random.randint(self.general_training_args["n_train_tasks"])
                env = self.train_tasks[idx]
                obs, info = env.reset()
                gravity, enable_wind, wind_power, turbulence_power = env.gravity, env.enable_wind, env.wind_power, env.turbulence_power
                episode_reward = 0.0
                actor_loss = 0.0
                critic_loss = 0.0

            # TODO I think the episode length is not long enough for sac to learn to land. The agent only learns
            #  to hover above the ground. Or I fucked up the sac while fixing sac2, because they share logic

                # Generate rollout
                for step in range(self.training_args["episode_length"]):

                    # Get actions
                    # why no_grad()?
                    with T.no_grad():
                        if t >= self.general_training_args["num_initial_steps"]:
                            action = agent.action(obs, addNoise=True, noise=noise)
                        else:
                            action = env.action_space.sample()

                    # TODO Ignore the "done" signal if it comes from hitting the time horizon.

                    # Take step in environment
                    new_obs, reward, done, _, _ = env.step(action)
                    episode_reward += reward

                    # Store experience
                    agent.experience(obs, action, reward, new_obs, done)

                    # Update obs
                    obs = new_obs
                    t += 1
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
            print(f"Training episode: {episode} Episode reward: {episode_reward} Average reward: {np.mean(reward_history)}")
            print(f"Gravity: {gravity} Wind: {enable_wind} Wind power: {wind_power} Turbulence power: {turbulence_power}")
            wandb.log({"Training episode": episode, "Episode reward": episode_reward, "Average reward": np.mean(reward_history),
                       "Gravity": gravity, "Wind": enable_wind, "Wind power": wind_power, "Turbulence power": turbulence_power})
            solved_tasks = []
            if episode % self.validation_args["eval_interval"] == 0:
                for task_id, eval_task in enumerate(self.eval_tasks):
                    solved = validate(agent, self.validation_args, experiment_path, episode, eval_task, task_id)
                    if solved:
                        print(f"solved task {task_id}!!")
                        solved_tasks.append(solved)

                if len(solved_tasks) == len(self.eval_tasks):
                    print(f"solved all tasks in a row!!")
                    break

