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

class PEARLExperiment(object):

    def __init__(self, cfg, train_tasks, eval_tasks):
        self.cfg = cfg
        self.general_training_args = cfg["training"]
        self.training_args = cfg["training"]["pearl"]
        # print(f"training args {json.dumps(self.training_args, indent=4)}")
        self.agent_args = cfg["agent"]["pearl"]
        # print(f"agent args{json.dumps(self.agent_args, indent=4)}")
        self.env_args = cfg["env"]
        # print(f"env args{json.dumps(self.env_args, indent=4)}")
        self.validation_args = cfg["validation"]
        # print(f"validation args{json.dumps(self.validation_args, indent=4)}")

        # creates list of env with different parameterizations
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks

        self.n_actions = int(np.prod(self.train_tasks[0].action_space.shape)) \
            if type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n

        env_info = {"obs_dim": int(np.prod(self.train_tasks[0].observation_space.shape)), "n_actions": self.n_actions,
                    "max_action": self.train_tasks[0].action_space.high}

        # this is critical so that the q and v functions have the right input size
        env_info["input_dims"] = env_info["obs_dim"] + self.agent_args["latent_size"]
        self.agent = PEARLAgent(**self.agent_args, **self.training_args, **self.general_training_args, **env_info)

        self.episode_reward = 0.0
        self.task_idx = 0

    def run(self, experiment_path, init_wandb):

        if self.general_training_args["pass_env_parameters"]:
            print("It is not possible to run PEARL with pass_env_parameters set to True\n returning from experiment")
            return
        else:
            experiment_name = self.agent_args["experiment_name"]

        if not self.general_training_args["random"]:
            T.manual_seed(self.general_training_args["seed"])
            T.backends.cudnn.deterministic = True
            T.backends.cudnn.benchmark = False
            np.random.seed(self.general_training_args["seed"])
            random.seed(self.general_training_args["seed"])

        # Weights and biases initialization
        if init_wandb:
            wandb.init(project="ADLR randomized envs with Meta RL", entity="tum-adlr-ws22-06", config=self.cfg)
        else:
            wandb.init(mode="disabled")

        agent_experiment_path = os.path.join(experiment_path, f"{experiment_name}")
        os.mkdir(agent_experiment_path)
        os.mkdir(os.path.join(agent_experiment_path, "saves"))

        if self.validation_args["log_model_wandb"]:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args["log_model_every_training_batch"])
            temp = self.validation_args["log_model_every_training_batch"]
            print(
                f"================= {f'Sending weights to W&B every {temp} batch'} =================")

        noise = ZeroNoise(self.n_actions)

        print_run_info(self.train_tasks[0], self.agent, self.agent_args, self.training_args, self.env_args,
                       self.validation_args, noise)

        reward_history = deque(maxlen=100)
        solved_tasks = [None] * self.validation_args["n_eval_tasks"]

        # meta-training loop
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for episode in range(self.general_training_args["episodes"]):
            self.episode_reward = 0.0
            actor_loss = 0.0
            critic_loss = 0.0

            if episode == 0:
                print('collecting initial pool of data for train and eval')
                for idx, env in enumerate(self.train_tasks):
                    env.reset()
                    self.task_idx = idx
                    self.roll_out(self.general_training_args["num_initial_steps"], 1, np.inf)
            # Sample data from train tasks.
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

                # TODO understand why we delete the replay buffer here although we filled it in loop before
                self.agent.encoder_replay_buffer.clear_buffer(idx)

                # collect some trajectories with z ~ prior
                if self.training_args["num_steps_prior"] > 0:
                    self.roll_out(self.training_args["num_steps_prior"], 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.training_args["num_steps_posterior"] > 0:
                    self.roll_out(self.training_args["num_steps_posterior"], 1,
                                      self.training_args["update_post_train"])
                # even if encoder is trained only on samples from the prior,
                # the policy needs to learn to handle z ~ posterior
                if self.training_args["num_extra_rl_steps_posterior"] > 0:
                    self.roll_out(self.training_args["num_extra_rl_steps_posterior"],
                                      1, self.training_args["update_post_train"], add_to_enc_buffer=False)

            reward_history.append(self.episode_reward)
            # Sample train tasks and compute gradient updates on parameters.
            print("Sample train tasks and compute gradient updates on parameters.")
            for train_step in range(self.general_training_args["num_train_steps_per_itr"]):
                indices = np.random.choice(self.general_training_args["n_train_tasks"], self.training_args["meta_batch"])
                loss = self.agent.optimize(indices)
                # Loss information kept for monitoring purposes during training
                actor_loss += loss['actor_loss']
                critic_loss += loss['critic_loss']

                wandb.log({"Training episode": episode, "Batch": episode * self.general_training_args["num_train_steps_per_itr"] + train_step,
                           "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})

            print(f"episode actor loss is: {loss['actor_loss']} \t episode critic loss is: {loss['critic_loss']}")
            print(f"Training episode: {episode} Episode reward: {self.episode_reward}"
                  f" Average reward: {np.mean(reward_history)}")
            print("_______________________________________________________________\n\n\n")
            wandb.log({"Training episode": episode, "Episode reward": self.episode_reward,
                       "Average reward": np.mean(reward_history)})

            if episode % self.validation_args["eval_interval"] == 0:
                print("starting evaluation")
                for task_id, eval_task in enumerate(self.eval_tasks):
                    solved = validate(self.agent, self.validation_args, agent_experiment_path, episode,
                                      eval_task, task_id, pearl=True)
                    if solved:
                        print(f"pearl solved task {task_id}!!")
                        solved_tasks[task_id] = solved

                if all(solved_tasks):
                    print(f"pearl solved all tasks (but not necessarily in a row)!!")
                    break
                print("evaluation over\n")

        print("pearl training is over\n following tasks have been solved\n")
        print(f"{['solved task: ' + str(s) for s, i in enumerate(solved_tasks) if i]}")
        with open(f"{experiment_path}/solved_env.txt", "a") as f:
            f.write(
                f"pearl has solved the following tasks\n"
                f" {['solved task: ' + str(s) for s, i in enumerate(solved_tasks) if i]}")

    # one path length is between 80 - 200 steps.
    def roll_out(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):

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
            for step in range(self.general_training_args["max_path_length"]):  # max path length=1000 >> num_samples=100
                a = self.agent.action(o, addNoise=True)
                next_o, r, d, _, _ = env.step(a)
                observations.append(o)
                rewards.append(r)
                terminals.append(d)
                actions.append(a)
                o = next_o
                self.episode_reward += r
                total_num_samples += 1

                # in baseline the training happens at this point
                # for that we would have to add the above observations etc. to the replay buffer
                # we could add the transitions directly to the buffer instead of collecting paths and adding
                # the paths
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
