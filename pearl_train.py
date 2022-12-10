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
from EnvironmentUtils import LunarEnvRandomFabric, LunarEnvFixedFabric, LunarEnvHypercubeFabric
from DataHandling.EnvironmentSampler import Sampler

import wandb
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ZeroNoise
from agents import PEARLAgent

from utils import print_run_info, validate

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


@hydra.main(version_base=None, config_path="conf", config_name="config")
class PEARLExperiment(object):

    def __init__(self, cfg: DictConfig):
        self.episode_reward = 0.0
        self.task_idx = 0
        self.cfg = cfg
        self.agent_args = cfg.agent
        self.env_args = cfg.env
        self.validation_args = cfg.validation
        self.training_args = cfg.training
        # env = LunarRandomizerWrapper(pass_env_params=training_args.pass_env_parameters, **env_args)
        self.train_env_fabric = LunarEnvFabric(pass_env_params=self.training_args.pass_env_parameters, **self.env_args)
        self.test_env_fabric = LunarEnvFabric(pass_env_params=self.training_args.pass_env_parameters,
                                              render_mode='rgb_array',
                                              **self.env_args)
        # creates list of env with different parametrizations
        self.train_tasks = self.create_train_tasks(self.train_env_fabric, self.training_args.num_train_tasks)
        self.env = self.train_tasks[0]
        self.n_actions = self.train_tasks[0].action_space.shape[0]\
            if type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n
        env_info = {"input_dims": self.train_tasks[0].observation_space.shape, "n_actions": self.n_actions,
                    "max_action": self.train_tasks[0].action_space.high}

        self.agent = PEARLAgent(**OmegaConf.to_object(self.agent_args), **OmegaConf.to_object(self.training_args),
                                **env_info)
        self.sampler = Sampler

    def run(self):

        experiment_name = self.agent_args.experiment_name
        env = self.train_env_fabric.generate_env()

        T.manual_seed(self.training_args.seed)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        np.random.seed(self.training_args.seed)
        random.seed(self.training_args.seed)
        # env.seed(self.args.seed)

        # Weights and biases initialization
        wandb.init(project="Model tests on the non-randomized env", entity="tum-adlr-ws22-06",
                   config=OmegaConf.to_object(self.cfg))

        # Experiment directory storage
        env_path = os.path.join("experiments", "LunarLanderContinuous-v2")
        if not os.path.exists(env_path):
            os.makedirs(env_path)

        # ugly, but acceptable
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

        if self.validation_args.log_model_wandb:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args.log_model_every_training_batch)
            print(
                f"================= {f'Sending weights to W&B every {self.validation_args.log_model_every_training_batch} batch'} =================")

        # temporary variant, possible problems with SAC
        if self.training_args.noise == "normal":
            noise = NormalActionNoise(mean=0, sigma=self.training_args.noise_param, size=self.n_actions)
        elif self.training_args.noise == "Ornstein":
            noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.n_actions), sigma=self.training_args.noise_param)
        else:
            noise = ZeroNoise(size=self.n_actions)

        print_run_info(env, self.agent.pi, self.agent_args, self.training_args, self.env_args, self.validation_args, noise)

        reward_history = deque(maxlen=100)

        '''meta-training loop'''
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for episode in range(self.training_args.episodes):
            self.episode_reward = 0.0
            actor_loss = 0.0
            critic_loss = 0.0

            if episode == 0:
                print('collecting initial pool of data for train and eval')
                for env in self.train_tasks:
                    self.env = env
                    self.env.reset()
                    self.collect_data(self.training_args.num_initial_steps, 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.training_args.num_tasks_sample):
                idx = np.random.randint(self.training_args.num_train_tasks)
                self.task_idx = idx
                self.env = self.train_tasks[idx]
                self.env.reset()
                self.agent.encoder_replay_buffer.task_buffers[idx].clear_buffer()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.training_args.train_batchesser):
                indices = np.random.choice(self.training_args.num_train_tasks, self.training_args.meta_batch)
                loss = self.agent.optimize(indices)
                # Loss information kept for monitoring purposes during training
                actor_loss += loss['actor_loss']
                critic_loss += loss['critic_loss']
                wandb.log({"Training episode": episode, "Batch": episode * self.training_args.train_batches + train_step,
                           "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})
                self._n_train_steps_total += 1

            reward_history.append(self.episode_reward)

            if episode % self.validation_args.eval_interval == 0:
                solved = self.validate(episode)
                if solved:
                    # return true, that agent solved environment
                    return True
        # agent did not solve environment
        return False

    def create_train_tasks(self, train_env_fabric, num_tasks):
        envs = []
        for i in range(num_tasks):
            envs.append(train_env_fabric.generate_env())
        return envs

    # generates rollout
    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples
        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.pi.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.agent.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.agent.encoder_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.agent.sample_context(self.task_idx)
                self.agent.pi.infer_posterior(context)
        self._n_env_steps_total += num_transitions

if __name__ == '__main__':
    exp = PEARLExperiment()
    exp.run()
