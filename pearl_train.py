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
from EnvironmentRandomizer import StateInjectorWrapper, LunarEnvFabric
from DataHandling.EnvironmentSampler import Sampler

import wandb
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ZeroNoise
from agents import PEARLAgent

from utils import print_run_info, validate

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class PEARLExperiment(object):

    def __init__(self, cfg: DictConfig):
        self.agent_args = cfg.agent
        self.env_args = cfg.env
        self.validation_args = cfg.validation
        self.training_args = cfg.training


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    agent_args = cfg.agent
    env_args = cfg.env
    validation_args = cfg.validation
    training_args = cfg.training

    experiment_name = agent_args.experiment_name

    # env = LunarRandomizerWrapper(pass_env_params=training_args.pass_env_parameters, **env_args)
    train_env_fabric = LunarEnvFabric(pass_env_params=training_args.pass_env_parameters, **env_args)
    test_env_fabric = LunarEnvFabric(pass_env_params=training_args.pass_env_parameters, render_mode='rgb_array',
                                     **env_args)
    env = train_env_fabric.generate_env()

    T.manual_seed(training_args.seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    # env.seed(args.seed)

    # Weights and biases initialization
    wandb.init(project="Model tests on the non-randomized env", entity="tum-adlr-ws22-06",
               config=OmegaConf.to_object(cfg))

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
        OmegaConf.save(cfg, f)

    n_actions = env.action_space.shape[0] if type(env.action_space) == gym.spaces.box.Box else env.action_space.n
    env_info = {"input_dims": env.observation_space.shape, "n_actions": n_actions, "max_action": env.action_space.high}

    algorithm = PEARLAgent

    agent = algorithm(**OmegaConf.to_object(agent_args), **OmegaConf.to_object(training_args),
                      **env_info)

    if validation_args.log_model_wandb:
        # assumes that the model has only one actor, we may also log different models differently
        wandb.watch(agent.pi, log="all", log_freq=validation_args.log_model_every_training_batch)
        print(
            f"================= {f'Sending weights to W&B every {validation_args.log_model_every_training_batch} batch'} =================")

    # temporary variant, possible problems with SAC
    if training_args.noise == "normal":
        noise = NormalActionNoise(mean=0, sigma=training_args.noise_param, size=n_actions)
    elif training_args.noise == "Ornstein":
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), sigma=training_args.noise_param)
    else:
        noise = ZeroNoise(size=n_actions)

    print_run_info(env, agent, agent_args, training_args, env_args, validation_args, noise)

    reward_history = deque(maxlen=100)

    t = 0

    # creates list of env with different parametrizations
    train_tasks = create_train_tasks(train_env_fabric)

    '''meta-training loop'''
    # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
    for episode in range(training_args.episodes):

        if episode == 0:
            print('collecting initial pool of data for train and eval')
            for env in train_tasks:
                env.reset()
                collect_data(self.num_initial_steps, 1, np.inf)
        # Sample data from train tasks.
        for i in range(self.num_tasks_sample):
            idx = np.random.randint(len(self.train_tasks))
            self.task_idx = idx
            self.env.reset_task(idx)
            self.enc_replay_buffer.task_buffers[idx].clear()

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
        for train_step in range(self.num_train_steps_per_itr):
            indices = np.random.choice(self.train_tasks, self.meta_batch)
            self._do_training(indices)
            self._n_train_steps_total += 1

    mb_size = self.embedding_mini_batch_size
    num_updates = self.embedding_batch_size // mb_size

    # sample context batch
    context_batch = self.sample_context(indices)

    # zero out context and hidden encoder state
    self.agent.clear_z(num_tasks=len(indices))

    # do this in a loop so we can truncate backprop in the recurrent encoder
    for i in range(num_updates):
        context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
        self._take_step(indices, context)

        # stop backprop
        self.agent.detach_z()

def create_train_tasks(train_env_fabric, num_tasks):
    envs = []
    for i in range(num_tasks):
        envs.append(train_env_fabric.generate_env())
    return envs

def collect_data(self, sampler, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
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
    self.agent.clear_z()

    num_transitions = 0
    while num_transitions < num_samples:
        paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                       max_trajs=update_posterior_rate,
                                                       accum_context=False,
                                                       resample=resample_z_rate)
        num_transitions += n_samples
        self.replay_buffer.add_paths(self.task_idx, paths)
        if add_to_enc_buffer:
            self.enc_replay_buffer.add_paths(self.task_idx, paths)
        if update_posterior_rate != np.inf:
            context = self.sample_context(self.task_idx)
            self.agent.infer_posterior(context)
    self._n_env_steps_total += num_transitions
    gt.stamp('sample')

if __name__ == '__main__':
    train()
