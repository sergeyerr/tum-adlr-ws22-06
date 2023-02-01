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
from pearl_train import PEARLExperiment
from baseline_train import BaselineExperiment

# TODO write a parser that takes as input a txt file with specifications on the experiment like:
#  pearl ood, sac ood, sac pass_param ood, sac, pearl, sac pass_param
#  and translates that into an experiment

# TODO to accelerate the experiment we can consider stopping the training if the average reward on the training data
#  and/or evaluation data is not decreasing for N consecutive episodes

# TODO increase max path length with each episode? It seems like pearl takes to long to solve a task. The trajectory
#  is simply cut off because it is longer than max path length (but we do not give negative reward). So the agent
#  is maybe learning to take a long time to land... maybe we should either prolong the path length to make pearl land
#  and receive a higher reward, or give pearl a large negative reward when it does not finish the trajectory

# TODO maybe increase the number of training episodes.

# TODO investigate what impact the embedding_batch_size of pearl has? I used 64 for each experiment. That means that
#  we use 64 transitions to infer the context to inform the policy. Can we abstract enough information from 64
#  transitions? What if we use a larger number, maybe the agent will understand more causalities?

# TODO use different seed for deterministic reset of the environment.

@hydra.main(version_base=None, config_path="conf", config_name="config")
def experiment(cfg: DictConfig):
    env_args = cfg.env
    validation_args = cfg.validation
    training_args = cfg.training

    # Experiment directory storage
    env_path = os.path.join("experiments", "LunarLanderContinuous-v2")
    if not os.path.exists(env_path):
        os.makedirs(env_path)

    # ugly, but acceptable
    experiment_counter = 0
    while True:
        try:
            experiment_path = os.path.join(env_path, f"experiment_{experiment_counter}")
            os.mkdir(experiment_path)
            break
        except FileExistsError as e:
            experiment_counter += 1

    with open(os.path.join(experiment_path, 'parameters.json'), 'w') as f:
        OmegaConf.save(OmegaConf.to_object(cfg), f)
    # TODO remove the pass_env_params argument from LunarEnv... we do not need anymore
    train_env_fabric = LunarEnvRandomFabric(pass_env_params=training_args.pass_env_parameters, **env_args)

    if validation_args.hypercube_validation:
        test_env_fabric = LunarEnvHypercubeFabric(pass_env_params=training_args.pass_env_parameters,
                                                       render_mode='rgb_array',
                                                       points_per_axis=validation_args[
                                                           "hypercube_points_per_axis"], **env_args)
        validation_args.n_eval_tasks = test_env_fabric.number_of_test_points()
        print(f"number of evaluation tasks is: {validation_args.n_eval_tasks}")
    else:
        test_env_fabric = LunarEnvRandomFabric(env_params=env_args,
                                               pass_env_params=training_args.pass_env_parameters,
                                               render_mode='rgb_array')

    # creates list of env with different parameterizations
    train_tasks, train_tasks_with_params, train_tasks_ood = create_tasks(train_env_fabric, training_args.n_train_tasks)
    eval_tasks, eval_tasks_with_params, eval_task_ood = create_tasks(test_env_fabric, validation_args.n_eval_tasks)

    config_dict = OmegaConf.to_object(cfg)

    # Out of distribution testing:
    # the training tasks will be in a different part of the environment parameter space compared to the eval tasks

    # pearl ood
    config_dict["agent"]["name"] = "pearl"
    pearl_experiment = PEARLExperiment(config_dict, train_tasks_ood, eval_task_ood)
    pearl_experiment.run(experiment_path, init_wandb=False, ood=True)

    # sac ood
    config_dict["agent"]["name"] = "sac"
    sac_experiment = BaselineExperiment(config_dict, train_tasks_ood, eval_task_ood)
    sac_experiment.run(experiment_path, init_wandb=False, ood=True, pass_params=False)

    # inside distribution

    # pearl
    config_dict["agent"]["name"] = "pearl"
    pearl_experiment = PEARLExperiment(config_dict, train_tasks, eval_tasks)
    pearl_experiment.run(experiment_path, init_wandb=False, ood=False)

    # sac with environment parameters
    config_dict["agent"]["name"] = "sac"
    sac_experiment = BaselineExperiment(config_dict, train_tasks_with_params, eval_tasks_with_params)
    sac_experiment.run(experiment_path, init_wandb=False, ood=False, pass_params=True)

    # sac
    config_dict["agent"]["name"] = "sac"
    sac_experiment = BaselineExperiment(config_dict, train_tasks, eval_tasks)
    sac_experiment.run(experiment_path, init_wandb=False, ood=False, pass_params=False)

    # TODO have a third set of tasks for final validation.


def create_tasks(env_fabric, num_tasks):
    envs = []
    envs_with_params = []
    ood_envs = []
    for t in range(int(num_tasks)):
        env, env_with_params, ood_env = env_fabric.generate_env()
        envs.append(env)
        ood_envs.append(ood_env)
        envs_with_params.append(env_with_params)
    return envs, envs_with_params, ood_envs

if __name__=='__main__':
    experiment()
