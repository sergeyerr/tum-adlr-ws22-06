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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def experiment(cfg: DictConfig):
    agent_args = cfg.agent
    env_args = cfg.env
    validation_args = cfg.validation
    training_args = cfg.training

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
    train_tasks, train_tasks_with_params = create_train_tasks(train_env_fabric, training_args.n_train_tasks)
    eval_tasks, eval_tasks_with_params = create_train_tasks(test_env_fabric, validation_args.n_eval_tasks)

    config_dict = OmegaConf.to_object(cfg)
    # pearl
    #config_dict["agent"]["name"] = "pearl"
    #pearl_experiment = PEARLExperiment(config_dict, train_tasks, eval_tasks)
    #pearl_experiment.run()
    # sac
    config_dict["agent"]["name"] = "sac"
    sac_experiment = BaselineExperiment(config_dict, train_tasks, eval_tasks)
    sac_experiment.run()
    # sac with environment parameters
    config_dict["agent"]["name"] = "sac"
    sac_experiment = BaselineExperiment(config_dict, train_tasks_with_params, eval_tasks_with_params)
    sac_experiment.run()

def create_train_tasks(env_fabric, num_tasks):
    envs = []
    envs_with_params = []
    for t in range(num_tasks):
        env, env_with_params = env_fabric.generate_env()
        envs.append(env)
        envs_with_params.append(env_with_params)
    return envs, envs_with_params

if __name__=='__main__':
    experiment()
