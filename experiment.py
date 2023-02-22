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

from utils import print_run_info, validate_task
from pearl_train import PEARLExperiment
from pearl_2_train import PEARLE2xperiment
from baseline_train import BaselineExperiment


# TODO increase max path length with each episode? It seems like pearl takes to long to solve a task. The trajectory
#  is simply cut off because it is longer than max path length (but we do not give negative reward). So the agent
#  is maybe learning to take a long time to land... maybe we should either prolong the path length to make pearl land
#  and receive a higher reward, or give pearl a large negative reward when it does not finish the trajectory

# TODO maybe increase the number of training episodes.

# TODO investigate what impact the embedding_batch_size of pearl has? I used 64 for each experiment. That means that
#  we use 64 transitions to infer the context to inform the policy. Can we abstract enough information from 64
#  transitions? What if we use a larger number, maybe the agent will understand more causalities?

# TODO use different seed for deterministic reset of the environment.

# TODO understand what impact it has how we gather data. We could gather only trajectories with
#  z~prior or z~posterior or z~posterior while not adding new data to the encoder or different com-
#  binations. What impact does it have to clear the encoder or not clearing it buffer? This could
#  be an experiment of its own, just to better understand how pearl behaves.

# TODO check if the exploration trajectory in the evaluation of pearl has lower reward than other two trajectories

# TODO try to use SAC2 inside PEARL



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

    # creates list of envs with different parameterizations
    # train_tasks, train_tasks_with_params, train_tasks_ood, train_tasks_odd_with_params
    train_tasks_array = create_sets_of_tasks(train_env_fabric, training_args.n_train_tasks)
    eval_tasks_array = create_sets_of_tasks(test_env_fabric, validation_args.n_eval_tasks)

    config_dict = OmegaConf.to_object(cfg)

    # SAC experiment
    #config_dict["agent"]["name"] = "sac"
    if config_dict['agent']['name'] == 'pearl2':
        experiment = PEARLE2xperiment(config_dict, train_tasks_array, eval_tasks_array, experiment_path)
    elif config_dict['agent']['name'] == 'pearl':
        experiment = PEARLExperiment(config_dict, train_tasks_array, eval_tasks_array, experiment_path)
    else:
        experiment = BaselineExperiment(config_dict, train_tasks_array, eval_tasks_array, experiment_path)
        
    experiment.run(init_wandb=validation_args['use_wandb'], ood=training_args['ood'], pass_params=training_args['pass_env_parameters'])
    
    # sac_experiment = BaselineExperiment(config_dict, train_tasks_array, eval_tasks_array, experiment_path)

    # # Pearl experiment
    # pearl_experiment = PEARLExperiment(config_dict, train_tasks_array, eval_tasks_array, experiment_path)

    # # SAC2 experiment
    # #config_dict["agent"]["name"] = "sac2"
    # sac2_experiment = BaselineExperiment(config_dict, train_tasks_array, eval_tasks_array, experiment_path)

    # outside distribution______________________________________________________

    # # informed sac2 ood
    # sac2_experiment.run(init_wandb=False, ood=True, pass_params=True)

    # # sac2 ood
    # sac2_experiment.run(init_wandb=False, ood=True, pass_params=False)

    # # informed sac ood
    # sac_experiment.run(init_wandb=False, ood=True, pass_params=True)

    # # sac ood
    # sac_experiment.run(init_wandb=False, ood=True, pass_params=False)

    # # pearl ood
    # pearl_experiment.run(init_wandb=False, ood=True)

    # # inside distribution____________________________________________________

    # # pearl
    # pearl_experiment.run(init_wandb=False, ood=False)

    # # informed sac2
    # sac2_experiment.run(init_wandb=False, ood=False, pass_params=True)

    # # sac2
    # sac2_experiment.run(init_wandb=False, ood=False, pass_params=False)

    # # informed sac
    # sac_experiment.run(init_wandb=False, ood=False, pass_params=True)

    # # sac
    # sac_experiment.run(init_wandb=False, ood=False, pass_params=False)

    # TODO have a third set of tasks for final validation.


def create_sets_of_tasks(env_fabric, num_tasks):
    envs = []
    envs_with_params = []
    ood_envs = []
    ood_wp_envs = []
    for t in range(int(num_tasks)):
        env, env_with_params, ood_env, ood_wp_env = env_fabric.generate_env()
        envs.append(env)
        ood_envs.append(ood_env)
        envs_with_params.append(env_with_params)
        ood_wp_envs.append(ood_wp_env)
    return envs, envs_with_params, ood_envs, ood_wp_envs

if __name__=='__main__':
    experiment()
