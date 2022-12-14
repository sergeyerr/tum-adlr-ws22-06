import argparse
import json
import os
import random
from collections import deque
import copy

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch as T
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from EnvironmentUtils import StateInjectorWrapper, LunarEnvRandomFabric, LunarEnvFixedFabric, LunarEnvHypercubeFabric

import wandb

from sweep_utils import train_sweep

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
global_cfg = None

def train_():
    global global_cfg
    run = wandb.init()
    cfg = copy.deepcopy(global_cfg)
    #print(wandb.config)
    # bad
    cfg.agent.update(wandb.config['agent_sweep'])
    cfg.training.update(wandb.config['training_sweep'])
    #cfg.agent.pi_lr = wandb.config.lr
   # wandb.config = OmegaConf.to_object(cfg)
    #print(wandb.config)
    #wandb.log({"Average evaluation reward" : 1})
    
    wandb.config.update(OmegaConf.to_object(cfg))
    train_sweep(cfg)
    
@hydra.main(version_base=None, config_path="conf", config_name="config_sweep") 
def agent_proxy(cfg : DictConfig):
    global global_cfg
    global_cfg = cfg
    #wandb.init(project="adlr_sweeps", entity="tum-adlr-ws22-06")
    wandb.login()
    wandb.agent(cfg.sweep.sweep_id, function=train_, count=cfg.sweep.count)
    
    
if __name__=='__main__':
    agent_proxy()