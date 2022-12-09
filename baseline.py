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
from EnvironmentRandomizer import StateInjectorWrapper, LunarEnvRandomFabric, LunarEnvFixedFabric, LunarEnvHypercubeFabric

import wandb
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ZeroNoise
from agents import DDPGAgent, SACAgent, SACAgent2

from utils import print_run_info, validate

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig):
    agent_args = cfg.agent
    env_args = cfg.env
    validation_args = cfg.validation
    training_args = cfg.training
    

    experiment_name = agent_args.experiment_name
    
   # env = LunarRandomizerWrapper(pass_env_params=training_args.pass_env_parameters, **env_args)
    if env_args.random:
        train_env_fabric = LunarEnvRandomFabric(env_params=env_args, pass_env_params=training_args.pass_env_parameters)
        if validation_args.hypercube_validation:
            test_env_fabric = LunarEnvHypercubeFabric(env_params=env_args, pass_env_params=training_args.pass_env_parameters,  
                                                      render_mode= 'rgb_array', points_per_axis=validation_args.hypercube_points_per_axis)
            validation_args.eval_eps = test_env_fabric.number_of_test_points()
        else:
            test_env_fabric = LunarEnvRandomFabric(env_params=env_args, pass_env_params=training_args.pass_env_parameters,
                                                   render_mode= 'rgb_array')
    else:
        train_env_fabric = LunarEnvFixedFabric(env_params=env_args, pass_env_params=training_args.pass_env_parameters)
        test_env_fabric = LunarEnvFixedFabric(env_params=env_args, pass_env_params=training_args.pass_env_parameters,  render_mode= 'rgb_array')
    env = train_env_fabric.generate_env()

    T.manual_seed(training_args.seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    #env.seed(args.seed)
    
    # Weights and biases initialization
    wandb.init(project="ADLR randomized envs", entity="tum-adlr-ws22-06", config=OmegaConf.to_object(cfg))


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
        OmegaConf.save(cfg, f)

    n_actions = env.action_space.shape[0] if type(env.action_space) == gym.spaces.box.Box else env.action_space.n
    env_info = {"input_dims":env.observation_space.shape, "n_actions": n_actions, "max_action": env.action_space.high}

    # TODO: Modify this to call any other algorithm
    if agent_args.name == "ddpg":
        algorithm = DDPGAgent
    elif agent_args.name == "sac":
        algorithm = SACAgent
    elif agent_args.name == "sac2":
        algorithm = SACAgent2
    

    agent = algorithm(**OmegaConf.to_object(agent_args), **OmegaConf.to_object(training_args),
                      **env_info)
    
    if validation_args.log_model_wandb:
        # assumes that the model has only one actor, we may also log different models differently
        wandb.watch(agent.pi, log="all", log_freq=validation_args.log_model_every_training_batch)
        print(f"================= {f'Sending weights to W&B every {validation_args.log_model_every_training_batch} batch'} =================")
        

    #temporary variant, possible problems with SAC
    if training_args.noise == "normal":
        noise = NormalActionNoise(mean=0, sigma=training_args.noise_param, size=n_actions)
    elif training_args.noise == "Ornstein":
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), sigma=training_args.noise_param)
    else:
        noise = ZeroNoise(size=n_actions)
        
    print_run_info(env, agent, agent_args, training_args, env_args, validation_args, noise)


    counter = 0
    reward_history = deque(maxlen=100)
    
    t = 0
    
    for episode in range(training_args.episodes):
        env = train_env_fabric.generate_env()
        obs, info = env.reset()
        gravity, enable_wind, wind_power, turbulence_power = env.gravity, env.enable_wind, env.wind_power, env.turbulence_power
        if training_args.noise != "Zero":
            noise.reset()
        episode_reward = 0.0
        actor_loss = 0.0
        critic_loss = 0.0

        # Generate rollout
        for step in range(training_args.episode_length):

            # Get actions
            # why no_grad()?
            with T.no_grad():
                if t >= training_args.start_steps:
                    action = agent.action(obs, addNoise=True, noise=noise)
                    #action = env.action_space.sample()
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
        
            if t >= training_args.update_after and t % training_args.update_every == 0 and agent.replay_buffer.size() > agent.min_replay_size: 
                for i in range(training_args.train_batches):
                    loss = agent.train()
                    # Loss information kept for monitoring purposes during training
                    actor_loss += loss['actor_loss']
                    critic_loss += loss['critic_loss']
                    wandb.log({"Training episode": episode, "Batch": (episode) * training_args.train_batches + i,
                            "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})
                agent.update()
                
            t += 1
            # End episode if done
            if done:
                break
        
               
        reward_history.append(episode_reward)
        print(f"Training episode: {episode} Episode reward: {episode_reward} Average reward: {np.mean(reward_history)}")
        print(f"Gravity: {gravity} Wind: {enable_wind} Wind power: {wind_power} Turbulence power: {turbulence_power}")
        wandb.log({"Training episode": episode, "Episode reward": episode_reward, "Average reward": np.mean(reward_history), 
                   "Gravity": gravity, "Wind": enable_wind, "Wind power": wind_power, "Turbulence power": turbulence_power})
        if episode % validation_args.eval_interval == 0:
            solved = validate(agent, validation_args, experiment_path, episode, test_env_fabric)
            if solved:
                break
                    
                    
if __name__=='__main__':
    train()