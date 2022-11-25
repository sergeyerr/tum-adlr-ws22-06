import gymnasium as gym
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch as T
from RLAgent import DDPGAgent, SACAgent, SACAgent2
from Noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise, ZeroNoise
import os
import argparse
import json
import random
from collections import deque
from gymnasium.wrappers import RecordVideo

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig):
    agent_args = cfg.agent
    env_args = cfg.env
    validation_args = cfg.validation
    training_args = cfg.training
    
    if validation_args.record_video_on_eval:
        from gymnasium.wrappers import RecordVideo

    experiment_name = agent_args.experiment_name

    env = gym.make('LunarLanderContinuous-v2')

    T.manual_seed(training_args.seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)
    #env.seed(args.seed)

    print(f"================= {'Environment Information'.center(30)} =================")
    print(f"Action space shape: {env.env.action_space.shape}")
    print(f"Action space upper bound: {env.env.action_space.high}")
    print(f"Action space lower bound: {env.env.action_space.low}")

    print(f"Observation space shape: {env.env.observation_space.shape}")
    print(f"Observation space upper bound: {np.max(env.env.observation_space.high)}")
    print(f"Observation space lower bound: {np.min(env.env.observation_space.low)}")
    
    print(f"================= {'Parameters'.center(30)} =================")
    print(f"================= {'Agent parameters'.center(30)} =================")
    for k, v in agent_args.items():
        print(f"{k:<20}: {v}")
        
    print(f"================= {'Training parameters'.center(30)} =================")
    for k, v in training_args.items():
        print(f"{k:<20}: {v}")
        
    print(f"================= {'Environment parameters'.center(30)} =================")
    for k, v in env_args.items():
        print(f"{k:<20}: {v}")
        
    print(f"================= {'Validation parameters'.center(30)} =================")
    for k, v in validation_args.items():
        print(f"{k:<20}: {v}")

    # Experiment directory storage
    counter = 1
    env_path = os.path.join("experiments", env_args.env)
    if not os.path.exists(env_path):
        os.makedirs(env_path)

    while True:
        try:
            experiment_path = os.path.join(env_path, f"{experiment_name}_{counter}")
            os.mkdir(experiment_path)
            os.mkdir(os.path.join(experiment_path, "saves"))
            break
        except FileExistsError as e:
            counter += 1

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

    print(f"================= {'Noise Information'.center(30)} =================")
    #temporary variant, possible problems with SAC
    if env_args.noise:
        if env_args.gaussian_noise:
            noise = NormalActionNoise(mean=0, sigma=env_args.noise_param, size=n_actions)
            print(noise)
        else:
            noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), sigma=env_args.noise_param)
    else:
        noise = ZeroNoise(size=n_actions)
    print(noise)

    print(f"================= {'Agent Information'.center(30)} =================")
    print(agent)

    print(f"================= {'Begin Training'.center(30)} =================")

    counter = 0
    reward_history = deque(maxlen=100)

    for episode in range(training_args.episodes):
        obs, info = env.reset()
        noise.reset()
        episode_reward = 0.0
        actor_loss = 0.0
        critic_loss = 0.0

        # Generate rollout and train agent
        for step in range(training_args.episode_length):

            # delete it possible
            if validation_args.render:
                env.render()

            # Get actions
            with T.no_grad():
                if episode >= training_args.exploration:
                    action = agent.action(obs, addNoise=True, noise=noise)
                else:
                    action = env.action_space.sample()

            # TODO Ignore the "done" signal if it comes from hitting the time horizon.

            # Take step in environment
            new_obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

            # Store experience
            agent.experience(obs, action, reward, new_obs, done)

            # Train agent
            if counter > training_args.train_after and counter % training_args.train_interval == 0:
                if agent.replay_buffer.size() > agent.min_replay_size:
                    counter = 0
                    loss = agent.train()

                    # Loss information kept for monitoring purposes during training
                    actor_loss += loss['actor_loss']
                    critic_loss += loss['critic_loss']

                    agent.update()

            # Update obs
            obs = new_obs

            # Update counter
            counter += 1

            # End episode if done
            if done:
                break

        reward_history.append(episode_reward)
        print(f"Episode: {episode} Episode reward: {episode_reward} Average reward: {np.mean(reward_history)}")
        # print(f"Actor loss: {actor_loss/(step/args.train_interval)} Critic loss: {critic_loss/(step/args.train_interval)}")

        # Evaluate
        if episode % validation_args.eval_interval == 0:
            evaluation_rewards = 0
            for evaluation_episode in range(validation_args.eval_eps):
                # TODO: specify occurencies of vids (hydra, conditional parameter)
                # use experiment_path folder
                if counter > training_args.train_after and validation_args.record_video_on_eval and evaluation_episode == 0:
                                        # create tmp env with videos
                    video_path = os.path.join(experiment_path, "videos", str(episode))
                    test_env = RecordVideo(gym.make('LunarLanderContinuous-v2', render_mode='rgb_array'), video_path)
                else:
                    test_env = gym.make('LunarLanderContinuous-v2')
                obs, info = test_env.reset()
                rewards = 0

                for step in range(validation_args.validation_episode_length):
                    # !!! careful with video recording, possibly delete it 
                    if validation_args.render:
                        test_env.render()

                    # Get deterministic action
                    with T.no_grad():
                        action = agent.action(obs, addNoise=False)

                    # Take step in environment
                    new_obs, reward, done, _, _ = test_env.step(action)

                    # Update obs
                    obs = new_obs

                    # Update rewards
                    rewards += reward

                    # End episode if done
                    if done:
                        break

                evaluation_rewards += rewards

            evaluation_rewards = round(evaluation_rewards / validation_args.eval_eps, 3)
            save_path = os.path.join(experiment_path, "saves")
            
            agent.save_agent(save_path)
            print(f"Episode: {episode} Average evaluation reward: {evaluation_rewards} Agent saved at {save_path}")
            with open(f"{experiment_path}/evaluation_rewards.csv", "a") as f:
                f.write(f"{episode}, {evaluation_rewards}\n")
            try:
                if evaluation_rewards > test_env.spec.reward_threshold * 1.1:  # x 1.1 because of small eval_episodes
                    print(f"Environment solved after {episode} episodes")
                    break
            except Exception as e:
                if evaluation_rewards > -120:
                    print(f"Environment solved after {episode} episodes")
                    break
                    
                    
if __name__=='__main__':
    train()