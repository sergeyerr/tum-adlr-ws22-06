import gymnasium as gym
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch as T
from DDPGAgent import DDPGAgent
from Noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
import os
import argparse
import json
import random
from collections import deque


device = T.device('cuda' if T.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="conf", config_name="ddpg")
def train(cfg : DictConfig):
    args = cfg.ddpg

    experiment_name = args.experiment_name

    env = gym.make('LunarLanderContinuous-v2')

    T.manual_seed(args.seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    #env.seed(args.seed)

    print(f"================= {'Environment Information'.center(30)} =================")
    print(f"Action space shape: {env.env.action_space.shape}")
    print(f"Action space upper bound: {env.env.action_space.high}")
    print(f"Action space lower bound: {env.env.action_space.low}")

    print(f"Observation space shape: {env.env.observation_space.shape}")
    print(f"Observation space upper bound: {np.max(env.env.observation_space.high)}")
    print(f"Observation space lower bound: {np.min(env.env.observation_space.low)}")

    print(f"================= {'Parameters'.center(30)} =================")
    for k, v in args.__dict__.items():
        print(f"{k:<20}: {v}")

    # Experiment directory storage
    counter = 1
    env_path = os.path.join("experiments", args.env)
    if not os.path.exists(env_path):
        os.mkdir(env_path)

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

    # TODO: Modify this to call any other algorithm
    algorithm = DDPGAgent

    agent = algorithm(args.pi_lr, args.q_lr, args.gamma, args.batch_size, args.min_replay_size, args.replay_buffer_size, args.tau,
                      input_dims=env.observation_space.shape,
                      n_actions=n_actions)

    print(f"================= {'Noise Information'.center(30)} =================")
    if args.gaussian_noise:
        noise = NormalActionNoise(mean=0, sigma=args.noise_param, size=n_actions)
        print(noise)
    else:
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), sigma=args.noise_param)
        print(noise)

    print(f"================= {'Agent Information'.center(30)} =================")
    print(agent)

    print(f"================= {'Begin Training'.center(30)} =================")

    counter = 0
    reward_history = deque(maxlen=100)

    for episode in range(args.episodes):
        obs, info = env.reset()
        noise.reset()
        episode_reward = 0.0
        actor_loss = 0.0
        critic_loss = 0.0

        # Generate rollout and train agent
        for step in range(args.episode_length):

            if args.render:
                env.render()

            # Get actions
            with T.no_grad():
                if episode >= args.exploration:
                    action = agent.action(obs) + T.tensor(noise(), dtype=T.float, device=device)
                    action = T.clamp(action, -1.0, 1.0)
                else:
                    action = agent.random_action()

            # Take step in environment
            new_obs, reward, done, _, _ = env.step(action.detach().cpu().numpy() * env.action_space.high)
            episode_reward += reward

            # Store experience
            agent.experience(obs, action.detach().cpu().numpy(), reward, new_obs, done)

            # Train agent
            if counter % args.train_interval == 0:
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
        if episode % args.eval_interval == 0:
            evaluation_rewards = 0
            for evalutaion_episode in range(args.eval_eps):
                obs, info = env.reset()
                rewards = 0

                for step in range(args.episode_length):
                    if args.render:
                        env.render()

                    # Get actions
                    with T.no_grad():
                        action = agent.action(obs)

                    # Take step in environment
                    new_obs, reward, done, _, _ = env.step(action.detach().cpu().numpy()  * env.action_space.high)

                    # Update obs
                    obs = new_obs

                    # Update rewards
                    rewards += reward

                    # End episode if done
                    if done:
                        break

                evaluation_rewards += rewards

            evaluation_rewards = round(evaluation_rewards / args.eval_eps, 3)
            save_path = os.path.join(experiment_path, "saves")
            agent.save_agent(save_path)
            print(f"Episode: {episode} Average evaluation reward: {evaluation_rewards} Agent saved at {save_path}")
            with open(f"{experiment_path}/evaluation_rewards.csv", "a") as f:
                f.write(f"{episode}, {evaluation_rewards}\n")
            try:
                if evaluation_rewards > env.spec.reward_threshold * 1.1: # x 1.1 because of small eval_episodes
                    print(f"Environment solved after {episode} episodes")
                    break
            except Exception as e:
                if evaluation_rewards > -120:
                    print(f"Environment solved after {episode} episodes")
                    break
                    
                    
if __name__=='__main__':
    train()