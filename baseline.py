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
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig, OmegaConf

import wandb
from Noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, ZeroNoise
from agents import DDPGAgent, SACAgent, SACAgent2

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
    
    # Weights and biases initialization
    wandb.init(project="Model tests on the non-randomized env", entity="tum-adlr-ws22-06", config=OmegaConf.to_object(cfg))

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
    env_path = os.path.join("experiments", env_args.env)
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
        

    print(f"================= {'Noise Information'.center(30)} =================")
    #temporary variant, possible problems with SAC
    if training_args.noise == "normal":
        noise = NormalActionNoise(mean=0, sigma=training_args.noise_param, size=n_actions)
    elif training_args.noise == "Ornstein":
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), sigma=training_args.noise_param)
    else:
        noise = ZeroNoise(size=n_actions)
    print(noise)
        

    print(f"================= {'Agent Information'.center(30)} =================")
    print(agent)

    print(f"================= {'Begin Training'.center(30)} =================")

    counter = 0
    reward_history = deque(maxlen=100)
    
    t = 0
    
    for episode in range(training_args.episodes):
        obs, info = env.reset()
        if training_args.noise != "Zero":
            noise.reset()
        episode_reward = 0.0
        actor_loss = 0.0
        critic_loss = 0.0

        # Generate rollout
        for step in range(training_args.episode_length):

            # Get actions
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
        
            
        #if episode < training_args.exploration:
            #print(f"generating episode: {episode}")
        #Train agent (different from  spinning up, I find it more logical to separate sampling and train)
        #if episode >= training_args.exploration and agent.replay_buffer.size() > agent.min_replay_size:
        # if agent.replay_buffer.size() > agent.min_replay_size:    
        #     for i in range(training_args.train_batches):
        #         loss = agent.train()
        #         # Loss information kept for monitoring purposes during training
        #         actor_loss += loss['actor_loss']
        #         critic_loss += loss['critic_loss']
        #         wandb.log({"Training episode": episode, "Batch": (episode) * training_args.train_batches + i,
        #                    "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})
        #     agent.update()
                
        # if we started to train the model:
        #if episode >= training_args.exploration:                
        reward_history.append(episode_reward)
        print(f"Training episode: {episode} Episode reward: {episode_reward} Average reward: {np.mean(reward_history)}")
        wandb.log({"Training episode": episode, "Episode reward": episode_reward, "Average reward": np.mean(reward_history)})
        # print(f"Actor loss: {actor_loss/(step/args.train_interval)} Critic loss: {critic_loss/(step/args.train_interval)}")
        
            # Evaluate
        if episode % validation_args.eval_interval == 0:
            evaluation_rewards = 0
            for evaluation_episode in range(validation_args.eval_eps):
                # TODO: specify occurencies of vids (hydra, conditional parameter)
                # use experiment_path folder
                if  validation_args.record_video_on_eval and evaluation_episode == 0:
                    # create tmp env with videos
                    video_path = os.path.join(experiment_path, "videos", str(episode))
                    test_env = RecordVideo(gym.make('LunarLanderContinuous-v2', render_mode='rgb_array'), video_path)
                else:
                    test_env = gym.make('LunarLanderContinuous-v2')
                    
                # log step-action-reward plot for each validation episode
                if validation_args.log_actions:
                    steps = []
                    # first action, second action, reward
                    actions_main = []
                    actions_left_right = []
                    #rewards_steps = []
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
                    
                    if validation_args.log_actions:
                        steps.append(step)
                        actions_main.append(action[0])
                        actions_left_right.append(action[1])
                        #rewards_steps.append(reward)

                    # End episode if done
                    if done:
                        break

                evaluation_rewards += rewards
                # seems to save only the last plot
                if validation_args.log_actions and evaluation_episode == 0:
                    wandb.log({"Validation after episode": episode, 
                                "Action plot" :  wandb.plot.line_series(xs=steps, ys=[actions_main, actions_left_right], keys=["Main engine", "left/right engine"], xname="step")})
                
                if validation_args.record_video_on_eval and evaluation_episode == 0:
                    wandb.log({"Validation after episode": episode, 
                                "Video" : wandb.Video(os.path.join(video_path, "rl-video-episode-0.mp4"), fps=4, format="gif")})
                
            evaluation_rewards = round(evaluation_rewards / validation_args.eval_eps, 3)
            save_path = os.path.join(experiment_path, "saves")
            
            agent.save_agent(save_path)
            print(f"Episode: {episode} Average evaluation reward: {evaluation_rewards} Agent saved at {save_path}")
            wandb.log({"Validation after episode": episode,  "Average evaluation reward": evaluation_rewards})
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