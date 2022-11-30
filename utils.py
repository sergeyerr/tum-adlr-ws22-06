import numpy as np
from gymnasium.wrappers import RecordVideo
import os
import gymnasium as gym
import torch as T
import wandb


def print_run_info(env, agent, agent_args, training_args, env_args, validation_args, noise):
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
        
    print(f"================= {'Noise Information'.center(30)} =================")
    print(noise)
    
    print(f"================= {'Agent Information'.center(30)} =================")
    print(agent)
    
    print(f"================= {'Begin Training'.center(30)} =================")
    
    
def validate(agent, validation_args, experiment_path, episode):
    '''
    doing all the validation stuff + logging
    returns, whether the env is solved
    '''
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
            return True
    except Exception as e:
        if evaluation_rewards > -120:
            print(f"Environment solved after {episode} episodes")
            return True
    return False