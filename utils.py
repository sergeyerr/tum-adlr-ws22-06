import numpy as np
from gymnasium.wrappers import RecordVideo
import os
import gymnasium as gym
import torch as T
import wandb
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


def validate_task(agent, validation_args, experiment_path, episode, in_eval_task, task_id, pearl=False):

    '''
    doing all the validation stuff + logging
    returns, whether the env is solved
    '''
    stop_reward = []

    gravity, enable_wind, wind_power, turbulence_power = in_eval_task.gravity, in_eval_task.enable_wind,\
        in_eval_task.wind_power, in_eval_task.turbulence_power

    evaluation_episodes = range(validation_args["eval_eps"])
    record_video_on_eval = validation_args["record_video_on_eval"]
    log_actions = validation_args["log_actions"]
    validation_episode_length = validation_args["validation_episode_length"]
    eval_stop_condition = validation_args["eval_stop_condition"]
    validation_traj_num = validation_args["validation_traj_num"]

    for evaluation_episode in evaluation_episodes:

        # log step-action-reward plot for each validation episode
        if log_actions:
            steps = []
            # first action, second action, reward
            actions_main = []
            actions_left_right = []
            #rewards_steps = []

        # each episode clear the memory
        if pearl:
            agent.clear_z()

        for traj in range(validation_traj_num):
            # sum of rewards collected in the trajectory
            rewards = 0
            # use experiment_path folder

            # TODO think about how the validation stop condition should be calculated:
            # instead of using traj=0 record traj=validation_traj_num-1. This makes sense, because pearl is
            #  uninformed in the first trajectory, but will have context starting from the 2nd trajectory.
            #  When using the 'min' stop condition pearl will also not perform as good, because the first uninformed
            #  trajectory will not have as high rewards as the upcoming informed trajectories

            # how to calculate the stop condition is a critical decision. If we decide to take min, then the uninformed
            # pearl trajectory will be the one deciding whether pearl solved the eval task. The uninformed trajectory,
            # however, is not representative of pearls performance.
            # if we take max, it might also not be representative. If we take average than three eval traj is too few,
            # as the first uninformed trajectory would need to have a reward way above 150 points in order to make for
            # the average reward to be larger than 240. This is quite unrealistic.
            # I think the best way would be to take the min between the two informed trajectories (or the average)

            # another idea would be to actually store the context of the eval tasks so that next time we do not need
            # an exploration trajectory

            if record_video_on_eval and evaluation_episode == 0 and traj == validation_traj_num-1:
                # create tmp env with videos
                video_path = os.path.join(experiment_path, "videos", str(episode), str(task_id))
                eval_task = RecordVideo(in_eval_task, video_path)
                with open(f"{video_path}/env_params.txt", "w") as f:
                    f.write(
                        f"gravity: {gravity}\n enable_wind: {enable_wind}\n, wind_power: {wind_power}\n, turbulence_power: {turbulence_power}\n")
            else:
                eval_task = in_eval_task

            if pearl and traj >= validation_args["num_exp_traj_eval"]:
                agent.infer_posterior(agent.context)

            obs, info = eval_task.reset()

            for step in range(validation_episode_length):

                # Get deterministic action
                with T.no_grad():
                    action = agent.action(obs, addNoise=False)

                # Take step in environment
                new_obs, reward, done, _, _ = eval_task.step(action)

                if pearl:
                    agent.update_context([obs, action, reward, new_obs])

                # Update obs
                obs = new_obs

                # Update rewards
                rewards += reward

                # if log_actions:
                #     steps.append(step)
                #     actions_main.append(action[0])
                #     actions_left_right.append(action[1])
                    #rewards_steps.append(reward)

                # End episode if done
                if done:
                    break

            # each entry in stop_reward is the total reward per trajectory
            # want to finish the evaluation once the average stop_reward over all trajectories is above threshold
            stop_reward.append(rewards)

        # seems to save only the last plot
        # if log_actions and evaluation_episode == 0:
        #     wandb.log({"Validation after episode": episode,
        #                "Gravity": gravity,
        #                "Wind": enable_wind,
        #                "Wind power": wind_power,
        #                "Turbulence power": turbulence_power,
        #                "Action plot":  wandb.plot.line_series(xs=steps, ys=[actions_main, actions_left_right], keys=["Main engine", "left/right engine"], xname="step")})

        # if record_video_on_eval and evaluation_episode == 0:
        #     wandb.log({"Validation after episode": episode,
        #                "Gravity": gravity,
        #                "Wind": enable_wind,
        #                "Wind power": wind_power,
        #                "Turbulence power": turbulence_power,
        #                 "Video": wandb.Video(os.path.join(video_path, "rl-video-episode-0.mp4"), fps=4, format="gif",
        #                                      caption=f"gravity: {gravity}, wind: {enable_wind}, wind power: {wind_power}, turbulence power: {turbulence_power}, episode: {episode}")})

        if pearl:
            print(f"stop reward of exploration traj: {stop_reward[0]}\n"
                  f"stop reward of 1st informed traj: {stop_reward[1]}\n"
                  f"stop reward of 2nd informed traj: {stop_reward[2]}")
            stop_reward = stop_reward[1:]  # this is done so that we do not count the exploration trajectory

        avg_reward = round(sum(stop_reward) / len(stop_reward), 3)
        min_reward = round(min(stop_reward), 3)
        max_reward = round(max(stop_reward), 3)
        
        wandb.log({"Validation episode": episode,  "Average evaluation reward": avg_reward, "Min evaluation reward": min_reward, "Max evaluation reward": max_reward, "Gravity": gravity, "Wind": enable_wind, "Wind power": wind_power, "Turbulence power": turbulence_power, "task_id": task_id})

        if eval_stop_condition == "avg":
            stop_reward = avg_reward
        elif eval_stop_condition == "min":
            stop_reward = min_reward
        elif eval_stop_condition == "max":
            stop_reward = max_reward
        else:
            raise ValueError(f"Unknown eval_stop_condition {eval_stop_condition}")



        print(f"Episode: {episode} | Average evaluation reward: {avg_reward} | Min evaluation reward: {min_reward}"
              f" | Max evaluation reward: {max_reward}")

        wandb.log({"Validation after episode": episode,  "Average evaluation reward": avg_reward,
                   "Min evaluation reward": min_reward, "Max evaluation reward": max_reward})

        with open(f"{experiment_path}/evaluation_rewards.csv", "a") as f:
            f.write(f"{episode}, {stop_reward}\n")
        try:
            if stop_reward > eval_task.spec.reward_threshold * 1.1:  # x 1.1 because of small eval_episodes
                # print(f"Environment solved after {episode} episodes")
                return True
        except Exception as e:
            if stop_reward > -120:
                # print(f"Environment solved after {episode} episodes")
                print(f"Exception handled. Stop reward is: {stop_reward}")
                return True

    return False


def get_agent_from_run_cfg(run_cfg):
    agent_args = run_cfg['agent']
    training_args = run_cfg['training']
    env_args = run_cfg['env']
    if agent_args['name'] == "ddpg":
        algorithm = DDPGAgent
    elif agent_args['name'] == "sac":
        algorithm = SACAgent
    elif agent_args['name'] == "sac2":
        algorithm = SACAgent2
    env = LunarEnvFixedFabric(env_params=env_args, pass_env_params=training_args['pass_env_parameters']).generate_env()
    env_info = {"input_dims":env.observation_space.shape, "n_actions": env.action_space.shape[0], "max_action": env.action_space.high}
    agent = algorithm(**agent_args, **training_args, **env_info)
    return agent


def load_best_model(agent, run):
    model_artifacts = [ x for x in  run.logged_artifacts() if x.type == "model" ]
    hist = run.history(keys=["Validation after episode", "Average evaluation reward"], pandas=False)
    
    ind = np.argmax([x["Average evaluation reward"] for x in hist])
    best_model_path = model_artifacts[ind].download()
    agent.load_agent(best_model_path)
    print(f"Best model loaded from {best_model_path} with reward {hist[ind]['Average evaluation reward']}")
    return agent
