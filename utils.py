import numpy as np
from gymnasium.wrappers import RecordVideo
import os
import gymnasium as gym
import torch as T
import wandb
from EnvironmentUtils import StateInjectorWrapper


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

    
    
def validate(agent, validation_args, experiment_path, episode, in_eval_task, task_id, pearl=False):
    '''
    doing all the validation stuff + logging
    returns, whether the env is solved
    '''
    stop_reward = []

    gravity, enable_wind, wind_power, turbulence_power = in_eval_task.gravity, in_eval_task.enable_wind,\
        in_eval_task.wind_power, in_eval_task.turbulence_power

    if pearl:
        evaluation_episodes = range(validation_args["eval_eps"])
        record_video_on_eval = validation_args["record_video_on_eval"]
        log_actions = validation_args["log_actions"]
        validation_episode_length = validation_args["validation_episode_length"]
        eval_stop_condition = validation_args["eval_stop_condition"]
        validation_traj_num = validation_args["validation_traj_num"]
    else:
        evaluation_episodes = range(validation_args.eval_eps)
        record_video_on_eval = validation_args.record_video_on_eval
        log_actions = validation_args.log_actions
        validation_episode_length = validation_args.validation_episode_length
        eval_stop_condition = validation_args.eval_stop_condition
        validation_traj_num = validation_args.validation_traj_num

    for evaluation_episode in evaluation_episodes:

        num_traj = 0
        rewards = 0

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
            # use experiment_path folder
            if record_video_on_eval and evaluation_episode == 0 and traj == 0:
                # create tmp env with videos
                video_path = os.path.join(experiment_path, "videos", str(episode), str(task_id))
                eval_task = RecordVideo(in_eval_task, video_path)
                with open(f"{video_path}/env_params.txt", "w") as f:
                    f.write(
                        f"gravity: {gravity}\n enable_wind: {enable_wind}\n, wind_power: {wind_power}\n, turbulence_power: {turbulence_power}\n")
            else:
                eval_task = in_eval_task

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

                if log_actions:
                    steps.append(step)
                    actions_main.append(action[0])
                    actions_left_right.append(action[1])
                    #rewards_steps.append(reward)

                # End episode if done
                if done:
                    break

            # each entry in stop_reward is the total reward per trajectory
            # want to finish the evaluation once the average stop_reward over all trajectories is above threshold
            stop_reward.append(rewards)

            num_traj += 1
            if pearl and num_traj > validation_args["num_exp_traj_eval"]:
                agent.infer_posterior(agent.context)


        # seems to save only the last plot
        if log_actions and evaluation_episode == 0:
            wandb.log({"Validation after episode": episode,
                       "Gravity": gravity,
                       "Wind": enable_wind,
                       "Wind power": wind_power,
                       "Turbulence power": turbulence_power,
                       "Action plot":  wandb.plot.line_series(xs=steps, ys=[actions_main, actions_left_right], keys=["Main engine", "left/right engine"], xname="step")})

        if record_video_on_eval and evaluation_episode == 0:
            wandb.log({"Validation after episode": episode,
                       "Gravity": gravity,
                       "Wind": enable_wind,
                       "Wind power": wind_power,
                       "Turbulence power": turbulence_power,
                        "Video": wandb.Video(os.path.join(video_path, "rl-video-episode-0.mp4"), fps=4, format="gif")})


    avg_reward = round(sum(stop_reward) / len(stop_reward), 3)
    min_reward = round(min(stop_reward), 3)
    
    if eval_stop_condition == "avg":
        stop_reward = avg_reward
    elif eval_stop_condition == "min":
        stop_reward = min_reward
    else:
        raise ValueError(f"Unknown eval_stop_condition {eval_stop_condition}")
    
    save_path = os.path.join(experiment_path, "saves")
    
    agent.save_agent(save_path)
    
    
    art = wandb.Artifact("lunar_lander_model", type="model")
    for f in os.listdir(save_path):
        art.add_file(os.path.join(save_path, f))
    wandb.log_artifact(art)
    
    
    print(f"Episode: {episode} | Average evaluation reward: {avg_reward} | Min evaluation reward: {min_reward} | Agent saved at {save_path}")
    
    wandb.log({"Validation after episode": episode,  "Average evaluation reward": avg_reward,
               "Min evaluation reward": min_reward})
    with open(f"{experiment_path}/evaluation_rewards.csv", "a") as f:
        f.write(f"{episode}, {stop_reward}\n")
    try:
        if stop_reward > eval_task.spec.reward_threshold * 1.1:  # x 1.1 because of small eval_episodes
            print(f"Environment solved after {episode} episodes")
            return True
    except Exception as e:
        if stop_reward > -120:
            print(f"Environment solved after {episode} episodes")
            return True
    return False
