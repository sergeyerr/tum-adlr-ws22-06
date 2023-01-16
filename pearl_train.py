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
from EnvironmentUtils import LunarEnvRandomFabric, LunarEnvFixedFabric, LunarEnvHypercubeFabric
from DataHandling.EnvironmentSampler import Sampler
from Noise import ZeroNoise

import wandb
from agents import PEARLAgent

from utils import print_run_info, validate

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class PEARLExperiment(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.training_args = cfg["training"]
        # print(f"training args {json.dumps(self.training_args, indent=4)}")
        self.agent_args = cfg["agent"]
        # print(f"agent args{json.dumps(self.agent_args, indent=4)}")
        self.env_args = cfg["env"]
        # print(f"env args{json.dumps(self.env_args, indent=4)}")
        self.validation_args = cfg["validation"]
        # print(f"validation args{json.dumps(self.validation_args, indent=4)}")

        self.train_env_fabric = LunarEnvRandomFabric(pass_env_params=self.training_args["pass_env_parameters"],
                                                     **self.env_args)

        if self.validation_args["hypercube_validation"]:
            self.test_env_fabric = LunarEnvHypercubeFabric(pass_env_params=self.training_args["pass_env_parameters"],
                                                      render_mode= 'rgb_array',
                                                      points_per_axis=self.validation_args["hypercube_points_per_axis"],
                                                      **self.env_args)
            self.validation_args["eval_eps"] = self.test_env_fabric.number_of_test_points()
        else:
            self.test_env_fabric = LunarEnvRandomFabric(env_params=self.env_args, pass_env_params=self.training_args["pass_env_parameters"],
                                                   render_mode= 'rgb_array')

        # creates list of env with different parameterizations
        self.train_tasks = self.create_train_tasks(self.train_env_fabric, self.training_args["n_train_tasks"])
        self.eval_tasks = self.create_train_tasks(self.test_env_fabric, self.validation_args["eval_eps"])

        self.n_actions = int(np.prod(self.train_tasks[0].action_space.shape)) \
            if type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n

        env_info = {"obs_dim": int(np.prod(self.train_tasks[0].observation_space.shape)), "n_actions": self.n_actions,
                    "max_action": self.train_tasks[0].action_space.high}

        # this is critical so that the q and v functions have the right input size
        env_info["input_dims"] = env_info["obs_dim"] + self.agent_args["latent_size"]
        self.agent = PEARLAgent(**self.agent_args, **self.training_args, **env_info)

        self.sampler = Sampler(self.train_tasks, self.eval_tasks, self.agent, self.training_args["max_path_length"])
        self.episode_reward = 0.0
        self.task_idx = 0

    def run(self):

        experiment_name = self.agent_args["experiment_name"]

        T.manual_seed(self.training_args["seed"])
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        np.random.seed(self.training_args["seed"])
        random.seed(self.training_args["seed"])
        # env.seed(self.args.seed)

        # Weights and biases initialization
        wandb.init(project="ADLR randomized envs with Meta RL", entity="tum-adlr-ws22-06",
                   config=self.cfg)

        # Experiment directory storage
        env_path = os.path.join("experiments", "LunarLanderContinuous-v2")
        if not os.path.exists(env_path):
            os.makedirs(env_path)

        # ugly, but acceptable
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
            OmegaConf.save(self.cfg, f)

        if self.validation_args["log_model_wandb"]:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args["log_model_every_training_batch"])
            temp = self.validation_args["log_model_every_training_batch"]
            print(
                f"================= {f'Sending weights to W&B every {temp} batch'} =================")

        noise = ZeroNoise(self.n_actions)

        print_run_info(self.train_tasks[0], self.agent, self.agent_args, self.training_args, self.env_args,
                       self.validation_args, noise)

        reward_history = deque(maxlen=100)

        # meta-training loop
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for episode in range(self.training_args["num_iterations"]):
            self.episode_reward = 0.0
            actor_loss = 0.0
            critic_loss = 0.0

            if episode == 0:
                print('collecting initial pool of data for train and eval')
                for idx, env in enumerate(self.train_tasks):
                    env.reset()
                    self.task_idx = idx
                    # TODO why do this if we clear encoder replay buffer later
                    self.collect_data(self.training_args["num_initial_steps"], 1, np.inf)
            # Sample data from train tasks.
            print('Sample data from train tasks')
            for i in range(self.training_args["num_tasks_sample"]):
                idx = np.random.randint(self.training_args["n_train_tasks"])
                self.task_idx = idx
                env = self.train_tasks[idx]
                env.reset()
                # TODO understand why we delete the replay buffer here although we filled it in loop before
                self.agent.encoder_replay_buffer.clear_buffer(idx)

                # collect some trajectories with z ~ prior
                if self.training_args["num_steps_prior"] > 0:
                    self.collect_data(self.training_args["num_steps_prior"], 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.training_args["num_steps_posterior"] > 0:
                    self.collect_data(self.training_args["num_steps_posterior"], 1,
                                      self.training_args["update_post_train"])
                # even if encoder is trained only on samples from the prior,
                # the policy needs to learn to handle z ~ posterior
                if self.training_args["num_extra_rl_steps_posterior"] > 0:
                    self.collect_data(self.training_args["num_extra_rl_steps_posterior"],
                                      1, self.training_args["update_post_train"], add_to_enc_buffer=False)

            # Sample train tasks and compute gradient updates on parameters.
            print("Sample train tasks and compute gradient updates on parameters.")
            for train_step in range(self.training_args["num_train_steps_per_itr"]):
                indices = np.random.choice(self.training_args["n_train_tasks"], self.training_args["meta_batch"])
                loss = self.agent.optimize(indices)
                # Loss information kept for monitoring purposes during training
                actor_loss += loss['actor_loss']
                critic_loss += loss['critic_loss']

                wandb.log({"Training episode": episode, "Batch": episode * self.training_args["num_train_steps_per_itr"] + train_step,
                           "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})

            # reward_history.append(self.episode_reward)
            print(f"episode actor loss is: {loss['actor_loss']} \t episode critic loss is: {loss['critic_loss']}")

            if episode % self.validation_args["eval_interval"] == 0:
                solved = validate(self.agent, self.validation_args, experiment_path, episode,
                                  self.test_env_fabric, pearl=True)
                if solved:
                    break

    def create_train_tasks(self, env_fabric, num_tasks):
        envs = []
        for i in range(num_tasks):
            envs.append(env_fabric.generate_env())
        return envs

    # generates rollout
    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):

        # get trajectories from current env in batch mode with given policy
        # collect complete trajectories until the number of collected transitions >= num_samples

        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            # paths is a list of dictionaries
            # each dictionary is a path. The values for the keys are two-dimensional np arrays
            paths, n_samples = self.sampler.obtain_samples(task_idx=self.task_idx,
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.agent.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.agent.encoder_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.agent.encoder_replay_buffer.sample_random_batch(self.task_idx,
                                                                               self.training_args["embedding_batch_size"],
                                                                               sample_context=True,
                                                                               use_next_obs_in_context=
                                                                               self.agent_args["use_next_obs_in_context"])
                self.agent.infer_posterior(context)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(cfg: DictConfig):
    config_dict = OmegaConf.to_object(cfg)
    experiment = PEARLExperiment(config_dict)
    experiment.run()


if __name__ == '__main__':
    start()
