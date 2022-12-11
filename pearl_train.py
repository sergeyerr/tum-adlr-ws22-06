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

import wandb
from agents import PEARLAgent

from utils import print_run_info, validate

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class PEARLExperiment(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.training_args = cfg["training"]
        print(self.training_args)
        self.agent_args = cfg["agent"]
        print(self.agent_args)
        self.env_args = cfg["env"]
        print(self.env_args)
        self.validation_args = cfg["validation"]
        self.train_env_fabric = LunarEnvRandomFabric(pass_env_params=self.training_args["pass_env_parameters"], **self.env_args)
        self.test_env_fabric = LunarEnvRandomFabric(pass_env_params=self.training_args["pass_env_parameters"],
                                                    render_mode='rgb_array', **self.env_args)
        # creates list of env with different parametrizations
        self.train_tasks = self.create_train_tasks(self.train_env_fabric, self.training_args["num_train_tasks"])
        self.eval_tasks = self.create_train_tasks(self.test_env_fabric, self.training_args["num_eval_tasks"])
        self.n_actions = self.train_tasks[0].action_space.shape[0]\
            if type(self.train_tasks[0].action_space) == gym.spaces.box.Box else self.train_tasks[0].action_space.n
        env_info = {"input_dims": self.train_tasks[0].observation_space.shape, "n_actions": self.n_actions,
                    "max_action": self.train_tasks[0].action_space.high}

        self.agent = PEARLAgent(**OmegaConf.to_object(self.agent_args), **OmegaConf.to_object(self.training_args),
                                **env_info)
        self.sampler = Sampler(self.train_tasks, self.eval_tasks, self.agent.pi, self.training_args["max_path_length"])
        self.episode_reward = 0.0
        self.task_idx = 0

    def run(self):

        experiment_name = self.agent_args.experiment_name

        T.manual_seed(self.training_args.seed)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
        np.random.seed(self.training_args.seed)
        random.seed(self.training_args.seed)
        # env.seed(self.args.seed)

        # Weights and biases initialization
        wandb.init(project="Model tests on the non-randomized env", entity="tum-adlr-ws22-06",
                   config=OmegaConf.to_object(self.cfg))

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

        if self.validation_args.log_model_wandb:
            # assumes that the model has only one actor, we may also log different models differently
            wandb.watch(self.agent.pi, log="all", log_freq=self.validation_args.log_model_every_training_batch)
            print(
                f"================= {f'Sending weights to W&B every {self.validation_args.log_model_every_training_batch} batch'} =================")

        reward_history = deque(maxlen=100)

        '''meta-training loop'''
        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for episode in range(self.training_args.episodes):
            self.episode_reward = 0.0
            actor_loss = 0.0
            critic_loss = 0.0

            if episode == 0:
                print('collecting initial pool of data for train and eval')
                for idx, env in enumerate(self.train_tasks):
                    env.reset()
                    self.task_idx = idx
                    self.collect_data(self.training_args.num_initial_steps, 1, np.inf)
            # Sample data from train tasks.
            for i in range(self.training_args.num_tasks_sample):
                idx = np.random.randint(self.training_args.num_train_tasks)
                self.task_idx = idx
                env = self.train_tasks[idx]
                env.reset()
                self.agent.encoder_replay_buffer.task_buffers[idx].clear_buffer()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.training_args.train_batches):
                indices = np.random.choice(self.training_args.num_train_tasks, self.training_args.meta_batch)
                loss = self.agent.optimize(indices)
                # Loss information kept for monitoring purposes during training
                actor_loss += loss['actor_loss']
                critic_loss += loss['critic_loss']
                wandb.log({"Training episode": episode, "Batch": episode * self.training_args.train_batches + train_step,
                           "train_actor_loss": loss['actor_loss'], "train_critic_loss": loss['critic_loss']})
                self._n_train_steps_total += 1

            # reward_history.append(self.episode_reward)

            if episode % self.validation_args.eval_interval == 0:
                solved = self.evaluate(episode)
                if solved:
                    # return true, that agent solved environment
                    return True
        # agent did not solve environment
        return False

    def evaluate(self, episode):

        indices = np.random.choice(range(self.training_args.num_train_tasks), self.training_args.num_eval_tasks)
        train_returns = []
        ### eval train tasks with posterior sampled from the training replay buffer
        for idx in indices:
            self.task_idx = idx
            env = self.train_tasks[self.task_idx]
            env.reset()
            paths = []
            for _ in range(self.training_args.num_steps_per_eval // self.training_args.max_path_length):
                context = self.agent.encoder_replay_buffer.sample_random_batch(self.task_idx,
                                                                               self.training_args.embedding_batch_size,
                                                                               sample_context=True,
                                                                               use_next_obs_in_context=
                                                                               self.training_args.use_next_obs_in_context
                                                                               )
                self.agent.pi.infer_posterior(context)
                path, _ = self.sampler.obtain_samples(self.task_idx,
                                                      deterministic=self.training_args.eval_deterministic,
                                                      max_samples=self.training_args.max_path_length,
                                                      accum_context=False, max_trajs=1, resample=np.inf)
                paths += path

            mean_reward = np.mean([sum(path["r"]) for path in paths])
            train_returns.append(mean_reward)
        train_returns = np.mean(train_returns)

        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self.eval_rollout(indices, episode)
        print('train online returns')
        print(train_online_returns)

        ### test tasks
        test_final_returns, test_online_returns = self.eval_rollout(range(len(self.eval_tasks)), episode)
        print('test online returns')
        print(test_online_returns)

    def eval_rollout(self, indices, episode, evalu=False):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, evalu=evalu)
                mean_reward = np.mean([sum(path["r"]) for path in paths])
                all_rets.append(mean_reward)
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def create_train_tasks(self, env_fabric, num_tasks):
        envs = []
        for i in range(num_tasks):
            envs.append(env_fabric.generate_env())
        return envs

    # does one roll out with deterministic policy
    def collect_paths(self, idx, evalu=False):
        self.task_idx = idx
        env = self.eval_taks[self.task_idx] if evalu else self.train_tasks[self.task_idx]
        env.reset()
        self.agent.pi.clear_z()
        paths = []
        num_trajectories = 0
        num_transitions = 0
        while num_transitions < self.training_args.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                    max_samples=self.num_steps_per_eval - num_transitions,
                                                    max_trajs=1, accum_context=True, task_idx=self.task_idx,
                                                    evalu=evalu)
            paths += path
            num_transitions += num
            num_trajectories += 1

            if num_trajectories >= self.training_args.num_exp_traj_eval:
                self.agent.pi.infer_posterior(self.agent.pi.context)

        return paths

    # generates rollout
    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):

        # get trajectories from current env in batch mode with given policy
        # collect complete trajectories until the number of collected transitions >= num_samples

        # start from the prior
        self.agent.pi.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            # paths is a list or dictionaries
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
                context = self.encoder_replay_buffer.sample_random_batch(self.task_idx,
                                                                         self.training_args.embedding_batch_size,
                                                                         sample_context=True,
                                                                         use_next_obs_in_context=
                                                                         self.training_args.use_next_obs_in_context)
                self.agent.pi.infer_posterior(context)
        self._n_env_steps_total += num_transitions

@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(cfg: DictConfig):
    config_dict = OmegaConf.to_object(cfg)
    experiment = PEARLExperiment(config_dict)
    experiment.run()
    # experiment = hydra.utils.instantiate(cfg.agent, cfg)
    # experiment.run()

if __name__ == '__main__':
    start()
