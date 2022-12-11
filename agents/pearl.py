
# this is an implementation of pearl with pytorch. They also implemented sac. It is super complicated because of all
# the inheritance
# https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/pearl.py
#
# thats the original implementation but it is not using pytorch so its super painful to read
# https://github.com/katerakelly/oyster
#
# this is a implementation that uses pytorch and only implements pearl, which should make it easier to understand
# I think it is actually the same thing as the original implementation
# https://github.com/waterhorse1/Pearl_relabel

# we will need a new replay buffer structure
# we will need functions to sample these replay buffers
# meta training, meta testing must be implemented
# the networks can mostly be used from sac

from .sac import SACAgent
import Networks
from DataHandling.ReplayBuffer import MultiTaskReplayBuffer
import torch


class PEARLAgent(SACAgent):

    #  inheriting from SACAgent to get all the networks and configs
    #  TODO make sure to change networks input size in config file, because they also take in latent variable

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        alpha = kwargs["alpha"]
        encoderParams = kwargs["encoderParams"]
        policyParams = kwargs["policyParams"]
        capacity = kwargs["ReplayBufferCapacity"]
        num_tasks = kwargs["num_train_tasks"]
        episode_length = kwargs["episode_length"]
        self.kl_lambda = kwargs["kl_lambda"]
        self.policy_mean_reg_weight = kwargs["policy_mean_reg_weight"]
        self.policy_std_reg_weight = kwargs["policy_std_reg_weight"]
        self.policy_pre_activation_weight = kwargs["policy_pre_activation_weight"]
        self.num_steps_prior = kwargs["num_steps_prior"]
        self.num_steps_posterior = kwargs["num_steps_posterior"]
        self.num_extra_rl_steps_posterior = kwargs["num_extra_rl_steps_posterior"]
        self.embedding_mini_batch_size = kwargs["embedding_mini_batch_size"]
        self.use_next_obs_in_context = kwargs["use_next_obs_in_context"]
        self.pi = Networks.PEARLPolicy(alpha, encoder_dict=encoderParams, policy_dict=policyParams)
        # stores experience
        self.replay_buffer = MultiTaskReplayBuffer(capacity, num_tasks)
        self.encoder_replay_buffer = MultiTaskReplayBuffer(capacity, num_tasks)

        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()
        self.vib_criterion = torch.nn.MSELoss()
        self.l2_reg_criterion = torch.nn.MSELoss()

    # performs one optimization step of agent
    def optimize(self, task_indices):

        # sample context batch
        context = self.encoder_replay_buffer.sample_random_batch(task_indices, sample_context=True,
                                                                 use_next_obs_in_context=self.use_next_obs_in_context)

        # zero out context and hidden encoder state
        self.pi.clear_z(num_tasks=len(task_indices))

        loss_results = {}
        num_tasks = len(task_indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.replay_buffer.sample_random_batch(task_indices)

        # run inference in networks
        policy_outputs, task_z = self.pi(obs, context)
        # TODO in networks SACActor output the necessary data
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.q_1(obs, actions, task_z)
        q2_pred = self.q_2(obs, actions, task_z)
        v_pred = self.value(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_value(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.pi.context_encoder.optimizer.zero_grad()
        kl_div = self.pi.compute_kl_div()
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.q_1.optimizer.zero_grad()
        self.q_1.optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.gamma * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        loss_results["critic_loss"] = qf_loss
        qf_loss.backward()
        self.q_1.optimizer.step()
        self.q_2.optimizer.step()
        self.pi.context_encoder.optimizer.step()

        # compute min Q on the new actions
        q1 = self.q_1(obs, new_actions, task_z.detach())
        q2 = self.q_2(obs, new_actions, task_z.detach())
        min_q_new_actions = torch.min(q1, q2)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.value.optimizer.zero_grad()
        vf_loss.backward()
        self.value.optimizer.step()
        self.update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        policy_loss = (log_pi - min_q_new_actions).mean()
        loss_results["actor loss"] = policy_loss.data

        # dont know what all that regularization is
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.pi.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.pi.policy.optimizer.step()

        return loss_results
