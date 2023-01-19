
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
import torch.nn.functional as F
import numpy as np


class PEARLAgent(SACAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        alpha = kwargs["pi_lr"]
        capacity = kwargs["replay_buffer_size"]
        num_tasks = kwargs["n_train_tasks"]
        latent_dim = kwargs["latent_size"]
        # TODO check if encoder in size is correct. I could be that this size is for the recurrent encoder
        encoder_in_size = kwargs["obs_dim"] + kwargs["n_actions"] + 1  # 1 is reward dimension
        encoder_in_size = encoder_in_size + kwargs["obs_dim"] if kwargs["use_next_obs_in_context"] else encoder_in_size
        encoder_out_size = latent_dim * 2 if kwargs['use_information_bottleneck'] else latent_dim
        policy_input_dims = kwargs["obs_dim"] + latent_dim
        self.use_next_obs_in_context = kwargs["use_next_obs_in_context"]
        self.latent_dim = latent_dim
        self.pi = Networks.SACActorNetwork(alpha=alpha, input_dims=policy_input_dims,
                                               max_action=kwargs["max_action"]).to(self.device)
        # TODO understand how the encoding actually works if encoder has no bottleneck architecture
        self.context_encoder = Networks.ContextEncoder(alpha=alpha, input_size=encoder_in_size,
                                                       out_size=encoder_out_size).to(self.device)

        # TODO check what these lines actually do. Can omit them?
        # self.register_buffer('z', torch.zeros(1, self.latent_dim))
        # self.register_buffer('z_means', torch.zeros(1, self.latent_dim))
        # self.register_buffer('z_vars', torch.zeros(1, self.latent_dim))

        self.z_means = None
        self.z_vars = None
        self.z = None
        self.context = None

        self.clear_z()

        # stores experience
        self.replay_buffer = MultiTaskReplayBuffer(capacity, num_tasks)
        self.encoder_replay_buffer = MultiTaskReplayBuffer(capacity, num_tasks)

        self.qf_criterion = torch.nn.MSELoss()
        self.vf_criterion = torch.nn.MSELoss()
        self.vib_criterion = torch.nn.MSELoss()
        self.l2_reg_criterion = torch.nn.MSELoss()

        ### params used in optimization
        self.kl_lambda = kwargs["kl_lambda"]
        self.policy_mean_reg_weight = kwargs["policy_mean_reg_weight"]
        self.policy_std_reg_weight = kwargs["policy_std_reg_weight"]
        self.policy_pre_activation_weight = kwargs["policy_pre_activation_weight"]
        self.num_steps_prior = kwargs["num_steps_prior"]
        self.num_steps_posterior = kwargs["num_steps_posterior"]
        self.num_extra_rl_steps_posterior = kwargs["num_extra_rl_steps_posterior"]
        self.embedding_batch_size = kwargs["embedding_batch_size"]
        self.batch_size = kwargs["batch_size"]


    def clear_z(self, num_tasks=1):
        # reset distribution over z to prior
        mu = torch.zeros(num_tasks, self.latent_dim)
        var = torch.ones(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.context = None

    def sample_z(self):
        posteriors = torch.distributions.normal.Normal(self.z_means, torch.sqrt(self.z_vars))
        self.z = posteriors.rsample()

    # TODO as expected this function is incorrect. The sizes dont make sense
    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no = inputs

        # These operations expand the dimensions of the arrays by two
        o = torch.from_numpy(o[None, None, ...]).to(device=self.device, dtype=torch.float)
        a = torch.from_numpy(a[None, None, ...]).to(device=self.device, dtype=torch.float)
        r = torch.from_numpy(np.array([r])[None, None, ...]).to(device=self.device, dtype=torch.float)
        no = torch.from_numpy(no[None, None, ...]).to(device=self.device, dtype=torch.float)

        # environment returns 1 d arrays
        # this concatenates the features along the feature dimension
        # o = [[[1, 2, 4],
        #      [1, 2, 4],
        #      [1, 2, 4]],
        #     [[1, 2, 4],
        #      [1, 2, 4],
        #      [1, 2, 4]]]
        # a = [[[1, 2],
        #       [1, 2],
        #       [1, 2]],
        #      [[1, 2],
        #       [1, 2],
        #       [1, 2]]]
        # data = [[[1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2]],
        #         [[1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2]]]
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        # this concatenates the context along the batch dimension
        # context = [[[1, 2, 4, 1, 2], first task
        #          [1, 2, 4, 1, 2], second transition from first task
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2]],
        #         [[1, 2, 4, 1, 2], second task
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2],
        #          [1, 2, 4, 1, 2]]]
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)


    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim), torch.ones(self.latent_dim))
        posteriors = torch.distributions.normal.Normal(self.z_means, torch.sqrt(self.z_vars))
        # this stuff is necessary else pytorch throws an error that the variable or not on same device
        posteriors.loc = posteriors.loc.to(device=self.device)
        posteriors.scale = posteriors.scale.to(device=self.device)
        prior.loc = prior.loc.to(device=self.device)
        prior.scale = prior.scale.to(device=self.device)
        kl_div = torch.distributions.kl.kl_divergence(posteriors, prior)
        kl_div_sum = torch.sum(kl_div)
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        # it is important to set dtype=torch.float (means float32) because context is a numpy array with float64
        # and the weight in the linear layer of context_encoder are float32.
        c = context.to(device=self.device, dtype=torch.float)
        params = self.context_encoder(c)
        params = params.view(context.size(0), -1, self.context_encoder.out_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        mu = params[:, :, :self.latent_dim]
        sigma_squared = F.softplus(params[:, :, self.latent_dim:])
        z_params = [self.product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        self.z_means = torch.stack([p[0] for p in z_params])
        self.z_vars = torch.stack([p[1] for p in z_params])

        self.sample_z()

    def product_of_gaussians(self, mus, s_squared):
        '''compute mu, sigma of product of gaussians'''
        sigmas_squared = torch.clamp(s_squared, min=1e-7)
        sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, sigma_squared

    def action(self, observation, addNoise=False, **kwargs):
        # sample action from the policy, conditioned on the task embedding
        z = self.z.to(self.device)
        obs = torch.from_numpy(observation).type(torch.float).to(self.device)
        obs = obs.view((-1, *obs.shape))  # enlargens by one dimension [*,*,*] -> [[*,*,*]]
        in_ = torch.cat([obs, z], dim=1)
        # with torch.no_grad():
        self.pi.eval()
        if addNoise:
            action, _ = self.pi.sample_normal(in_, reparameterize=False)  # in_ (1,13)
        else:
            action, _ = self.pi.sample_normal(in_, reparameterize=False, deterministic=True)
        return action.cpu().detach().numpy()[0]  # action is [[a1,a2]] so [0] is [a1,a2]

    # performs one optimization step of agent
    def optimize(self, task_indices):

        # sample context batch
        context = self.encoder_replay_buffer.sample_random_batch(task_indices, batch_size=self.embedding_batch_size,
                                                                 sample_context=True,
                                                                 use_next_obs_in_context=self.use_next_obs_in_context)

        # zero out context and hidden encoder state
        self.clear_z(num_tasks=len(task_indices))

        loss_results = {}
        num_tasks = len(task_indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.replay_buffer.sample_random_batch(task_indices,
                                                                                        batch_size=self.batch_size)

        # important step to make sure there are no errors like "found at least two devices, cpu and cuda:0!"
        # or "dtype mismatch"
        context = context.to(device=self.device, dtype=torch.float)
        obs = obs.to(dtype=torch.float, device=self.device)
        actions = actions.to(dtype=torch.float, device=self.device)
        rewards = rewards.to(dtype=torch.float, device=self.device)  # .view((-1, 1))
        next_obs = next_obs.to(dtype=torch.float, device=self.device)
        terms = terms.to(dtype=torch.float, device=self.device)  # .view((-1, 1))

        # flattens out the task dimension
        t, b, _ = obs.size()  # meta_batch, batch_size, obs_dim=8
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # run inference in networks
        self.infer_posterior(context)
        self.sample_z()
        task_z = self.z.to(self.device)  # shape (meta_batch, latent_dim) meaning for each task one z
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)  # now task_z has same first dim as obs, so meta_batch*batch_size, obs_dim

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)  # meta_batch*batch_size, obs_dim+latent_dim

        policy_outputs = self.pi.sample_normal(in_, reparameterize=True, return_log_prob=True)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # tanh_action, mean, T.log(sigma), log_probs, None, sigma, None, action

        # Q and V networks
        # encoder will only get gradients from Q nets
        # TODO check if concatentation in dim=1 is correct. Also check the SACCriticNetwork
        #  (in reference the cat with dim 1)
        # TODO also check if order ob inputs matter. In reference implementation they have o, a, z
        q1_pred = self.q_1(torch.cat([obs, task_z], dim=1), actions)
        q2_pred = self.q_2(torch.cat([obs, task_z], dim=1), actions)
        v_pred = self.value(torch.cat([obs, task_z.detach()], dim=1))
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_value(torch.cat([next_obs, task_z], dim=1))

        # KL constraint on z if probabilistic
        self.context_encoder.optimizer.zero_grad()
        kl_div = self.compute_kl_div()
        kl_loss = self.kl_lambda * kl_div
        kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.q_1.optimizer.zero_grad()
        self.q_2.optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        # TODO check self.scale parameter
        rewards_flat = rewards_flat * self.scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.gamma * target_v_values
        q_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        loss_results["critic_loss"] = q_loss.data
        q_loss.backward()
        self.q_1.optimizer.step()
        self.q_2.optimizer.step()
        self.context_encoder.optimizer.step()

        # compute min Q on the new actions
        q1 = self.q_1(torch.cat([obs, task_z.detach()], dim=1), new_actions)
        q2 = self.q_2(torch.cat([obs, task_z.detach()], dim=1), new_actions)
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

        # dont know what all that regularization is
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value ** 2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        loss_results["actor_loss"] = policy_loss.data

        self.pi.optimizer.zero_grad()
        policy_loss.backward()
        self.pi.optimizer.step()

        return loss_results
