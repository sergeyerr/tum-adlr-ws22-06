import torch as T
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

shared_CNN = None
device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class DDPGActor(nn.Module):

    def __init__(self, alpha=0.0001, input_dims=[8], fc1_dims=400, fc2_dims=300, n_actions=2):
        super(DDPGActor, self).__init__()
        self.alpha = alpha
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        # TODO: check if activation is missing here
        x = T.tanh(self.mu(x))

        return x


class DDPGCritic(nn.Module):
    def __init__(self, beta=0.001, input_dims=[8], fc1_dims=400, fc2_dims=300, n_actions=2):
        super(DDPGCritic, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, fc2_dims)
        f3 = 0.0003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value


class SACCriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(SACCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        if isinstance(input_dims, list):
            self.input_dims = int(np.prod(input_dims))

        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc2_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q


class SACValueNetwork(nn.Module):
        def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256):
            super(SACValueNetwork, self).__init__()
            self.input_dims = input_dims
            self.fc1_dims = fc1_dims
            self.fc2_dims = fc2_dims
            if isinstance(input_dims, list):
                self.input_dims = int(np.prod(input_dims))

            self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            self.v = nn.Linear(self.fc2_dims, 1)

            self.optimizer = optim.Adam(self.parameters(), lr=beta)
            self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

            self.to(self.device)

        def forward(self, state):
            state_value = self.fc1(state)
            state_value = F.relu(state_value)
            state_value = self.fc2(state_value)
            state_value = F.relu(state_value)

            v = self.v(state_value)

            return v


class SACActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2):
        super(SACActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6

        if isinstance(input_dims, list):
            self.input_dims = int(np.prod(input_dims))

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        log_sigma = self.sigma(prob)

        #in spinning up they clamp between -20 and 2
        # sigma = T.clamp(log_sigma, min=self.reparam_noise, max=1)
        log_sigma = T.clamp(log_sigma, -20, 2)
        # in spinning up they do
        sigma = T.exp(log_sigma)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True, deterministic=False, return_log_prob=False):
        mu, sigma = self.forward(state)
        probabilities = torch.distributions.normal.Normal(mu, sigma)

        if reparameterize:
            action = probabilities.rsample()
        else:
            action = probabilities.sample()

        if deterministic:
            action = mu

        # in spinning up it is:
        log_probs = probabilities.log_prob(action).sum(axis=-1)
        log_probs -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=-1)
        tanh_action = T.tanh(action)*T.tensor(self.max_action).to(self.device)

        if return_log_prob:
            # this is from the pearl implementation
            log_probs = probabilities.log_prob(action)
            log_probs -= T.log(1 - tanh_action*tanh_action + self.reparam_noise)
            log_probs = log_probs.sum(dim=1, keepdims=True)
            return tanh_action, mu, T.log(sigma), log_probs, action
        else:
            return tanh_action, log_probs


class ContextEncoder(nn.Module):
    def __init__(self, alpha, input_size, out_size, fc1_dims=200, fc2_dims=200, fc3_dims=200):
        super(ContextEncoder, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.input_size = input_size  # obs_dim+act_dim+latent_dim = 11
        self.out_size = out_size
        if isinstance(input_size, list):
            self.input_size = int(np.prod(input_size))

        self.fc1 = nn.Linear(self.input_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.z_layer = nn.Linear(self.fc3_dims, self.out_size)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, context):
        value = self.fc1(context)
        value = F.relu(value)
        value = self.fc2(value)
        value = F.relu(value)
        value = self.fc3(value)
        value = F.relu(value)
        latent_z = self.z_layer(value)

        return latent_z


class PEARLPolicy(nn.Module):

    def __init__(self, alpha, latent_dim, policy_input_dims, encoder_in_size, max_action,
                 encoder_out_size, use_next_obs_in_context):
        super(PEARLPolicy, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        self.policy = SACActorNetwork(alpha=alpha, input_dims=policy_input_dims, max_action=max_action).to(self.device)
        self.context_encoder = ContextEncoder(alpha=alpha, input_size=encoder_in_size,
                                              out_size=encoder_out_size).to(self.device)

        self.register_buffer('z', torch.zeros(1, self.latent_dim))
        self.register_buffer('z_means', torch.zeros(1, self.latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, self.latent_dim))

        self.z_means = None
        self.z_vars = None
        self.z = None
        self.context = None

        self.use_next_obs_in_context = use_next_obs_in_context

        self.clear_z()

        self.to(self.device)

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
        o, a, r, no, d, info = inputs

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

    def get_action(self, observation, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        # TODO in sac.py we set policy.eval() before sampling
        z = self.z.to(self.device)
        obs = torch.from_numpy(observation).type(torch.float).to(self.device)
        obs = obs.view((-1, *obs.shape))  # enlargens by one dimension [*,*,*] -> [[*,*,*]]
        in_ = torch.cat([obs, z], dim=1)

        with torch.no_grad():
            action, _ = self.policy.sample_normal(in_, deterministic=deterministic)  # in_ (1,13)
        return action.cpu().detach().numpy()[0]  # action is [[a1,a2]] so [0] is [a1,a2]

    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        # context and obs are here already dtype float and on correct device
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z.to(self.device)  # shape (meta_batch, latent_dim) meaning for each task one z

        t, b, _ = obs.size()  # meta_batch, batch_size, obs_dim=8
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)  # now task_z has same first dim as obs, so meta_batch*batch_size, obs_dim

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)  # meta_batch*batch_size, obs_dim+latent_dim

        policy_outputs = self.policy.sample_normal(in_, reparameterize=True, return_log_prob=True)  # , return_log_prob=True)

        return policy_outputs, task_z