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

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc2_dims)
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

            self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
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
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2):
        super(SACActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
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

    def sample_normal(self, state, reparameterize=True, deterministic=False):
        mu, sigma = self.forward(state)
        # in spinning up: torch.distributions.normal.Normal()
        probabilities = torch.distributions.normal.Normal(mu, sigma)
        # probabilities = torch.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        if deterministic:
            actions = mu

        # action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)
        # log_probs = probabilities.log_prob(actions)
        # log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        # log_probs = log_probs.sum(1, keepdim=True)
        # in spinning up it is:
        log_probs = probabilities.log_prob(actions).sum(axis=-1)
        log_probs -= (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=-1)
        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)

        return action, log_probs


class ContextEncoder(nn.Module):
    def __init__(self, alpha, in_size, out_size, fc1_dims=200, fc2_dims=200, fc3_dims=200):
        super(ContextEncoder, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.in_size = in_size
        self.out_size = out_size

        self.fc1 = nn.Linear(*self.input_size, self.fc1_dims)
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

    def __init__(self, alpha, encoder_dict, policy_dict):
        super(PEARLPolicy, self).__init__()
        self.latent_dim = encoder_dict["latent_dim"]
        self.policy = SACActorNetwork(alpha=alpha, input_dims=policy_dict["input_dims"],
                                       max_action=policy_dict["max_action"])
        self.context_encoder = ContextEncoder(alpha=alpha, in_size=encoder_dict["in_size"],
                                               out_size=encoder_dict["out_size"])

        self.register_buffer('z', torch.zeros(1, self.latent_dim))
        self.register_buffer('z_means', torch.zeros(1, self.latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, self.latent_dim))

        self.z_means = None
        self.z_vars = None
        self.z = None
        self.context = None
        # TODO pass this to init
        self.use_next_obs_in_context = encoder_dict["use_next_obs_in_context"]

        self.clear_z()

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

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        o = torch.from_numpy(o)
        a = torch.from_numpy(a)
        r = torch.from_numpy(r)
        no = torch.from_numpy(no)

        # TODO the environment does not return a 3 dimensional array so dim=2 will throw an error...
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
    #          [1, 2]]]
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
        kl_div = torch.distributions.kl.kl_divergence(posteriors, prior)
        kl_div_sum = torch.sum(kl_div)
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        mu = params[:, :self.latent_dim]
        sigma_squared = F.softplus(params[:, self.latent_dim:])
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

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = torch.from_numpy(obs)
        in_ = torch.cat([obs, z], dim=1)
        # TODO the sample_normal function is different from what ref. impl. is doing
        action, _ = self.policy.sample_normal(in_, deterministic=deterministic)
        return action

    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        # TODO the policy does not output same things as in ref implementation
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z