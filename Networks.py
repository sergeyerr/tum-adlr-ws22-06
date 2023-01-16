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

    def sample_normal(self, state, reparameterize, deterministic=False, return_log_prob=False):
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
