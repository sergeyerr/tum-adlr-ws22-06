import Networks
from .ReplayBuffer import ReplayBuffer
import Networks
import torch
import os
import numpy as np

class SACAgent(object):
    def __init__(self, **kwargs):

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pi_lr = kwargs["pi_lr"]
        self.q_lr = kwargs["q_lr"]
        self.input_dims = kwargs["input_dims"]
        self.n_actions = kwargs["n_actions"]
        self.max_action = kwargs["max_action"]

        # Neural networks
        # Policy Network
        self.pi = Networks.SACActorNetwork(alpha=self.pi_lr, input_dims=self.input_dims, n_actions=self.n_actions,
                                           max_action=self.max_action).to(self.device)
        self.q_1 = Networks.SACCriticNetwork(beta=self.q_lr, input_dims=self.input_dims, n_actions=self.n_actions)
        self.q_2 = Networks.SACCriticNetwork(beta=self.q_lr, input_dims=self.input_dims, n_actions=self.n_actions)
        self.value = Networks.SACValueNetwork(beta=self.q_lr, input_dims=self.input_dims)
        self.target_value = Networks.SACValueNetwork(beta=self.q_lr, input_dims=self.input_dims)

        # Sync weights
        self.sync_weights()

        # Replay buffer
        self.min_replay_size = kwargs["min_replay_size"]
        self.replay_buffer = ReplayBuffer(kwargs["replay_buffer_size"])

        # Constants
        self.tau = kwargs["tau"]
        self.gamma = kwargs["gamma"]
        self.batch_size = kwargs["batch_size"]
        self.scale = kwargs["reward_scale"]

    def action(self, observation, addNoise=False, **kwargs):
        obs = torch.from_numpy(observation).type(torch.float).to(self.device)
        obs = obs.view((-1, *obs.shape))
        self.pi.eval()
        if addNoise:
            actions, _ = self.pi.sample_normal(obs, reparameterize=False)
        else:
            actions, _ = self.pi.sample_normal(obs, reparameterize=False, deterministic=True)
        return actions.cpu().detach().numpy()[0]

    def train(self):
        # Loss statistics
        loss_results = {}
        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs, actions, rewards, new_obs, done = sample['o'], sample['a'], sample['r'], sample['o2'], sample['d']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        replay_actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).view((-1, 1))
        new_obs = torch.from_numpy(np.stack(new_obs)).to(dtype=torch.float, device=self.device)
        # done = torch.tensor(done, dtype=torch.bool, device=self.device).view((-1, 1))
        done = torch.tensor(done).to(self.device)

        # get values for states
        value = self.value.forward(obs).view(-1)
        value_ = self.target_value.forward(new_obs).view(-1)
        # terminal states are defines as having value=0
        value_[done] = 0.0

        # calculate critic value
        actions, log_probs = self.pi.sample_normal(obs, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.q_1.forward(obs, actions)
        q2_new_policy = self.q_2.forward(obs, actions)
        q_value = torch.min(q1_new_policy, q2_new_policy)
        q_value = q_value.view(-1)

        # backprop thru value network with the critic value
        self.value.optimizer.zero_grad()
        value_target = q_value - log_probs
        value_loss = 0.5*torch.nn.functional.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # calculate critic value again but with reparameterization trick on
        actions, log_probs = self.pi.sample_normal(obs, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.q_1.forward(obs, actions)
        q2_new_policy = self.q_2.forward(obs, actions)
        q_value = torch.min(q1_new_policy, q2_new_policy)
        q_value = q_value.view(-1)

        # backprop thru actor
        actor_loss = log_probs - q_value
        actor_loss = torch.mean(actor_loss)
        loss_results['actor_loss'] = actor_loss.data
        self.pi.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.pi.optimizer.step()

        # calculate loss of critics
        self.q_1.optimizer.zero_grad()
        self.q_2.optimizer.zero_grad()
        q_hat = self.scale * rewards.view(-1) + self.gamma * value_
        q1_old_policy = self.q_1.forward(obs, replay_actions).view(-1)
        q2_old_policy = self.q_2.forward(obs, replay_actions).view(-1)
        q_1_loss = 0.5 * torch.nn.functional.mse_loss(q1_old_policy, q_hat)
        q_2_loss = 0.5 * torch.nn.functional.mse_loss(q2_old_policy, q_hat)

        q_loss = q_1_loss + q_2_loss
        loss_results['critic_loss'] = q_loss.data
        q_loss.backward()
        self.q_1.optimizer.step()
        self.q_2.optimizer.step()

        return loss_results

    def experience(self, o, a, r, o2, d):
        self.replay_buffer.record(o, a, r, o2, d)

    def update(self):
        with torch.no_grad():
            for value_param, target_value_param in zip(self.value.parameters(), self.target_value.parameters()):
                target_value_param.data = (1.0 - self.tau) * target_value_param.data + self.tau * value_param.data

    def sync_weights(self):
        with torch.no_grad():
            for value_param, target_value_param in zip(self.value.parameters(), self.target_value.parameters()):
                target_value_param.data = value_param.data

    def save_agent(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_value_path = os.path.join(save_path, "target_value_network.pth")
        torch.save(self.target_value.state_dict(), target_value_path)

        critic_1_path = os.path.join(save_path, "q_1_network.pth")
        torch.save(self.q_1.state_dict(), critic_1_path)

        critic_2_path = os.path.join(save_path, "q_2_network.pth")
        torch.save(self.q_2.state_dict(), critic_2_path)

        pi_path = os.path.join(save_path, "pi_network.pth")
        torch.save(self.pi.state_dict(), pi_path)

        value_path = os.path.join(save_path, "value_network.pth")
        torch.save(self.value.state_dict(), value_path)

    def load_agent(self, save_path):
        pi_path = os.path.join(save_path, "pi_network.pth")
        self.pi.load_state_dict(torch.load(pi_path))
        self.pi.eval()

        target_value_path = os.path.join(save_path, "target_value_network.pth")
        self.target_value.load_state_dict(torch.load(target_value_path))
        self.target_value.eval()

        value_path = os.path.join(save_path, "value_network.pth")
        self.value.load_state_dict(torch.load(value_path))
        self.value.eval()

        critic_1_path = os.path.join(save_path, "q_1_network.pth")
        self.q_1.load_state_dict(torch.load(critic_1_path))
        self.q_1.eval()

        critic_2_path = os.path.join(save_path, "q_2_network.pth")
        self.q_2.load_state_dict(torch.load(critic_2_path))
        self.q_2.eval()

        self.sync_weights()

    def __str__(self):
        return str(self.pi) + str(self.value)


