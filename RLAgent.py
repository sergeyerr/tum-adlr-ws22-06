import Networks
from ReplayBuffer import ReplayBuffer
import Networks
import torch
import os
import numpy as np
import random


class DDPGAgent(object):
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
        self.pi = Networks.DDPGActor(alpha=self.pi_lr, input_dims=self.input_dims, n_actions=self.n_actions).to(self.device)
        self.target_pi = Networks.DDPGActor(alpha=self.pi_lr, input_dims=self.input_dims, n_actions=self.n_actions).to(self.device)
        self.pi_optimizer = self.pi.optimizer

        # Evaluation Network
        self.q = Networks.DDPGCritic(beta=self.q_lr, input_dims=self.input_dims, n_actions=self.n_actions).to(self.device)
        self.target_q = Networks.DDPGCritic(beta=self.q_lr, input_dims=self.input_dims, n_actions=self.n_actions).to(self.device)
        self.q_optimizer = self.q.optimizer

        # Sync weights
        self.sync_weights()

        # Replay buffer
        self.min_replay_size = kwargs["min_replay_size"]
        self.replay_buffer = ReplayBuffer(kwargs["replay_buffer_size"])

        # Constants
        self.tau = kwargs["tau"]
        self.gamma = kwargs["gamma"]
        self.batch_size = kwargs["batch_size"]
        # self.scale = kwargs["scale"]

    def action(self, observation, addNoise=False, **kwargs):
        obs = torch.from_numpy(observation).type(torch.float).to(self.device)
        obs = obs.view((-1, *obs.shape))
        self.pi.eval()
        action = self.pi(obs)[0]
        if addNoise:
            noise = kwargs['noise']
            action += torch.tensor(noise(), dtype=torch.float, device=self.device)
            action = torch.clamp(action, -1.0, 1.0)

        return action.detach().cpu().numpy() * self.max_action

    def train(self):
        # Loss statistics
        loss_results = {}
        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs, actions, rewards, new_obs, done = sample['o'], sample['a'], sample['r'], sample['o2'], sample['d']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).view((-1,1))
        new_obs = torch.from_numpy(np.stack(new_obs)).to(dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device).view((-1,1))

        self.target_pi.eval()
        self.target_q.eval()
        self.q.eval()

        # Train q network
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - done) * self.target_q(new_obs, self.target_pi(new_obs))
        predicted = self.q(obs, actions)
        loss = ((targets - predicted) ** 2).mean()
        loss_results['critic_loss'] = loss.data

        self.q_optimizer.zero_grad()
        self.q.train()
        loss.backward()
        self.q_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = False

        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs = sample['o']

        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)

        self.q.eval()
        self.pi.eval()

        # Train pi network
        predicted = self.q(obs, self.pi(obs))
        loss = -predicted.mean()
        loss_results['actor_loss'] = loss.data
        self.pi_optimizer.zero_grad()
        self.pi.train()
        loss.backward()
        self.pi_optimizer.step()

        for p in self.q.parameters():
            p.requires_grad = True

        return loss_results

    def experience(self, o, a, r, o2, d):
        self.replay_buffer.record(o, a, r, o2, d)

    def update(self):
        with torch.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(), self.target_pi.parameters()):
                target_pi_param.data = (1.0 - self.tau) * target_pi_param.data + self.tau * pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(), self.target_q.parameters()):
                target_q_param.data = (1.0 - self.tau) * target_q_param.data + self.tau * q_param.data

    def sync_weights(self):
        with torch.no_grad():
            for pi_param, target_pi_param in zip(self.pi.parameters(), self.target_pi.parameters()):
                target_pi_param.data = pi_param.data

            for q_param, target_q_param in zip(self.q.parameters(), self.target_q.parameters()):
                target_q_param.data = q_param.data

    def save_agent(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        torch.save(self.target_pi.state_dict(), target_pi_path)

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        torch.save(self.target_q.state_dict(), target_q_path)

        pi_path = os.path.join(save_path, "pi_network.pth")
        torch.save(self.pi.state_dict(), pi_path)

        q_path = os.path.join(save_path, "q_network.pth")
        torch.save(self.q.state_dict(), q_path)

    def load_agent(self, save_path):
        pi_path = os.path.join(save_path, "pi_network.pth")
        self.pi.load_state_dict(torch.load(pi_path))
        self.pi.eval()

        target_pi_path = os.path.join(save_path, "target_pi_network.pth")
        self.target_pi.load_state_dict(torch.load(target_pi_path))
        self.target_pi.eval()

        q_path = os.path.join(save_path, "q_network.pth")
        self.q.load_state_dict(torch.load(q_path))
        self.q.eval()

        target_q_path = os.path.join(save_path, "target_q_network.pth")
        self.target_q.load_state_dict(torch.load(target_q_path))
        self.target_q.eval()

        self.sync_weights()

    def __str__(self):
        return str(self.pi) + str(self.q)


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


class SACAgent2(SACAgent):
    # TODO this SAC algorithm does not work yet, it diverges.
    def __init__(self, **kwargs):
        self.q_lr = kwargs["q_lr"]
        self.input_dims = kwargs["input_dims"]
        self.n_actions = kwargs["n_actions"]
        self.q_1_target = Networks.SACCriticNetwork(beta=self.q_lr, input_dims=self.input_dims,
                                                    n_actions=self.n_actions)
        self.q_2_target = Networks.SACCriticNetwork(beta=self.q_lr, input_dims=self.input_dims,
                                                    n_actions=self.n_actions)

        # we not backprop thru target networks, so we wont need gradients
        for p in self.q_2_target.parameters():
            p.requires_grad = False

        for p in self.q_1_target.parameters():
            p.requires_grad = False

        super().__init__(**kwargs)

        self.alpha = kwargs["alpha"]

    def compute_loss_q(self, obs, actions, rewards, new_obs, done):
        # update q networks
        # current q estimate
        q1 = self.q_1(obs, actions)
        q2 = self.q_2(obs, actions)

        # Bellman update using current policy and new observation
        with torch.no_grad():
            next_action, logprobs = self.pi(new_obs) # I think you got flow of gradients wrong here
            q1_target = self.q_1_target(new_obs, next_action)
            q2_target = self.q_2_target(new_obs, next_action)
            q_target = torch.min(q1_target, q2_target)
            bellman = rewards + self.gamma * (1-done) * (q_target - self.alpha * logprobs)

        # MSE loss of current q against bellman
        loss_q1 = ((q1 - bellman)**2).mean()
        loss_q2 = ((q2 - bellman)**2).mean()
        loss_q = loss_q1 + loss_q2
        return loss_q
    
    def compute_loss_pi(self, obs):
        # update policy
        action, logprobs = self.pi(obs)
        q1 = self.q_1(obs, action)
        q2 = self.q_2(obs, action)
        q = torch.min(q1, q2)
        loss_pi = (self.alpha * logprobs - q).mean()
        return loss_pi

    def train(self):
        # Loss statistics
        loss_results = {}
        # Get samples from replay buffer
        sample = self.replay_buffer.get_batch(self.batch_size)
        obs, actions, rewards, new_obs, done = sample['o'], sample['a'], sample['r'], sample['o2'], sample['d']

        
        # Convert samples to tensors
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).view((-1, 1))
        new_obs = torch.from_numpy(np.stack(new_obs)).to(dtype=torch.float, device=self.device)
        done = torch.tensor(done, dtype=torch.float, device=self.device).view((-1, 1))

        # backprop # probably you zeroed grads in the wrond place (normally it stays before all computation, but it seems not important)
        self.q_1.optimizer.zero_grad()
        self.q_2.optimizer.zero_grad()
        loss_q = self.compute_loss_q(obs, actions, rewards, new_obs, done)
        loss_results['critic_loss'] = loss_q.data
          
        loss_q.backward()
        self.q_1.optimizer.step()
        self.q_2.optimizer.step()

        # update policy
        # don't track quality function gradients
        # for p in self.q_1.parameters():
        #     p.requires_grad = False
        # for p in self.q_2.parameters():
        #     p.requires_grad = False

        # compute loss pi given updated q functions
        self.pi.optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(obs)
        loss_results['actor_loss'] = loss_pi.data

        # update policy
        loss_pi.backward()
        self.pi.optimizer.step()

        # un-freeze quality networks
        # for p in self.q_1.parameters():
        #     p.requires_grad = True
        # for p in self.q_2.parameters():
        #     p.requires_grad = True

        return loss_results

    def update(self):

        with torch.no_grad():
            for q1_param, target_q1_param in zip(self.q_1.parameters(), self.q_1_target.parameters()):
                target_q1_param.data = (1.0 - self.tau) * target_q1_param.data + self.tau * q1_param.data

            for q2_param, target_q2_param in zip(self.q_2.parameters(), self.q_2_target.parameters()):
                target_q2_param.data = (1.0 - self.tau) * target_q2_param.data + self.tau * q2_param.data

    def sync_weights(self):
        with torch.no_grad():
            for q1_param, target_q1_param in zip(self.q_1.parameters(), self.q_1_target.parameters()):
                target_q1_param.data = q1_param.data

            for q2_param, target_q2_param in zip(self.q_2.parameters(), self.q_2_target.parameters()):
                target_q2_param.data = q2_param.data

    def adjust_temperature(self):
        pass

