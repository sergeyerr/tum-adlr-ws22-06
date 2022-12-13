from DataHandling.ReplayBuffer import ReplayBuffer
import Networks
import torch
import os
import numpy as np


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
        self.pi = Networks.DDPGActor(alpha=self.pi_lr, input_dims=self.input_dims,
                                     n_actions=self.n_actions).to(self.device)
        self.target_pi = Networks.DDPGActor(alpha=self.pi_lr, input_dims=self.input_dims,
                                            n_actions=self.n_actions).to(self.device)
        self.pi_optimizer = self.pi.optimizer

        # Evaluation Network
        self.q = Networks.DDPGCritic(beta=self.q_lr, input_dims=self.input_dims,
                                     n_actions=self.n_actions).to(self.device)
        self.target_q = Networks.DDPGCritic(beta=self.q_lr, input_dims=self.input_dims,
                                            n_actions=self.n_actions).to(self.device)
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
        #self.target_pi.requires_grad_ = False
        #self.target_q.requires_grad_ = False
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

    def update_target_network(self):
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
