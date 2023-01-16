from . import SACAgent
import Networks
import torch
import os
import numpy as np
import random

class SACAgent2(SACAgent):
    # based on https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
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
            next_action, logprobs = self.pi.sample_normal(new_obs, reparameterize=True)
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
        action, logprobs = self.pi.sample_normal(obs, reparameterize=True)
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

    def update_target_network(self):

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
                
                
    def save_agent(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)


        critic_1_path = os.path.join(save_path, "q_1_network.pth")
        torch.save(self.q_1.state_dict(), critic_1_path)

        critic_2_path = os.path.join(save_path, "q_2_network.pth")
        torch.save(self.q_2.state_dict(), critic_2_path)

        pi_path = os.path.join(save_path, "pi_network.pth")
        torch.save(self.pi.state_dict(), pi_path)

        target_value_path = os.path.join(save_path, "target_q1_network.pth")
        torch.save(self.q_1_target.state_dict(), target_value_path)
        
        target_value_path = os.path.join(save_path, "target_q2_network.pth")
        torch.save(self.q_2_target.state_dict(), target_value_path)

    def load_agent(self, save_path):
        pi_path = os.path.join(save_path, "pi_network.pth")
        self.pi.load_state_dict(torch.load(pi_path))
        self.pi.eval()

        critic_1_path = os.path.join(save_path, "q_1_network.pth")
        self.q_1.load_state_dict(torch.load(critic_1_path))
        self.q_1.eval()

        critic_2_path = os.path.join(save_path, "q_2_network.pth")
        self.q_2.load_state_dict(torch.load(critic_2_path))
        self.q_2.eval()
        
        target_q1_path = os.path.join(save_path, "target_q1_network.pth")
        self.q_1_target.load_state_dict(torch.load(target_q1_path))
        self.q_1_target.eval()
        
        target_q2_path = os.path.join(save_path, "target_q2_network.pth")
        self.q_2_target.load_state_dict(torch.load(target_q2_path))
        self.q_2_target.eval()

        self.sync_weights()

    def adjust_temperature(self):
        pass

