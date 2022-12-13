# this calls will create rollouts of agent for specific task
import numpy as np
import torch


class Sampler(object):

    # this sampler will use the current policy and environment/task to create rollout
    # this will affect the environment

    def __init__(self, train_tasks, eval_tasks, policy, max_path_length):
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        # don't forget to update policy when it is changed
        self.policy = policy
        self.max_path_length = max_path_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def obtain_samples(self, task_idx, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1,
                       deterministic=False, evalu=False):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = self.rollout(task_idx, policy, max_path_length=self.max_path_length, accum_context=accum_context,
                                deterministic=deterministic, evalu=evalu)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['o'])
            n_trajs += 1
            # resample z every resample paths
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

    def rollout(self, task_idx, policy, max_path_length=np.inf, accum_context=True, deterministic=False, evalu=False):

        observations = []
        actions = []
        rewards = []
        terminals = []
        env = self.eval_tasks[task_idx] if evalu else self.train_tasks[task_idx]
        o, _ = env.reset()
        next_o = None
        path_length = 0

        while path_length < max_path_length:
            a = policy.get_action(o, deterministic=deterministic)
            next_o, r, d, env_info, _ = env.step(a)

            # Convert samples to tensors
            # next_o = torch.tensor(next_o, dtype=torch.float, device=self.device)
            # a = torch.tensor(a, dtype=torch.float, device=self.device)
            # r = torch.tensor(r, dtype=torch.float, device=self.device).view((-1, 1))
            # d = torch.tensor(d, dtype=torch.float, device=self.device).view((-1, 1))

            # a [*,*], o = [*,*,*,*,*,*,*,*], r float, d bool, env_info bool
            # update the agent's current context
            if accum_context:
                # note d and env_info are actually not stored
                policy.update_context([o, a, r, next_o, d, env_info])  # I expect an error here
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            path_length += 1
            o = next_o
            if d:
                break

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)  # this expanditure might be a problem.
            )
        )
        return dict(
            o=observations,  # np.array [[1,2,4,5],[1,2,4,5],[1,2,4,5],...]
            a=actions,
            r=np.array(rewards).reshape(-1, 1),  # [[1],[1.4],[34],...]
            o2=next_observations,
            d=np.array(terminals).reshape(-1, 1),  # [[false],[false],[true],...]
        )
