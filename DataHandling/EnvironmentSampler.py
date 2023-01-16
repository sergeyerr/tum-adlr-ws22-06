# this calls will create rollouts of agent for specific task
import numpy as np
import torch


class Sampler(object):

    # this sampler will use the current policy and environment/task to create rollout
    # this will affect the environment

    def __init__(self, train_tasks, eval_tasks, agent, max_path_length):
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        # don't forget to update policy when it is changed
        self.agent = agent
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
        agent = self.agent
        paths = []
        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = self.rollout(task_idx, agent, max_path_length=self.max_path_length, accum_context=accum_context,
                                deterministic=deterministic, evalu=evalu)
            # save the latent context that generated this trajectory
            path['context'] = agent.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['o'])
            n_trajs += 1
            # resample z every resample paths
            if n_trajs % resample == 0:
                agent.sample_z()
        return paths, n_steps_total

    def rollout(self, task_idx, agent, max_path_length=np.inf, accum_context=True, deterministic=False, evalu=False):

        observations = []
        actions = []
        rewards = []
        terminals = []
        env = self.eval_tasks[task_idx] if evalu else self.train_tasks[task_idx]
        o, _ = env.reset()
        next_o = None
        path_length = 0

        while path_length < max_path_length:
            a = agent.action(o, addNoise=not deterministic)
            next_o, r, d, env_info, _ = env.step(a)

            # a [*,*], o = [*,*,*,*,*,*,*,*], r float, d bool, env_info bool
            # update the agent's current context
            if accum_context:
                # note d and env_info are actually not stored
                agent.update_context([o, a, r, next_o])
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

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                    max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                    accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths
