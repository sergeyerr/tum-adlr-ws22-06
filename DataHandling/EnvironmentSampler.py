# this calls will create rollouts of agent for specific task
import numpy as np
class Sampler(object):

    # this sampler will use the current policy and environment/task to create rollout
    # this will affect the environment

    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length

    def obtain_samples(self, max_samples=np.inf, max_trajs=np.inf, accum_context=True, resample=1):
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
            path = self.rollout( self.env, policy, max_path_length=self.max_path_length, accum_context=accum_context)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # resample z every resample paths
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

    def rollout(self, env, policy, max_path_length, accum_context=True):
        pass

    def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
        """
        The following value for the following keys will be a 2D array, with the
        first dimension corresponding to the time dimension.
         - observations
         - actions
         - rewards
         - next_observations
         - terminals
        The next two elements will be lists of dictionaries, with the index into
        the list being the index into the time
         - agent_infos
         - env_infos
        :param env:
        :param agent:
        :param max_path_length:
        :param accum_context: if True, accumulate the collected context
        :param animated:
        :param save_frames: if True, save video of rollout
        :return:
        """
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        o = env.reset()
        next_o = None
        path_length = 0

        if animated:
            env.render()
        while path_length < max_path_length:
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            # update the agent's current context
            if accum_context:
                agent.update_context([o, a, r, next_o, d, env_info])
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            agent_infos.append(agent_info)
            path_length += 1
            o = next_o
            if animated:
                env.render()
            if save_frames:
                from PIL import Image
                image = Image.fromarray(np.flipud(env.get_image()))
                env_info['frame'] = image
            env_infos.append(env_info)
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
                np.expand_dims(next_o, 0)
            )
        )
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
        )