from collections import deque
from random import choices
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.queue = deque(maxlen=capacity)

    def record(self, obs, action, reward, new_obs, done):
        # for the case where we record a path not a single transition
        if len(obs.shape) == 2:
            for o, a, r, no, d in zip(obs, action, reward, new_obs, done):
                entry = (o, a, r, no, d)
                self.queue.append(entry)
        else:
            entry = (obs, action, reward, new_obs, done)

        self.queue.append(entry)

    def get_batch(self, batch_size):
        sample = choices(self.queue, k=batch_size)
        out_dict = {'o':[],'a':[],'r':[],'o2':[],'d':[]}
        for o, a, r, o2, d in sample:
            out_dict['o'].append(o)
            out_dict['a'].append(a)
            out_dict['r'].append(r)
            out_dict['o2'].append(o2)
            out_dict['d'].append(d)
        out_dict['o'] = np.array(out_dict['o'])
        out_dict['a'] = np.array(out_dict['a'])
        out_dict['r'] = np.array(out_dict['r'])
        out_dict['o2'] = np.array(out_dict['o2'])
        out_dict['d'] = np.array(out_dict['d'])
        return out_dict

    # TODO
    def add_path(self, path):
        for i, (obs, action, reward, next_obs, terminal)\
                in enumerate(zip(path["o"], path["a"], path["r"], path["o2"], path["d"])):
            self.record(obs, action, reward, next_obs, terminal)

    def clear(self):
        self.queue.clear()

    def size(self):
        return len(self.queue)


class MultiTaskReplayBuffer(object):

    def __init__(self, capacity, num_tasks):
        self.capacity = capacity
        self.num_tasks = num_tasks
        self.task_buffers = dict([(idx, ReplayBuffer(capacity)) for idx in num_tasks])

    def record(self, task_id, obs, action, reward, new_obs, done):
        self.task_buffers[task_id].record(obs, action, reward, new_obs, done)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def random_batch(self, task_id, batch_size):
        out_dict = self.task_buffers[task_id].get_batch(batch_size)
        return out_dict

    # TODO sampling context and sampling sac could be moved here instead
    # so you can call encoder_replay_buffer.sample_random_batch(indices, sample_context=True)
    def sample_random_batch(self, task_indices, batch_size, sample_context=False, use_next_obs_in_context=False):
        # out dimensions: [(num tasks, batch size, feature_dim) for each feature (observation, action, reward ...)]
        # this is done to ensure that task_indices is iterable even if its just one integer
        if not hasattr(task_indices, '__iter__'):
            task_indices = [task_indices]
        out = []
        for idx in task_indices:
            out.append(self.random_batch(idx, batch_size))
        # out has now following form:[dict1, dict2, ...] with num_tasks many dicts
        # each dict is: {"o":[[*,*,*,*,*], [*,*,*,*,*], [*,*,*,*,*]], "a":[[*,*], [*,*], [*,*]], "r"...]
        # each key has a nested list as value with dim(batch_size, feature_dim)
        outer_list = []
        inner_list = []
        for dictionary in out:
            inner_list.clear()
            for k, v in dictionary.items():
                inner_list.append(torch.tensor(v))
            outer_list.append(inner_list)
        # outer_list has now following structure: [[tensor(batch_size, obs_dim), tensor(batch_size, act_dim)], ...]
        outer_list = [[x[i][None, ...] for x in outer_list] for i in range(len(outer_list[0]))]
        # outer_list now consists of lists each holding same feature from different tasks as tensors
        # the [None, ...] operations makes sure that in the following contactenation we receive a 3 dimensonal tensor
        out = [torch.cat(x, dim=0) for x in outer_list]
        # out dimensions: [(num tasks, batch size, feature_dim) for each feature (observation, action, reward ...)]
        # if we are sampling the context the output dimension is
        # tensor(num_tasks, batch_size, all_feature_dim_concatenated)
        # we neglect the terminals (thus :-1) and or the next_obs (:-2)
        if sample_context:
            if use_next_obs_in_context:
                out = torch.cat(out[:-1], dim=2)
            else:
                out = torch.cat(out[:-2], dim=2)
        return out

    def clear_buffer(self, task_id):
        self.task_buffers[task_id].clear()


