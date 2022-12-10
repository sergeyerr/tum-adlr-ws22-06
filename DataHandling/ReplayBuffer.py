from collections import deque
from random import choices
import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.queue = deque(maxlen=capacity)

    def record(self, obs, action, reward, new_obs, done):
        entry = (obs, action, reward, new_obs, done)
        self.queue.append(entry)

    def get_batch(self, batch_size):
        sample = choices(self.queue, k=batch_size)
        out_dict = {'o':[],'a':[],'r':[],'o2':[],'d':[]}
        # TODO I think this is wrong. We do not need the loop
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

    def random_batch(self, task_id, batch_size):
        self.task_buffers[task_id].get_batch(batch_size)

    # TODO sampling context and sampling sac could be moved here instead
    # so you can call encoder_replay_buffer.sample_random_batch(indices, sample_context=True)
    def sample_random_batch(self, task_indices, sample_context=False):
        pass


    def clear_buffer(self, task_id):
        self.task_buffers[task_id].clear()


