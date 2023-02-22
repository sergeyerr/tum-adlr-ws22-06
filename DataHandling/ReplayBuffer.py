from collections import deque
from random import choices, choice
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
        self.task_buffers = dict([(idx, ReplayBuffer(capacity)) for idx in range(num_tasks)])

    def record(self, task_id, obs, action, reward, new_obs, done):
        self.task_buffers[task_id].record(obs, action, reward, new_obs, done)

    def add_paths(self, task, paths):
        for path in paths:
            self.task_buffers[task].add_path(path)

    def random_batch(self, task_id, batch_size):
        out_dict = self.task_buffers[task_id].get_batch(batch_size)
        return out_dict

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
        # tensor(num_tasks, batch_size, all_feature_dim_concatenated)  all_feature_dim_concatenated = 11
        # we neglect the terminals (thus :-1) and or the next_obs (:-2)
        if sample_context:
            if use_next_obs_in_context:
                out = torch.cat(out[:-1], dim=2)
            else:
                out = torch.cat(out[:-2], dim=2)
        return out

    def clear_buffer(self, task_id):
        self.task_buffers[task_id].clear()
        

# For saving runs separatelt and get history of observations for samples
# DON'T FORGET, THAT RUNS CAPACITY IS NOT THE SAME AS OBSERVATIONS CAPACITY
class RunsReplayBuffer(object):
    def __init__(self, task_capacity, context_size = 64):
        self.queue = deque(maxlen=task_capacity)
        self.context_size = context_size

    def record_trajectory(self, obs, action, reward, new_obs, done):
        # for the case where we record a path not a single transition
        self.queue.append((obs, action, reward, new_obs, done))
        # else:
        #     raise ValueError("Only full trajectories are accepted")
            
            

    def get_batch(self, batch_size):
        run_samples = choices(self.queue, k=batch_size)
        context_samples = []
        observation_samples = {'o':[],'a':[],'r':[],'o2':[],'d':[]}
        
        for run_sample in run_samples:
            obs_idx = choice(range(len(run_sample[0])))
            context_dict = dict()
            context_dict['o'] = np.array(run_sample[0][max(obs_idx - self.context_size, 0):obs_idx])
            context_dict['a'] = np.array(run_sample[1][max(obs_idx - self.context_size, 0):obs_idx])
            context_dict['r'] = np.array(run_sample[2][max(obs_idx - self.context_size, 0):obs_idx])
            #context_dict['o2'] = np.array(run_sample[3][obs_idx - self.context_size:obs_idx])
            #context_dict['d'] = np.array(run_sample[4][obs_idx - self.context_size:obs_idx])
            observation_samples['o'].append(run_sample[0][obs_idx])
            observation_samples['a'].append(run_sample[1][obs_idx])
            observation_samples['r'].append(run_sample[2][obs_idx])
            observation_samples['o2'].append(run_sample[3][obs_idx])
            observation_samples['d'].append(run_sample[4][obs_idx])
            context_samples.append(context_dict)
        
        observation_samples['o'] = np.array(observation_samples['o'])
        observation_samples['a'] = np.array(observation_samples['a'])
        observation_samples['r'] = np.array(observation_samples['r'])
        observation_samples['o2'] = np.array(observation_samples['o2'])
        observation_samples['d'] = np.array(observation_samples['d'])
        return context_samples, observation_samples


    def clear(self):
        self.queue.clear()

    def size(self):
        return len(self.queue)
    
    
class MultiTaskRunsReplayBuffer(object):

    def __init__(self, capacity, num_tasks, context_size = 64):
        self.capacity = capacity
        self.num_tasks = num_tasks
        self.task_buffers = dict([(idx, RunsReplayBuffer(capacity, context_size)) for idx in range(num_tasks)])

    def add_paths(self, task_idx, paths):
        for path in paths:
            self.record_trajectory(task_idx, path['o'], path['a'], path['r'], path['o2'], path['d'])

    def record_trajectory(self, task_id, obs, action, reward, new_obs, done):
        self.task_buffers[task_id].record_trajectory(obs, action, reward, new_obs, done)


    def random_batch(self, task_id, batch_size):
        context, observations = self.task_buffers[task_id].get_batch(batch_size)
        return context, observations

    def sample_random_batch(self, task_indices, batch_size):
        # out dimensions: [(num tasks, batch size, feature_dim) for each feature (observation, action, reward ...)]
        # this is done to ensure that task_indices is iterable even if its just one integer
        if not hasattr(task_indices, '__iter__'):
            task_indices = [task_indices]
        out = []
        out_context = []
        for idx in task_indices:
            task_contexts, observations = self.random_batch(idx, batch_size)
            out.append(observations)
            out_context.append(task_contexts)
        # out context shape: (num_tasks list, batch_size x dict x context length x dim (reward/action/obs))
        
        # out has now following form:[dict1, dict2, ...] with num_tasks many dicts
        # each dict is: {"o":[[*,*,*,*,*], [*,*,*,*,*], [*,*,*,*,*]], "a":[[*,*], [*,*], [*,*]], "r"...]
        # each key has a nested list as value with dim(batch_size, feature_dim)
        per_task_list = []
        inner_list = []
        # make list of torch tensors, same structure as out
        context_list = []
        
        
        for dictionary in out:
            inner_list.clear()
            for k, v in dictionary.items():
                inner_list.append(torch.tensor(v))
            per_task_list.append(inner_list)
            
            
            
            
        # outer_list has now following structure: [[tensor(batch_size, obs_dim), tensor(batch_size, act_dim)], ...]
        per_task_list = [[x[i][None, ...] for x in per_task_list] for i in range(len(per_task_list[0]))]
        # outer_list now consists of lists each holding same feature from different tasks as tensors
        # the [None, ...] operations makes sure that in the following contactenation we receive a 3 dimensonal tensor
        out = [torch.cat(x, dim=0) for x in per_task_list]
        
        # out dimensions: [(num tasks, batch size, feature_dim) for each feature (observation, action, reward ...)]
        # if we are sampling the context the output dimension is
        # tensor(num_tasks, batch_size, all_feature_dim_concatenated)  all_feature_dim_concatenated = 11
        
        
        for task_contexts in out_context:
          per_task_list = [] 
          for dictionary in task_contexts:
            o = torch.tensor(dictionary['o'])
            a = torch.tensor(dictionary['a'])
            r = torch.tensor(dictionary['r'])
            
            per_task_list.append(torch.cat((o, a, r), dim=1))
          # not need to concatenate contexts for one task! 
          # final shape tasks (list) X batch_size (list) X tensor (length of context X feature_dim)
          context_list.append(per_task_list)
              
            
        return out, context_list

    def clear_buffer(self, task_id):
        self.task_buffers[task_id].clear()

