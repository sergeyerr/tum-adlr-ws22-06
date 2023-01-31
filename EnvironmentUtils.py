import random
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class ValidationHypercube:
    def __init__(self, points_per_axis=3, check_no_wind=True,  **env_params):
        '''
        env_params - config dict, with bounds for gravity, wind and turbulence;
        points_per_axis - number of points per dimenstion of the hypercube
        
        will generate a hypercube of points, where each point is a tuple of (gravity, wind, wind_power, turbulence_power)
        '''
        self.gravity_lower, self.gravity_upper = env_params['gravity_lower'], env_params['gravity_upper']
        self.wind_power_lower, self.wind_power_upper = env_params['wind_power_lower'], env_params['wind_power_upper']
        self.turbulence_power_lower, self.turbulence_power_upper = env_params['turbulence_power_lower'], env_params['turbulence_power_upper']
        self.points_per_axis = points_per_axis
        self.check_no_wind = check_no_wind
        self.eval_ood_gravity_lower = env_params['out_of_distribution']['eval']['gravity_lower']
        self.eval_ood_gravity_upper = env_params['out_of_distribution']['eval']['gravity_upper']
        self.eval_ood_wind_probability = env_params['out_of_distribution']['eval']['wind_probability']
        self.eval_ood_wind_power_lower = env_params['out_of_distribution']['eval']['wind_power_lower']
        self.eval_ood_wind_power_upper = env_params['out_of_distribution']['eval']['wind_power_upper']
        self.eval_ood_turbulence_power_lower = env_params['out_of_distribution']['eval']['turbulence_power_lower']
        self.eval_ood_turbulence_power_upper = env_params['out_of_distribution']['eval']['turbulence_power_upper']
        
    
    def get_points(self):
        no_wind_ponts = []
        wind_points = []
        gravity_linspace = np.linspace(self.gravity_lower, self.gravity_upper, self.points_per_axis) 
        wind_linspace= np.linspace(self.wind_power_lower, self.wind_power_upper, self.points_per_axis)
        turbulence_linspace = np.linspace(self.turbulence_power_lower, self.turbulence_power_upper, self.points_per_axis)   
        no_wind_points = [(g, 0, 0, 0) for g in gravity_linspace]
        wind_points = [(g, 1, w, t) for g in gravity_linspace for w in wind_linspace for t in turbulence_linspace]

        ood_no_wind_ponts = []
        ood_wind_points = []
        ood_gravity_linspace = np.linspace(self.eval_ood_gravity_lower, self.eval_ood_gravity_upper,
                                           self.points_per_axis)
        ood_wind_linspace = np.linspace(self.eval_ood_wind_power_lower, self.eval_ood_wind_power_upper,
                                        self.points_per_axis)
        ood_turbulence_linspace = np.linspace(self.eval_ood_turbulence_power_lower,
                                              self.eval_ood_turbulence_power_upper, self.points_per_axis)
        ood_no_wind_points = [(g, 0, 0, 0) for g in ood_gravity_linspace]
        ood_wind_points = [(g, 1, w, t) for g in ood_gravity_linspace for w in ood_wind_linspace for t in ood_turbulence_linspace]

        if self.check_no_wind:
            return no_wind_ponts + wind_points, ood_no_wind_ponts + ood_wind_points
        return wind_points, ood_wind_points


class DetermenisticResetWrapper(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        
        
    def reset(self, **kwargs):
        
        #return self.env.reset(**kwargs, seed=self.seed)
        return self.env.reset(seed=self.seed)


class StateInjectorWrapper(gym.Wrapper):
    def __init__(self, env, pass_env_params,
                 gravity_lower, gravity_upper, wind_power_lower, wind_power_upper, turbulence_power_lower, turbulence_power_upper):
        
        super().__init__(env)
        self.pass_env_params = pass_env_params
        self.gravity_lower, self.gravity_upper = gravity_lower, gravity_upper
        self.wind_power_lower, self.wind_power_upper = wind_power_lower, wind_power_upper
        self.turbulence_power_lower, self.turbulence_power_upper = turbulence_power_lower, turbulence_power_upper
        
        if pass_env_params:
            #self.env.observation_space = (self.env.observation_space.shape[0] + 4,)
            self.env.observation_space = Box(low=np.concatenate((self.env.observation_space.low, np.array([0, 0, 0, 0]))), 
                                             high=np.concatenate((self.env.observation_space.high, np.array([1, 1, 1, 1]))))

    def reset(self, **kwargs):
        #print(self.env.gravity, self.gravity)
        #self.env = gym.make('LunarLander-v2', continuous=True, render_mode=self.render_mode, gravity=gravity, enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power)
        obs, info = self.env.reset(**kwargs)
        if self.pass_env_params:
            obs = self.modify_observation_with_params(obs)
        return obs, info
    
    def modify_observation_with_params(self, obs):
        gravity, enable_wind, wind_power, turbulence_power = self.env.gravity, self.env.enable_wind, self.env.wind_power, self.env.turbulence_power
        gravity = (gravity - self.gravity_lower) / (self.gravity_upper - self.gravity_lower)
        enable_wind = 1 if enable_wind else 0
        wind_power = (wind_power - self.wind_power_lower) / (self.wind_power_upper - self.wind_power_lower)
        turbulence_power = (turbulence_power - self.turbulence_power_lower) / (self.turbulence_power_upper - self.turbulence_power_lower)

        return np.concatenate((obs, [gravity, enable_wind, wind_power, turbulence_power]))
        
    def step(self, action):
        next_state, reward, done, info, _ = self.env.step(action)
        if self.pass_env_params:
            next_state = self.modify_observation_with_params(next_state)
        return next_state, reward, done, info, _
    
    
class LunarEnvFixedFabric:
    '''Parent class for all fabrics'''
    def __init__(self, pass_env_params, render_mode=None, **env_params):
        self.random_type = env_params['random_type']
        self.pass_env_params = pass_env_params
        if self.random_type != 'Uniform' and self.random_type != 'Fixed':
            raise NotImplementedError("Only uniform randomization is supported")
        self.gravity_lower = env_params['gravity_lower']
        self.gravity_upper = env_params['gravity_upper']
        self.wind_probability = env_params['wind_probability']
        self.wind_power_lower = env_params['wind_power_lower']
        self.wind_power_upper = env_params['wind_power_upper']
        self.turbulence_power_lower = env_params['turbulence_power_lower']
        self.turbulence_power_upper = env_params['turbulence_power_upper']
        self.default_gravity = env_params['default_gravity']
        self.default_wind = env_params['default_wind']
        self.default_wind_power = env_params['default_wind_power']
        self.default_turbulence_power = env_params['default_turbulence_power']
        self.render_mode_pass = render_mode
        self.deterministic_reset = env_params['deterministic_reset']
        self.seed = env_params['seed'] if 'seed' in env_params else 42
        self.train_ood_gravity_lower = env_params['out_of_distribution']['train']['gravity_lower']
        self.train_ood_gravity_upper = env_params['out_of_distribution']['train']['gravity_upper']
        self.train_ood_wind_probability = env_params['out_of_distribution']['train']['wind_probability']
        self.train_ood_wind_power_lower = env_params['out_of_distribution']['train']['wind_power_lower']
        self.train_ood_wind_power_upper = env_params['out_of_distribution']['train']['wind_power_upper']
        self.train_ood_turbulence_power_lower = env_params['out_of_distribution']['train']['turbulence_power_lower']
        self.train_ood_turbulence_power_upper = env_params['out_of_distribution']['train']['turbulence_power_upper']
        self.eval_ood_gravity_lower = env_params['out_of_distribution']['eval']['gravity_lower']
        self.eval_ood_gravity_upper = env_params['out_of_distribution']['eval']['gravity_upper']
        self.eval_ood_wind_probability = env_params['out_of_distribution']['eval']['wind_probability']
        self.eval_ood_wind_power_lower = env_params['out_of_distribution']['eval']['wind_power_lower']
        self.eval_ood_wind_power_upper = env_params['out_of_distribution']['eval']['wind_power_upper']
        self.eval_ood_turbulence_power_lower = env_params['out_of_distribution']['eval']['turbulence_power_lower']
        self.eval_ood_turbulence_power_upper = env_params['out_of_distribution']['eval']['turbulence_power_upper']
        
    def generate_env(self):
        env = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                  render_mode=self.render_mode_pass, gravity=self.default_gravity , 
                                  enable_wind=self.default_wind, wind_power=self.default_wind_power, turbulence_power=self.default_turbulence_power), self.pass_env_params, 
                                    self.gravity_lower, self.gravity_upper, self.wind_power_lower,
                                    self.wind_power_upper, self.turbulence_power_lower, self.turbulence_power_upper)
        if self.deterministic_reset:
            return DetermenisticResetWrapper(env, self.seed)
        else:
            return env


class LunarEnvRandomFabric(LunarEnvFixedFabric):
    def __init__(self, pass_env_params, render_mode=None, **env_params):
        super().__init__(pass_env_params, render_mode, **env_params)
        
    def generate_env(self):
        if self.random_type == "Uniform":
            gravity, enable_wind, wind_power, turbulence_power, ood_gravity, ood_wind, ood_wind_power,\
                ood_turbulence_power = self.get_uniform_parameters()
        else:
            raise NotImplementedError("Only uniform randomization is supported")
        
        env = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                            render_mode=self.render_mode_pass, gravity=gravity, 
                            enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power), False,
                            self.gravity_lower, self.gravity_upper, self.wind_power_lower,
                            self.wind_power_upper, self.turbulence_power_lower, self.turbulence_power_upper)

        env_with_params = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                                        render_mode=self.render_mode_pass, gravity=gravity,
                                                        enable_wind=enable_wind, wind_power=wind_power,
                                                        turbulence_power=turbulence_power), True,
                                               self.gravity_lower, self.gravity_upper, self.wind_power_lower,
                                               self.wind_power_upper, self.turbulence_power_lower,
                                               self.turbulence_power_upper)

        env_ood = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                                render_mode=self.render_mode_pass, gravity=ood_gravity,
                                                enable_wind=ood_enable_wind, wind_power=ood_wind_power,
                                                turbulence_power=ood_turbulence_power), True,
                                       self.train_ood_gravity_lower, self.train_ood_gravity_upper,
                                       self.train_ood_wind_power_lower, self.train_ood_wind_power_upper,
                                       self.train_ood_turbulence_power_lower, self.train_ood_turbulence_power_upper)
        
        if self.deterministic_reset:
            return DetermenisticResetWrapper(env, self.seed), DetermenisticResetWrapper(env_with_params, self.seed),\
                DetermenisticResetWrapper(env_ood, self.seed)
        else:
            return env, env_with_params, env_ood
        
    def get_uniform_parameters(self):
        gravity = random.uniform(self.gravity_lower, self.gravity_upper)
        wind = random.random() < self.wind_probability
        wind_power = random.uniform(self.wind_power_lower, self.wind_power_upper)
        turbulence_power = random.uniform(self.turbulence_power_lower, self.turbulence_power_upper)

        ood_gravity = random.uniform(self.train_ood_gravity_lower, self.train_ood_gravity_upper)
        ood_wind = random.random() < self.train_ood_wind_probability
        ood_wind_power = random.uniform(self.train_ood_wind_power_lower, self.train_ood_wind_power_upper)
        ood_turbulence_power = random.uniform(self.train_ood_turbulence_power_lower, self.train_ood_turbulence_power_upper)
        return gravity, wind, wind_power, turbulence_power, ood_gravity, ood_wind, ood_wind_power, ood_turbulence_power
    


class LunarEnvHypercubeFabric(LunarEnvFixedFabric):
    '''Fabric for generating environments with parameters from hypercube grid 
    '''
    def __init__(self, pass_env_params, render_mode=None, points_per_axis = 3, check_without_wind = True, **env_params):
        super().__init__(pass_env_params, render_mode, **env_params)
        self.test_parameters = ValidationHypercube(points_per_axis=points_per_axis, check_no_wind=check_without_wind, **env_params).get_points()
        self.iter = 0
    
    
    def generate_env(self):
        
        gravity, enable_wind, wind_power, turbulence_power, ood_gravity, ood_enable_wind, ood_wind_power,\
            ood_turbulence_power = self.test_parameters[self.iter]
        self.iter += 1
        self.iter = self.iter % len(self.test_parameters)
        env = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                  render_mode=self.render_mode_pass, gravity=gravity, 
                                  enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power), False,
                                    self.gravity_lower, self.gravity_upper, self.wind_power_lower,
                                    self.wind_power_upper, self.turbulence_power_lower, self.turbulence_power_upper)

        env_with_params = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                            render_mode=self.render_mode_pass, gravity=gravity,
                                            enable_wind=enable_wind, wind_power=wind_power,
                                            turbulence_power=turbulence_power), True,
                                   self.gravity_lower, self.gravity_upper, self.wind_power_lower,
                                   self.wind_power_upper, self.turbulence_power_lower, self.turbulence_power_upper)

        env_ood = StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                                render_mode=self.render_mode_pass, gravity=ood_gravity,
                                                enable_wind=ood_enable_wind, wind_power=ood_wind_power,
                                                turbulence_power=ood_turbulence_power), True,
                                       self.train_ood_gravity_lower, self.train_ood_gravity_upper,
                                       self.train_ood_wind_power_lower, self.train_ood_wind_power_upper,
                                       self.train_ood_turbulence_power_lower, self.train_ood_turbulence_power_upper)

        if self.deterministic_reset:
            return DetermenisticResetWrapper(env, self.seed), DetermenisticResetWrapper(env_with_params, self.seed), \
                DetermenisticResetWrapper(env_ood, self.seed)
        else:
            return env, env_with_params, env_ood
        
        
    def number_of_test_points(self):
        return len(self.test_parameters) / 2
        

    