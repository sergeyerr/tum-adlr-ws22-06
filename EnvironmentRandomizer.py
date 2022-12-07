import random
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


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
    
    


class LunarEnvFabric:
    def __init__(self, pass_env_params,
                random_type, gravity_lower, gravity_upper, wind_probability, wind_power_lower, wind_power_upper,
                turbulence_power_lower, turbulence_power_upper, 
                default_gravity=-10, default_wind=False, default_wind_power=15, default_turbulence_power=1.5, render_mode=None):
        self.random_type = random_type
        self.pass_env_params = pass_env_params
        if random_type != 'Uniform' and random_type != "Fixed":
            raise NotImplementedError("Only uniform randomization is supported")
        self.gravity_lower = gravity_lower
        self.gravity_upper = gravity_upper
        self.wind_probability = wind_probability
        self.wind_power_lower = wind_power_lower
        self.wind_power_upper = wind_power_upper
        self.turbulence_power_lower = turbulence_power_lower
        self.turbulence_power_upper = turbulence_power_upper
        self.default_gravity = default_gravity
        self.default_wind = default_wind
        self.default_wind_power = default_wind_power
        self.default_turbulence_power = default_turbulence_power
        self.render_mode_pass = render_mode
        
    def generate_env(self):
        if self.random_type == "Uniform":
            gravity, enable_wind, wind_power, turbulence_power = self.get_uniform_parameters()
        elif self.random_type == "Fixed":
            gravity, enable_wind, wind_power, turbulence_power = self.get_default_parameters()
        else:
            raise NotImplementedError("Only uniform randomization is supported")
        
        return StateInjectorWrapper(gym.make('LunarLander-v2', continuous=True,
                                  render_mode=self.render_mode_pass, gravity=gravity, 
                                  enable_wind=enable_wind, wind_power=wind_power, turbulence_power=turbulence_power), self.pass_env_params, 
                                    self.gravity_lower, self.gravity_upper, self.wind_power_lower,
                                    self.wind_power_upper, self.turbulence_power_lower, self.turbulence_power_upper)
        
    def get_uniform_parameters(self):
        gravity = random.uniform(self.gravity_lower, self.gravity_upper)
        wind = random.random() < self.wind_probability
        wind_power = random.uniform(self.wind_power_lower, self.wind_power_upper)
        turbulence_power = random.uniform(self.turbulence_power_lower, self.turbulence_power_upper)
        return gravity, wind, wind_power, turbulence_power
    
    def get_default_parameters(self):
        return self.default_gravity, self.default_wind, self.default_wind_power, self.default_turbulence_power

    