from custom_lunar_lander import CustomLunarLander
import gymnasium as gym
from EnvironmentUtils import *


                
                
env = DetermenisticResetWrapper(gym.make('LunarLander-v2', continuous=True, enable_wind=True), seed=0)
print(env.wind_idx) 
obs = env.reset()
for i in range(100):
    random_action = env.action_space.sample()
    obs, reward, done, _, _= env.step(random_action)
   # print(env.wind_idx) 
print(env.wind_idx) 
env.reset()
print(env.wind_idx) 