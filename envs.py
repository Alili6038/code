import gym
import numpy as np

class NoisyEnv:
    
    def __init__(self, env_name, std_obs):
        self._env = gym.make(env_name)
        self._std_obs = std_obs

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _add_obs_noise(self, obs):
        obs += np.random.normal(scale=1, size=obs.shape) * self._std_obs
        return obs

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        
        obs = self._add_obs_noise(obs)      
        return obs, rew, done, info
        
    def reset(self):
        obs = self._env.reset()
        obs = self._add_obs_noise(obs)
        return obs