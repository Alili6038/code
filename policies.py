from abc import ABC, abstractmethod
import numpy as np

from gym.spaces import Discrete, Box

class GymPolicy(ABC):

    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    @abstractmethod
    def __call__(self, obs):
        pass


class RandomPolicy(GymPolicy):
    
    def __call__(self, obs):
        return self._action_space.sample()


class LinearPolicy(GymPolicy):
    
    def __init__(self, *args, params=None, **kwargs):
        super().__init__(*args, **kwargs)

        if params is None:
            params = LinearPolicy.sample_params(self._observation_space, self._action_space, zero=True)
        self._params = params

        self._discrete = isinstance(self._action_space, Discrete)
        
    def __call__(self, obs):
        raw_action = self._params.dot(obs)
        if self._discrete:
            raw_action = (np.clip(raw_action, -1, 1) + 1 - 1e-10) / 2
            action = int(raw_action * self._action_space.n)
        else:
            action = raw_action
            if len(self._params.shape) == 1:
                action = np.array([action])

        return action       
    
    @property
    def params(self):
        return self._params.copy()
    
    @params.setter
    def params(self, params):
        self._params = params.copy()    

    @staticmethod
    def sample_params(observation_space, action_space, zero=False):
        zero_mult = 0 if zero else 1
        if isinstance(action_space, Discrete) or \
            isinstance(action_space, Box) and action_space.shape[0] == 1:
            params = np.random.rand(observation_space.shape[0]) * zero_mult
        else:
            params = np.random.rand(observation_space.shape[0], action_space.shape[0]) * zero_mult
        return params
    
    @staticmethod
    def sample(observation_space, action_space, zero=False):
        params = LinearPolicy.sample_params(observation_space, action_space, zero=zero)       
        return LinearPolicy(observation_space, action_space, params=params)