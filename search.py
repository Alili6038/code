from abc import ABC, abstractmethod
import numpy as np
from tqdm import trange

from utils import evaluate, std_policy_perturbations


class PolicySearch(ABC):

    def __init__(self, observation_space, action_space, pi_class):
        self._observation_space = observation_space
        self._action_space = action_space
        self._pi_class = pi_class
        self._pi = self._pi_class.sample(self._observation_space, self._action_space, zero=True)

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def update(self, cur_rew, sample_rews, sample_pis):
        pass

    @abstractmethod
    def get_best_pi(self):
        pass 


class FiniteDifferenceSearch(PolicySearch):
    
    def __init__(self, *args, lr, n_samples, sample_radius, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr = lr
        self._n_samples = n_samples
        self._sample_radius = sample_radius
        
    def sample(self):
        cur_params = self._pi.params
        delta_sample_params = self._sample_radius * (np.random.random((self._n_samples, cur_params.shape[0])) * 2 - 1)
        
        return [self._pi_class(self._observation_space, self._action_space, params=cur_params + delta_params) for delta_params in delta_sample_params]

    def update(self, cur_rew, sample_rews, sample_pis):
        cur_params = self._pi.params
        
        delta_sample_rews = np.array([sample_rews]) - cur_rew
        delta_sample_params = np.array([sample_pi.params - cur_params for sample_pi in sample_pis])
        
        mean_delta_params = np.mean(delta_sample_params * delta_sample_rews.reshape(-1, 1), axis=0)
        
        next_params = cur_params + self._lr * mean_delta_params
        self._pi = self._pi_class(self._observation_space, self._action_space, params=next_params)
        
    def get_best_pi(self):
        return self._pi

    
class RandomSearch(PolicySearch):
    
    def __init__(self, lr, n_samples, sample_radius, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr = lr
        self._n_samples = n_samples
        self._sample_radius = sample_radius
        
    def sample(self):
        cur_params = self._pi.params
        delta_sample_params = self._sample_radius * (np.random.random((self._n_samples, cur_params.shape[0])) * 2 - 1)
        
        return [self._pi_class(self._observation_space, self._action_space, params=cur_params + delta_params) for delta_params in delta_sample_params] \
                + [self._pi_class(self._observation_space, self._action_space, params=cur_params - delta_params) for delta_params in delta_sample_params]

    def update(self, cur_rew, sample_rews, sample_pis):
        cur_params = self._pi.params
        N = len(sample_rews) // 2
        
        delta_sample_rews = sample_rews[:N] - sample_rews[N:]
        delta_sample_params = np.array([sample_pi.params - cur_params for sample_pi in sample_pis[:N]])
        
        mean_delta_params = np.mean(delta_sample_params * delta_sample_rews.reshape(-1, 1), axis=0)        
        
        next_params = cur_params + self._lr / N * mean_delta_params
        self._pi = self._pi_class(self._observation_space, self._action_space, params=next_params)
        
    def get_best_pi(self):
        return self._pi

    
class CrossEntropySearch(PolicySearch):
    
    def __init__(self, lr, n_samples, init_std, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr = lr
        self._n_samples = n_samples 
        self._mu_params = np.zeros(4)
        self._sigma_params = np.eye(4) * init_std
        self._top_k = top_k
        
    def sample(self):
        return [self._pi_class(self._observation_space, self._action_space, params=sampled_params) 
                for sampled_params in 
                np.random.multivariate_normal(self._mu_params, self._sigma_params, size=self._n_samples)]

    def update(self, cur_rew, sample_rews, sample_pis):
        top_k_idx = np.argsort(sample_rews)[-self._top_k:]
        
        sample_params = np.array([sample_pi.params for sample_pi in sample_pis])[top_k_idx]
        
        self._mu_params = np.mean(sample_params, axis=0)
        self._sigma_params = np.cov(sample_params, rowvar=False)
        
    def get_best_pi(self):
        return self._pi_class(self._observation_space, self._action_space, params=self._mu_params)

    
class GeneticAlgorithmSearch(PolicySearch):
    
    def __init__(self, n_samples, noise_std, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_samples = n_samples
        self._noise_std = noise_std
        self._pis = [self._pi_class.sample(self._observation_space, self._action_space) for _ in range(n_samples)]
        self._top_k = top_k
        
    def sample(self):
        return self._pis

    def update(self, cur_rew, sample_rews, sample_pis):
        top_k_idx = np.argsort(sample_rews)[-1:-self._top_k-1:-1]
        
        surviving_pis = [self._pis[idx] for idx in top_k_idx]
        self._pis = []
        for _ in range(self._n_samples):
            i, j = np.random.choice(np.arange(self._top_k), size=2, replace=False) 
            
            params = surviving_pis[i].params
            for k in range(len(params)):
                if np.random.rand() > 0.5:
                    params[k] = surviving_pis[j].params[k]
            params += np.random.normal(scale=self._noise_std, size=params.shape)
            self._pis.append(self._pi_class(self._observation_space, self._action_space, params=params))
        
    def get_best_pi(self):
        return self._pis[0]


class LinearWeightedRewardFunction:

    def __init__(self, weight_std_reward, weight_std_policy):
        self._weight_std_reward = weight_std_reward
        self._weight_std_policy = weight_std_policy
        self._weights = np.array([1, weight_std_reward, weight_std_policy])
        
    def __call__(self, pi, pi_class, env, n_evals, T):
        rews = evaluate(pi, env, n_evals, T=T)

        rews_vec = np.zeros(3)
        rews_vec[0] = np.mean(rews)        

        if self._weight_std_reward > 0:
            rews_vec[1] = np.std(rews)
        if self._weight_std_policy > 0:
            rews_vec[2] = std_policy_perturbations(pi, pi_class, env, n_evals, T)
        
        return rews_vec.dot(self._weights)


def policy_search(env, opt, rew_function, pi_class, n_iterations, n_evals, T):
    rews = []
    
    for _ in trange(n_iterations):
        best_pi = opt.get_best_pi()
        cur_rew = rew_function(best_pi, pi_class, env, n_evals, T)
        
        rews.append(cur_rew)
        
        sample_pis = opt.sample()
        sample_rews = np.array([rew_function(pi, pi_class, env, n_evals, T) for pi in sample_pis])
        opt.update(cur_rew, sample_rews, sample_pis)        
        
    return best_pi, rews