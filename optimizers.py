import numpy as np


class FiniteDifferenceSearch:
    
    def __init__(self, init_pi, pi_class, lr, n_samples, sample_radius):
        self._lr = lr
        self._n_samples = n_samples
        self._sample_radius = sample_radius
        self._pi = init_pi
        
    def sample(self):
        cur_params = self._pi.get_params()
        delta_sample_params = sample_radius * (np.random.random((n_samples, cur_params.shape[0])) * 2 - 1)
        
        return [LinearCartPolePolicy(cur_params + delta_params) for delta_params in delta_sample_params]

    def update(self, cur_rew, sample_rews, sample_pis):
        cur_params = self._pi.get_params()
        
        delta_sample_rews = np.array([sample_rews]) - cur_rew
        delta_sample_params = np.array([sample_pi.get_params() - cur_params for sample_pi in sample_pis])
        
        mean_delta_params = np.mean(delta_sample_params * delta_sample_rews.reshape(-1, 1), axis=0)
        
        next_params = cur_params + lr * mean_delta_params
        self._pi = LinearCartPolePolicy(next_params)
        
    def get_best_pi(self):
        return self._pi

    
class RandomSearch:
    
    def __init__(self, lr, n_samples, sample_radius):
        self._lr = lr
        self._n_samples = n_samples
        self._sample_radius = sample_radius
        self._pi = LinearCartPolePolicy(np.zeros(4))
        
    def sample(self):
        cur_params = self._pi.get_params()
        delta_sample_params = sample_radius * (np.random.random((n_samples, cur_params.shape[0])) * 2 - 1)
        
        return [LinearCartPolePolicy(cur_params + delta_params) for delta_params in delta_sample_params] \
                + [LinearCartPolePolicy(cur_params - delta_params) for delta_params in delta_sample_params]

    def update(self, cur_rew, sample_rews, sample_pis):
        cur_params = self._pi.get_params()
        N = len(sample_rews) // 2
        
        delta_sample_rews = sample_rews[:N] - sample_rews[N:]
        delta_sample_params = np.array([sample_pi.get_params() - cur_params for sample_pi in sample_pis[:N]])
        
        mean_delta_params = np.mean(delta_sample_params * delta_sample_rews.reshape(-1, 1), axis=0)        
        
        next_params = cur_params + lr / N * mean_delta_params
        self._pi = LinearCartPolePolicy(next_params)
        
    def get_best_pi(self):
        return self._pi

    
class CrossEntropySearch:
    
    def __init__(self, lr, n_samples, init_std, top_k):
        self._lr = lr
        self._n_samples = n_samples 
        self._mu_params = np.zeros(4)
        self._sigma_params = np.eye(4) * init_std
        self._top_k = top_k
        
    def sample(self):
        return [LinearCartPolePolicy(sampled_params) 
                for sampled_params in 
                np.random.multivariate_normal(self._mu_params, self._sigma_params, size=self._n_samples)]

    def update(self, cur_rew, sample_rews, sample_pis):
        top_k_idx = np.argsort(sample_rews)[-self._top_k:]
        
        sample_params = np.array([sample_pi.get_params() for sample_pi in sample_pis])[top_k_idx]
        
        self._mu_params = np.mean(sample_params, axis=0)
        self._sigma_params = np.cov(sample_params, rowvar=False)
        
    def get_best_pi(self):
        return LinearCartPolePolicy(self._mu_params)

    
class GeneticAlgorithmSearch:
    
    def __init__(self, n_samples, noise_std, top_k):
        self._n_samples = n_samples
        self._noise_std = noise_std
        self._pis = [LinearCartPolePolicy.sample() for _ in range(n_samples)]
        self._top_k = top_k
        
    def sample(self):
        return self._pis

    def update(self, cur_rew, sample_rews, sample_pis):
        top_k_idx = np.argsort(sample_rews)[-1:-self._top_k-1:-1]
        
        surviving_pis = [self._pis[idx] for idx in top_k_idx]
        self._pis = []
        for _ in range(self._n_samples):
            i, j = np.random.choice(np.arange(self._top_k), size=2, replace=False) 
            
            params = surviving_pis[i].get_params()
            for k in range(len(params)):
                if np.random.rand() > 0.5:
                    params[k] = surviving_pis[j].get_params()[k]
            params += np.random.normal(scale=self._noise_std, size=4)
            self._pis.append(LinearCartPolePolicy(params))
        
    def get_best_pi(self):
        return self._pis[0]