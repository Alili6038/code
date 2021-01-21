import numpy as np
from IPython import display
import random
import os
import matplotlib
import matplotlib.pyplot as plt

def play_policy(env, pi, T):
    obs = env.reset()
    render_img = plt.imshow(env.render(mode='rgb_array'))
    for _ in range(T):
        render_img.set_data(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)

        action = pi(obs)
        obs, _, _, _ = env.step(action)

def rollout(pi, env, T):
    obs = env.reset()
    rews = []
    for _ in range(T):
        action = pi(obs)
        obs, rew, done, _ = env.step(action)
        rews.append(rew)
        
        if done:
            break
            
    return np.sum(rews)            
    
def evaluate(pi, env, n, T):
    rews = []
    for _ in range(n):
        rews.append(rollout(pi, env, T))
        
    return np.array(rews)

def plot_rews(rews, max_rew):
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(len(rews)), rews, color='blue', label='Learning Progress')
    plt.plot([0, len(rews)], [max_rew]*2, '-', color='red', label='Optimal Policy')
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.ylim([0, max_rew + 100])
    plt.xlim([0, len(rews) - 1])
    plt.legend()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def std_policy_perturbations(pi, pi_class, env, n_evals, T, n_samples=10, eps=0.01):
    cur_params = pi.params
    
    # Sample n_samples number of policies around pi with range of eps 
    delta_params = eps * (np.random.random((n_samples, cur_params.shape[0])) * 2 - 1)
    sampled_policies = [pi_class(env.observation_space, env.action_space, params=cur_params + delta_params[i]) for i in range(n_samples)]
    
    # Evaluate all n_samples policies n_evals number of times by calling the evaluate function
    rews = [np.mean(evaluate(pi, env, n_evals, T)) for pi in sampled_policies]
    
    # return negative standard deviation of rewards
    return np.std(rews)