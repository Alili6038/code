import os
import argparse
import logging

import gym
import numpy as np
from tqdm import trange

from policies import RandomPolicy

_ENVS = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'MountainCarContinuous-v0', 'Pendulum-v0']

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str, default='CartPole-v1', choices=_ENVS)
    args = parser.parse_args()

    logging.info('Making env {}'.format(args.env_name))
    env = gym.make(args.env_name)
    pi = RandomPolicy(env.observation_space, env.action_space)

    all_obs = []
    for _ in trange(100):
        obs = env.reset()
        all_obs.append(obs)
        while True:
            obs, _, done, _ = env.step(pi(obs))
            all_obs.append(obs)

            if done: break

    std_obs = np.std(all_obs, axis=0)
    print(repr(std_obs))
    import IPython; IPython.embed(); exit(0)
    