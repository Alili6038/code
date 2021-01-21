import os
import argparse
import logging

import gym
import numpy as np
import pandas as pd

from policies import LinearPolicy
from envs import NoisyEnv
from utils import set_seed, evaluate
from search import policy_search, RandomSearch, CrossEntropySearch, GeneticAlgorithmSearch, LinearWeightedRewardFunction


_ENVS = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'MountainCarContinuous-v0', 'Pendulum-v0']
_OBS_STD_NOISES = {
    'CartPole-v1': np.array([0.09946977, 0.56992475, 0.09983184, 0.84534435]),
    'MountainCar-v0': np.array([0.08002528, 0.00682503]),
    'Acrobot-v1': np.array([0.16098949, 0.39649959, 0.40995107, 0.58745095, 1.11704985, 1.90546688]),
    'MountainCarContinuous-v0': np.array([0.16280604, 0.01299322]),
    'Pendulum-v0': np.array([0.6354162 , 0.65614156, 3.433707])
}


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-sd', type=int, default=0)
    parser.add_argument('--save_file', '-sf', type=str, default='data.csv')
    parser.add_argument('--force_overwrite', '-f', action='store_true')
    # reward function params
    parser.add_argument('--weight_std_reward', '-wsr', type=float, default=0)
    parser.add_argument('--weight_std_policy', '-wsp', type=float, default=0)
    # env params
    parser.add_argument('--env_name', '-e', type=str, default='CartPole-v1', choices=_ENVS)
    parser.add_argument('--std_obs_scale_train', '-str', type=float, default=0)
    parser.add_argument('--std_obs_scales_test', '-ste', type=list, default=[0, 0.25, 0.5, 1])
    # search method params
    parser.add_argument('--search_method', '-sm', type=str, default='RS', choices=['RS', 'CEM', 'GA'])
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--n_samples', '-ns', type=int, default=10)
    parser.add_argument('--sample_radius', '-sr', type=float, default=0.1)
    # search params
    parser.add_argument('--n_iterations', '-ni', type=int, default=30)
    parser.add_argument('--T', '-T', type=int, default=200)
    parser.add_argument('--n_evals', '-ne', type=int, default=10)
    parser.add_argument('--top_k', '-tk', type=int, default=5)
    args = parser.parse_args()

    if args.force_overwrite:
        os.remove(args.save_file)

    data = vars(args)

    logging.info('Setting seed to {}'.format(args.seed))
    set_seed(args.seed)

    logging.info('Making env {}'.format(args.env_name))
    env_tr = NoisyEnv(args.env_name, _OBS_STD_NOISES[args.env_name] * args.std_obs_scale_train)

    logging.info('Policy Search for {}'.format(args.search_method))
    rew_function = LinearWeightedRewardFunction(args.weight_std_reward, args.weight_std_policy)
    pi_class = LinearPolicy
    if args.search_method == 'RS':
        search_method = RandomSearch(args.learning_rate, args.n_samples, args.sample_radius, env_tr.observation_space, env_tr.action_space, pi_class)
    elif args.search_method == 'CEM':
        search_method = CrossEntropySearch(args.learning_rate, args.n_samples, args.sample_radius, args.top_k, env_tr.observation_space, env_tr.action_space, pi_class)
    elif args.search_method == 'GA':
        search_method = GeneticAlgorithmSearch(args.n_samples, args.sample_radius, args.top_k, env_tr.observation_space, env_tr.action_space, pi_class)
    best_pi, train_rews = policy_search(env_tr, search_method, rew_function, pi_class, args.n_iterations, args.n_evals, args.T)
    data['train_rews'] = repr(train_rews)

    logging.info('Evaluating policy')
    data['final_train_rews'] = repr(evaluate(best_pi, env_tr, args.n_evals, args.T))

    for std_obs_scale_test in args.std_obs_scales_test:
        env_t = NoisyEnv(args.env_name, _OBS_STD_NOISES[args.env_name] * std_obs_scale_test)
        data['final_test_rews_{}'.format(std_obs_scale_test)] = repr(evaluate(best_pi, env_t, args.n_evals, args.T))

    logging.info('Saving to {}'.format(args.save_file))
    df = pd.DataFrame.from_dict([data])
    file_exists = os.path.isfile(args.save_file)
    df.to_csv(args.save_file, header=not file_exists, mode='a')