#!/usr/bin/python3

import argparse
import json
import random
import gym
import math
import numpy as np
from keras.models import load_model

def run_ep(env, model=None):
    """Run an episode in the given environment and return states, actions, and rewards.

    model is assumed to have a predict() method. If no model is given, samples random actions.

    Returns 1D np arrays obs, acs, rews that contain one datapoint for each timestep taken.

    """
    state = env.reset()
    # some of our models use one state, some backtrack by some number of states in a rolling window
    # here we initially use several copies of the initial state, then roll the window
    if model:
        n_inputs = model.input_shape[1] // len(state)
        model_input = np.array(list(state) * n_inputs)
    done = False
    obs = []
    acs = []
    rews = []
    while not done:
        if model:
            model_input = np.concatenate((model_input[len(state):], state))
            action = model.predict(np.reshape(np.array(model_input), (1, -1)))
        else:
            action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        obs.append(state)
        acs.append(action)
        rews.append(reward)
        state = next_state

    return np.array(obs), np.array(acs), np.array(rews)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='Gym Environment to execute', required=True)
    parser.add_argument('--model', help='Trained keras model to predict actions from states', required=True)
    parser.add_argument('--eps', help='Number of episodes to evaluate', type=int, default=100)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-o', '--out', help='File to save results to')
    args = parser.parse_args()

    env = gym.make(args.env)
    model = load_model(args.model)

    obs_all = {
        'model': [],
        'random': [],
    }
    acs_all = {
        'model': [],
        'random': [],
    }
    rews_all = {
        'model': [],
        'random': [],
    }
    for ep in range(args.eps):
        for res_key, m in [('model', model), ('random', None)]:
            obs, acs, rews = run_ep(env, m)
            obs_all[res_key].append(obs)
            acs_all[res_key].append(acs)
            rews_all[res_key].append(rews)
            if args.verbose:
                total_rew = sum(rews)
                ep_len = len(rews)
                print(f'Episode {ep} final reward: {rews[-1],.3}, total reward: {total_rew,.3}, steps: {ep_len}')

    # print summary results even in non-verbose mode
    def summary_stats(rewards):
        total_rews = [sum(r) for r in rewards]
        total_eps = [len(r) for r in rewards]
        return (
            np.mean(total_rews), np.std(total_rews),
            np.mean(total_eps), np.std(total_eps),
        )

    print(f'Env {args.env} evaluated over {args.eps} episodes:')
    rew_mean, rew_std, eps_mean, eps_std = summary_stats(rews_all['model'])
    print(f'  Model {args.model}:')
    print(f'    Reward: mean={rew_mean:.2f},std={rew_std:.2f}')
    print(f'    Ep Len: mean={eps_mean:.2f},std={eps_std:.2f}')
    rew_mean, rew_std, eps_mean, eps_std = summary_stats(rews_all['random'])
    print(f'  Random Actions:')
    print(f'    Reward: mean={rew_mean:.2f},std={rew_std:.2f}')
    print(f'    Ep Len: mean={eps_mean:.2f},std={eps_std:.2f}')

    if args.out:
        obs = np.array(obs)
        acs = np.array(acs)
        rews = np.array(rews)
        np.savez(args.out, obs=obs, acs=acs, rews=rews)
