#!/usr/bin/python3

import argparse
import json
import random
import gym
import math
import numpy as np
from keras.models import load_model

def run_ep(env, model):
    state = env.reset()
    # some of our models use one state, some backtrack by some number of states in a rolling window
    # here we initially use several copies of the initial state, then roll the window
    n_inputs = model.input_shape[1] // len(state)
    model_input = np.array(list(state) * n_inputs)
    done = False
    obs = []
    acs = []
    rews = []
    while not done:
        model_input = np.concatenate((model_input[len(state):], state))
        action = model.predict(np.reshape(np.array(model_input), (1, -1)))
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
    parser.add_argument('--eps', help='Number of episodes to evaluate', type=int, required=True)
    parser.add_argument('-o', '--out', help='File to save results to')
    args = parser.parse_args()

    env = gym.make(args.env)
    model = load_model(args.model)

    obs_all = []
    acs_all = []
    rews_all = []
    for ep in range(args.eps):
        obs, acs, rews = run_ep(env, model)
        obs_all.append(obs)
        acs_all.append(acs)
        rews_all.append(rews)
        total_rew = sum(rews)
        ep_len = len(rews)
        print(f'Episode {ep} final reward: {rews[-1],.3}, total reward: {total_rew,.3}, steps: {ep_len}')

    if args.out:
        obs = np.array(obs)
        acs = np.array(acs)
        rews = np.array(rews)
        np.savez(args.out, obs=obs, acs=acs, rews=rews)
