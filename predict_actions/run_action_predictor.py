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
    done = False
    while not done:
        obs = []
        acs = []
        rews = []
        state = np.reshape(np.array(state), (1, -1))
        action = model.predict(state)
        next_state, reward, done, _ = env.step(action)
        obs.append(state)
        acs.append(action)
        rews.append(reward)

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
        print(f'Episode {ep} final reward: {rews[-1],.3}')

    if args.out:
        obs, acs, rews = map(np.array, [obs_all, acs_all, rews_all])
        np.savez(args.out, obs=obs, acs=acs, rews=rews)
