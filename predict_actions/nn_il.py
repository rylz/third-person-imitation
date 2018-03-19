#!/usr/bin/python3
__doc__="""Neural Nets for Imitation Learning

Learn the mapping from state to action as demonstrated in expert state sequences.

"""

import argparse
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
import numpy as np 
import tensorflow as tf 

def get_expert_training_data(data, n_traj=None):
    """Extract training examples from specifically-structured expert demonstrations.

    The expert demonstrations are structured into episodes - we flatten those into a series of
    state-action pairs that are all treated independently.

    n_traj indicates how many of episodes to pull data from. The first N episodes will be used,
    unless n_traj=None in which case all episodes will be used.

    """
    n_traj = n_traj or len(data['obs'])
    X_train = np.vstack(data['obs'][:n_traj])
    y_train = np.vstack(data['acs'][:n_traj])
    return (X_train, y_train)

def get_expert_training_pairs(data, n_traj=None):
    """Extract pairs of training examples from specifically-structured expert demonstrations.

    The expert demonstrations are structured into episodes - we flatten those into a series of
    state-action pairs that are all treated independently, then extract inputs as (s_i, s_{i+1})
    pairs and outputs as actions that were taken at state s_i. The idea is to allow more inference
    of momentum and trajectory in action prediction.

    n_traj indicates how many of episodes to pull data from. The first N episodes will be used,
    unless n_traj=None in which case all episodes will be used.

    """
    X_train, y_train = get_expert_training_data(data, n_traj)
    X_train1 = X_train[:-1,:]
    X_train2 = X_train[1:,:]
    X_train = np.hstack((X_train1, X_train2))
    y_train = y_train[1:]
    return (X_train, y_train)

def construct_nns(n_traj=None):
    """Construct Neural Nets for each of four environments using manually tuned, hard-coded params.

    Note: depends on data from the GAIL paper being present in data/gail_paper. You can download
    this data: https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing

    Returns a dictionary with keys:
        'hopper*', 'walker2d*', 'humanoid*', 'halfcheetah*',
    where the suffixes indicate details of neural net parameters, and the values are keras models
    for constructed and trained neural nets.

    """
    def input_size(env):
      return len(env['obs'][0][0])

    def output_size(env):
      return len(env['acs'][0][0])

    result = {}

    # construct and train hopper NN
    hopper = np.load("data/gail_paper/stochastic.trpo.Hopper.0.00.npz")
    model = Sequential()
    model.add(Dense(30, input_shape=(input_size(hopper),)))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(output_size(hopper)))
    model.add(Activation('linear'))
    model.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_hopper, y_train_hopper = get_expert_training_data(hopper, n_traj=n_traj)
    model.fit(X_train_hopper, y_train_hopper, batch_size=256, epochs=40, verbose=1)
    result['hopper_30x10'] = model

    # construct and train hopper NN
    walker = np.load("data/gail_paper/stochastic.trpo.Walker2d.0.00.npz")
    model_walker = Sequential()
    model_walker.add(Dense(50, input_shape=(input_size(walker),)))
    model_walker.add(Activation('relu'))
    model_walker.add(Dense(25))
    model_walker.add(Activation('relu'))
    model_walker.add(Dense(output_size(walker)))
    model_walker.add(Activation('linear'))
    model_walker.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_walker, y_train_walker = get_expert_training_data(walker, n_traj=n_traj)
    model_walker.fit(X_train_walker, y_train_walker, batch_size=256, epochs=40, verbose=1)
    result['walker_50x25'] = model_walker

    # construct and train half cheetah NN
    halfcheetah = np.load("data/gail_paper/stochastic.trpo.HalfCheetah.0.00.npz")
    model_cheetah = Sequential()
    model_cheetah.add(Dense(60, input_shape=(input_size(halfcheetah),)))
    model_cheetah.add(Activation('relu'))
    model_cheetah.add(Dense(35))
    model_cheetah.add(Activation('relu'))
    model_cheetah.add(Dense(output_size(halfcheetah)))
    model_cheetah.add(Activation('linear'))
    model_cheetah.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_cheetah, y_train_cheetah = get_expert_training_data(halfcheetah, n_traj=n_traj)
    model_cheetah.fit(X_train_cheetah, y_train_cheetah, batch_size=256, epochs=40, verbose=1)
    result['halfcheetah_60x35'] = model_cheetah

    # construct and train humanoid NN
    humanoid = np.load("data/gail_paper/stochastic.trpo.Humanoid.0.00.npz")
    model_humanoid = Sequential()
    model_humanoid.add(Dense(60, input_shape=(input_size(humanoid),)))
    model_humanoid.add(Activation('relu'))
    model_humanoid.add(Dense(35))
    model_humanoid.add(Activation('relu'))
    model_humanoid.add(Dense(output_size(humanoid)))
    model_humanoid.add(Activation('linear'))
    model_humanoid.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_humanoid, y_train_humanoid = get_expert_training_data(humanoid, n_traj=n_traj)
    model_humanoid.fit(X_train_humanoid, y_train_humanoid, batch_size=256, epochs=40, verbose=1)
    result['humanoid_60x35'] = model_humanoid

    return result

def construct_2state_nns(n_traj=None):
    """Construct Neural Nets for each of four environments using manually tuned, hard-coded params.

    Note: depends on data from the GAIL paper being present in data/gail_paper. You can download
    this data: https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing

    Builds neural nets that take pairs of two prior states as input.

    Returns a dictionary with keys:
        'hopper*', 'walker2d*', 'humanoid*', 'halfcheetah*',
    where the suffixes indicate details of neural net parameters, and the values are keras models
    for constructed and trained neural nets.

    """

    def input_size2(env):
      return len(env['obs'][0][0])*2

    def output_size(env):
      return len(env['acs'][0][0])
    result = {}

    # construct and train hopper NN
    hopper = np.load("data/gail_paper/stochastic.trpo.Hopper.0.00.npz")
    model = Sequential()
    model.add(Dense(50, input_shape=(input_size2(hopper),)))
    model.add(Activation('relu'))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dense(output_size(hopper)))
    model.add(Activation('linear'))
    model.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_hopper, y_train_hopper = get_expert_training_pairs(hopper, n_traj=n_traj)
    model.fit(X_train_hopper, y_train_hopper, batch_size=256, epochs=40, verbose=1)
    result['hopper_50x20'] = model

    # construct and train walker NN
    walker = np.load("data/gail_paper/stochastic.trpo.Walker2d.0.00.npz")
    model_walker2 = Sequential()
    model_walker2.add(Dense(70, input_shape=(input_size2(walker),)))
    model_walker2.add(Activation('relu'))
    model_walker2.add(Dense(30))
    model_walker2.add(Activation('relu'))
    model_walker2.add(Dense(output_size(walker)))
    model_walker2.add(Activation('linear'))
    model_walker2.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_walker2, y_train_walker2 = get_expert_training_pairs(walker, n_traj=n_traj)
    model_walker2.fit(X_train_walker2, y_train_walker2, batch_size=256, epochs=40, verbose=1)
    result['walker_70x30'] = model_walker2

    # construct and train half cheetah NN
    halfcheetah = np.load("data/gail_paper/stochastic.trpo.HalfCheetah.0.00.npz")
    model_cheetah2 = Sequential()
    model_cheetah2.add(Dense(70, input_shape=(input_size2(halfcheetah),)))
    model_cheetah2.add(Activation('relu'))
    model_cheetah2.add(Dense(30))
    model_cheetah2.add(Activation('relu'))
    model_cheetah2.add(Dense(output_size(halfcheetah)))
    model_cheetah2.add(Activation('linear'))
    model_cheetah2.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_cheetah2, y_train_cheetah2 = get_expert_training_pairs(halfcheetah, n_traj=n_traj)
    model_cheetah2.fit(X_train_cheetah2, y_train_cheetah2, batch_size=256, epochs=40, verbose=1)
    result['halfcheetah_70x30'] = model_cheetah2

    # construct and train humanoid NN
    humanoid = np.load("data/gail_paper/stochastic.trpo.Humanoid.0.00.npz")
    model_humanoid = Sequential()
    model_humanoid.add(Dense(70, input_shape=(input_size2(humanoid),)))
    model_humanoid.add(Activation('relu'))
    model_humanoid.add(Dense(30))
    model_humanoid.add(Activation('relu'))
    model_humanoid.add(Dense(output_size(humanoid)))
    model_humanoid.add(Activation('linear'))
    model_humanoid.compile(loss='mse',optimizer='Adam', metrics=['accuracy'])
    X_train_humanoid, y_train_humanoid = get_expert_training_pairs(humanoid, n_traj=n_traj)
    model_humanoid.fit(X_train_humanoid, y_train_humanoid, batch_size=256, epochs=40, verbose=1)
    result['humanoid_70x30'] = model_humanoid

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', help='number of states to use as input', default=1)
    parser.add_argument('--ntraj', help='how many episodes to use for training', type=int)
    parser.add_argument('--modelout', help='prefix for files to save models to', default='')
    args = parser.parse_args()

    assert args.states in (1, 2), 'only 1 or 2 state training is implemented'

    suffix = ''
    if args.states == 1:
        models = construct_nns(n_traj=args.ntraj)
    elif args.states == 2:
        models = construct_nns(n_traj=args.ntraj)
        suffix = '_2states'

    if args.ntraj:
        suffix = f'_{args.ntraj}traj{suffix}'
    for name, model in models.items():
        model.save(f'experts/{args.modelout}{name}{suffix}.h5')
