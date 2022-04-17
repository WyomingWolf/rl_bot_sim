import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", type=str, nargs='*', default=['PPO', 'TD3', 'SAC'],
	help="List of algorithms")

args = vars(ap.parse_args())

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


for algo in args['algorithm']:
    path = os.path.join('logs', algo)
    x_data = None
    rewards = None
    legend = []

    envs = sorted(os.listdir(path))
    for env in envs:
        env_path = os.path.join(path, env)
        legend.append(env)
        
        y_data = None

        for seed in sorted(os.listdir(env_path)):
            log_folder = os.path.join(env_path, seed)
            x, y = ts2xy(load_results(log_folder), 'timesteps')
            y = moving_average(y, window=20)

            if x_data is None:
                x_data = x[len(x) - len(y):]

            if y_data is None:
                y_data = y
            else:
                if max(x_data.shape) <= len(y):
                    if len(y) > max(x_data.shape):
                        y = y[len(y) - max(x_data.shape):] 
                    y_data = np.vstack((y_data, y))
            

        env_fig = plt.figure(env)  
        for i in range(y_data.shape[0]):
            plt.plot(x_data, y_data[i,:])
        plt.show()

        if len(y_data.shape) != 1: 
            y_data = np.mean(y_data, axis=0)

        if rewards is None:
            rewards = y_data
        else:
            rewards = np.vstack((rewards, y_data))

    num_lines = rewards.shape[0]
    fig = plt.figure('Title')
    for i in range(num_lines):
        plt.plot(x_data, rewards[i,:])
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(algo)
    plt.legend(legend) 
    plt.show()     


