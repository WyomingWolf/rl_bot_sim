import os
import numpy as np
import argparse
import math
import gym
import csv
import re
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from envs.ant_mini import AntMiniEnv 
from envs.ant_mini_sensor import AntMiniSensorEnv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", type=str, nargs='*', default=['PPO', 'TD3', 'SAC'],
	help="List of algorithms")

args = vars(ap.parse_args())

def load_model(algo, model_path, env):
    #SAC
    if algo == 'SAC':
        try:
            model = SAC.load(model_path, env, print_system_info=True)
        except Exception as e:
            print(e)
    # TD3
    elif algo == 'TD3':
        try:
            model = TD3.load(model_path, env, print_system_info=True)
        except Exception as e:
            print(e)
    # DDPG
    elif algo == 'DDPG':
        try:
            model = DDPG.load(model_path, env, print_system_info=True)
        except Exception as e:
            print(e)
    # A2C
    elif algo == 'A2C':
        try:
            model = A2C.load(model_path, env, print_system_info=True)
        except Exception as e:
            print(e)
    # PPO
    else:
        try:
            model = PPO.load(model_path, env, print_system_info=True)
        except Exception as e:
            print(e)

    return model

n_steps = 1000
data = []

for algo in args['algorithm']:
    path = os.path.join('logs', algo)
    x_data = None
    rewards = None
    legend = []

    envs = sorted(os.listdir(path))
    for env in envs:
        env_args = env.split('-')
        if env_args[0] == 'Ant_mini_sensor':
            gym_env = AntMiniSensorEnv(sensor_level=re.sub('\D', '',env_args[1]))
        else:
            gym_env = AntMiniEnv()
        env_path = os.path.join(path, env)

        for seed in os.listdir(env_path):
            model_path = os.path.join(os.path.join(env_path, seed), 'best_model')
            algo_name = algo.split('_')
            model = load_model(algo_name[0], model_path, gym_env)

            x_velocity = np.zeros(n_steps)
            y_velocity = np.zeros(n_steps)
            quat_error = np.zeros(n_steps)
            position = []
            obs = gym_env.reset()
            for i in range(n_steps):
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = gym_env.step(action)
                x_velocity[i] = info["x_velocity"]
                y_velocity[i] = info["y_velocity"]
                quat_error[i] = info["quat_square_error"]
                position = [info["x_position"], info["y_position"]]
                #gym_env.render()
                if done:
                    obs = gym_env.reset()

            RMSD = math.sqrt(np.sum(quat_error)/n_steps)
            x_avg = np.mean(x_velocity)
            x_var = np.var(x_velocity)
            y_avg = np.mean(y_velocity)
            y_var = np.var(y_velocity)

            data.append([algo, env, seed.replace('seed_', ''), x_avg, x_var, y_avg, y_var, RMSD])
            #print([algo, env, x_avg, seed, x_var, y_avg, y_var, RMSD])
            

filename = "logs/results.csv"
fields = ['Algo', 'Env', 'Seed', 'X_avg', 'X_var', 'Y_avg', 'Y_var', 'Quat_rmsd']
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(data)

algos = set([i[0] for i in data])
algos = sorted(list(algos))
envs = set([i[1] for i in data])
envs = sorted(list(envs))

summary = []
for algo in algos:
    print(algo)  
    for env in envs:
        print(env) 
        sum_data = [] 
        for i in data:
            if i[:2] == [algo, env]:
                sum_data.append(i[3:])

        tmp = [algo] + [env] + np.mean(np.array(sum_data), axis=0).tolist()
        print(tmp)
        summary.append(tmp)

filename = "logs/summary.csv"
fields = ['Algo', 'Env', 'X_avg', 'X_var', 'Y_avg', 'Y_var', 'Quat_rmsd']
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(summary)
        



