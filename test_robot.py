import os
import gym
import time
import math
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg') 
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
import re
from envs.ant_mini import AntMiniEnv 
from envs.ant_mini_sensor import AntMiniSensorEnv
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", default="PPO",
	help="PPO, SAC, A2C, DDPG or TD3")
ap.add_argument("-e", "--env", default="Ant_mini_sensor-v5",
	help="Ant_mini-v4, Ant_mini_sensor-v1, Ant_min_senosr-v2, etc")
ap.add_argument("-se", "--seed", default='12345',
	help="RNG sedd")
ap.add_argument("-s", "--save", default=True, type=bool, 
	help="Boolean: Save test results")
args = vars(ap.parse_args())

n_steps = 500
env_args = args['env'].split('-')
if env_args[0] == 'Ant_mini_sensor':
    env = AntMiniSensorEnv(sensor_level=re.sub('\D', '',env_args[1]))
else:
    env = AntMiniEnv()

cwd = os.path.abspath(os.getcwd())
seed_dir = 'seed_' + args['seed']
path = os.path.join('logs', os.path.join(args['algorithm'], os.path.join(args['env'], seed_dir)))
log_dir = os.path.join(cwd, path)


model_path = os.path.join(path, 'best_model')


print(model_path)
algo_args = args['algorithm'].split('_')
#SAC
if algo_args[0] == 'SAC':
    try:
        model = SAC.load(model_path, env, print_system_info=True)
    except Exception as e:
        print(e)
# TD3
elif algo_args[0] == 'TD3':
    try:
        model = TD3.load(model_path, env, print_system_info=True)
    except Exception as e:
        print(e)
# DDPG
elif algo_args[0] == 'DDPG':
    try:
        model = DDPG.load(model_path, env, print_system_info=True)
    except Exception as e:
        print(e)
# A2C
elif algo_args[0] == 'A2C':
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

print("Model Loaded")

x_velocity = np.zeros(n_steps)
y_velocity = np.zeros(n_steps)
quat_error = np.zeros(n_steps)
position = []
obs = env.reset()
for i in range(n_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    x_velocity[i] = info["x_velocity"]
    y_velocity[i] = info["y_velocity"]
    quat_error[i] = info["quat_square_error"]
    position = [info["x_position"], info["y_position"]]
    env.render()
    if done:
      obs = env.reset()

    time.sleep(0.05)

RMSD = math.sqrt(np.sum(quat_error)/n_steps)
print("Average X Velocity:", np.mean(x_velocity))
print("X Velocity Variance:", np.var(x_velocity))

print("Average Y Velocity:", np.mean(y_velocity))
print("Y Velocity Variance:", np.var(y_velocity))

print(position)

print("Quaternion RMSD:", RMSD)

'''
file_path = os.path.join(path, 'results.txt')
with open(file_path, 'w') as f:
    f.write(str(np.mean(x_velocity)) + '\n')
    f.write(str(np.var(x_velocity)) + '\n')
    f.write(str(np.mean(y_velocity)) + '\n')
    f.write(str(np.var(y_velocity)) + '\n')
    f.write(str(RMSD) + '\n')

plt.plot(x_velocity)
plt.xlabel('Timesteps')
plt.ylabel('Velocity')
plt.title('Robot Velocity')

save_path = os.path.join(path, 'Model_Velocity.png')
plt.savefig(save_path)
'''