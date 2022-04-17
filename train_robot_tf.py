import os
import gym
import time
import numpy as np
import tensorflow as tf
from stable_baselines import A2C
from stable_baselines import ACKTR
from stable_baselines import DDPG
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import TD3
from stable_baselines import TRPO
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.bench.monitor import Monitor
import argparse
from utils import SaveOnBestTrainingRewardCallback
from utils import plot_results

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", default="PPO",
	help="A2C, ACKTR, DDPG, PPO, SAC, TD3, or TRPO")
ap.add_argument("-e", "--env", default="Ant-v3",
	help="Ant-v3 or Ant_mod-v1")
ap.add_argument("-m", "--model", required=False,
	help="Load trained model")
ap.add_argument("-sa", "--save_as", default="trained_model",
	help="Name saved model")
ap.add_argument("-se", "--seed", default=12345,
	help="RNG seed")
ap.add_argument("-t", "--timesteps", type=int, default=3000000,
	help="Total training steps")
ap.add_argument("-d", "--device", type=int, default="0",
	help="GPU number: [0, 1, 2, 3]")
args = vars(ap.parse_args())

seed = 12345
total_timesteps = args['timesteps']
cwd = os.path.abspath(os.getcwd())
path = os.path.join('tensorflow/logs', os.path.join(args['env'], args['algorithm']))
log_dir = os.path.join(cwd, path)
os.makedirs(log_dir, exist_ok=True)

# Initialize environment
env = gym.make(args['env'])
env = Monitor(env, log_dir)

# Select gpu
CUDA_VISIBLE_DEVICES = args['device']
"""
device = '/device:GPU:' + str(args['device'])
#device = device.join(args['device'])
print(device)

if len(tf.config.list_physical_devices('GPU')) <= args['device']:
    device = join('/device:GPU:', args['device'])
    print(device)
else:
    print('GPU not available')
"""
"""
if tc.is_available():
    print('GPU COUNT:', tc.device_count())
    tc.set_device(args['device'])
    print(tc.get_device_name(args['device']))
"""

# Initialize agent
# A2C
if args['algorithm'] == 'A2C':
    if args['model'] is not None:
        try:
            model = A2C.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = A2C('MlpPolicy', env, verbose=1, seed=seed)
# ACKTR
elif args['algorithm'] == 'ACKTR':
    if args['model'] is not None:
        try:
            model = ACKTR.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = ACKTR('MlpPolicy', env, verbose=1, seed=seed)
# DDPG
elif args['algorithm'] == 'DDPG':
    if args['model'] is not None:
        try:
            model = DDPG.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
        model = DDPG('MlpPolicy', env, action_noise=action_noise, train_freq=(1, 'episode'), gradient_steps=-1, verbose=1, seed=seed)
# SAC
elif args['algorithm'] == 'SAC':
    if args['model'] is not None:
        try:
            model = SAC.load(args["model"], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = SAC('MlpPolicy', env, train_freq=(1, 'episode'), gradient_steps=-1, verbose=1, seed=seed)
# TD3
elif args['algorithm'] == 'TD3':
    if args['model'] is not None:
        try:
            model = TD3.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
        model = TD3('MlpPolicy', env, action_noise=action_noise, train_freq=(1, 'episode'), gradient_steps=-1, verbose=1, seed=seed)
# TRPO
elif args['algorithm'] == 'TRPO':
    if args["model"] is not None:
        try:
            model = TRPO.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = TRPO('MlpPolicy', env, verbose=1, seed=seed)
# PPO
else:
    if args["model"] is not None:
        try:
            model = PPO2.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = PPO2('MlpPolicy', env, verbose=1, seed=seed)

callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
model.learn(total_timesteps=total_timesteps, callback=callback)

plot_results(log_dir)
