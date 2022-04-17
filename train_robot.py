import os
import gym
import time
import numpy as np
import torch.cuda as tc
import re
from envs.ant_mini import AntMiniEnv 
from envs.ant_mini_sensor import AntMiniSensorEnv
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_util import make_vec_env
import argparse
from gym.wrappers.time_limit import TimeLimit
from utils import SaveOnBestTrainingRewardCallback
from utils import DynamicsRandomizationCallback
from utils import plot_results
from robosuite.utils.mjmod import DynamicsModder
from stable_baselines3.common.callbacks import CallbackList

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--algorithm", default="PPO",
	help="PPO, SAC, A2C, DDPG or TD3")
ap.add_argument("-dr", "--dom_rand", default=False, action=argparse.BooleanOptionalAction,
	help="Domain Randomization")
ap.add_argument("-e", "--env", default="Ant_mini-v1",
	help="Ant_mini-v1, Ant_mini_sensor-v0, Ant_mini_senosr-v1, etc")
ap.add_argument("-m", "--model", required=False,
	help="Load trained model")
ap.add_argument("-sa", "--save_as", default="trained_model",
	help="Name saved model")
ap.add_argument("-se", "--seed", type=int, default=12345,
	help="RNG seed")
ap.add_argument("-si", "--size", default="m",
	help="Policy size: s, m, l or xl")
ap.add_argument("-t", "--timesteps", type=int, default=3000000,
	help="Total training steps")
ap.add_argument("-d", "--device", type=int, default=0,
	help="Select GPU: [0, 1, 2, 3]")
ap.add_argument("-de", "--depth", type=int, default=2,
	help="Network depth: 2 or more")
args = vars(ap.parse_args())

seed = args['seed']
total_timesteps = args['timesteps']
cwd = os.path.abspath(os.getcwd())
seed_dir = 'seed_' + str(args['seed'])
path = os.path.join('logs', os.path.join(args['algorithm'], os.path.join(args['env'], seed_dir)))
log_dir = os.path.join(cwd, path)
os.makedirs(log_dir, exist_ok=True)

# Initialize environment
'''
if args['algorithm'] == 'A2C':
    env = make_vec_env(args['env'], n_envs=4)
else:   
'''
env_args = args['env'].split('-')

n_steps = 1000
if env_args[0] == 'Ant_mini_sensor':
    env = AntMiniSensorEnv(sensor_level=re.sub('\D', '',env_args[1]))
else:
    env = AntMiniEnv()
env = TimeLimit(env, max_episode_steps=n_steps)
if args['dom_rand'] == True:
    print('Dynamics Randomization Enabled')
    modder = DynamicsModder(sim=env.sim, randomize_viscosity=False)
env = Monitor(env, log_dir)

# Select gpu
if tc.is_available():
    print('GPU COUNT:', tc.device_count())
    tc.set_device(args['device'])
    print(tc.get_device_name(args['device']))

'''
if args['depth'] < 2:
    depth = 2
else:
    depth = args['depth']

if args['size'] == 's':
    net = [128]*depth
    policy_kwargs = dict(net_arch=net)
elif args['size'] == 'l':
    net = [512]*depth
    policy_kwargs = dict(net_arch=net)
elif args['size'] == 'xl':
    net = [1024]*depth
    policy_kwargs = dict(net_arch=net)
else:
     net = [256]*depth
     policy_kwargs = dict(net_arch=net)
'''
policy_kwargs = None

# Initialize agent
#SAC
if args['algorithm'] == 'SAC':
    if args['model'] is not None:
        try:
            model = SAC.load(args["model"], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        #net = [128, 128]
        #policy_kwargs = dict(net_arch=net)
        model = SAC('MlpPolicy', env, train_freq=(1, 'episode'), gradient_steps=-1, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
# TD3
elif args['algorithm'] == 'TD3':
    if args['model'] is not None:
        try:
            model = TD3.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3('MlpPolicy', env, action_noise=action_noise, train_freq=(1, 'episode'), gradient_steps=-1, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
# DDPG
elif args['algorithm'] == 'DDPG':
    if args['model'] is not None:
        try:
            model = DDPG.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG('MlpPolicy', env, action_noise=action_noise, train_freq=(1, 'episode'), gradient_steps=-1, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
# A2C
elif args['algorithm'] == 'A2C':
    if args['model'] is not None:
        try:
            model = A2C.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = A2C('MlpPolicy', env, n_steps=1024, learning_rate= 0.001, policy_kwargs=policy_kwargs, verbose=1, seed=seed)
# PPO
else:
    if args["model"] is not None:
        try:
            model = PPO.load(args['model'], env, print_system_info=True)
        except Exception as e:
            print(e)
    else:
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, seed=seed)

save_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)

if args['dom_rand'] == True:
    random_callback = DynamicsRandomizationCallback(check_freq=1000, modder=modder)
    callback = CallbackList([save_callback, random_callback])
else:
    callback = CallbackList([save_callback])

model.learn(total_timesteps=total_timesteps, callback=callback)

plot_results(log_dir)
