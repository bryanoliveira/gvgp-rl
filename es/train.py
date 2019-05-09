import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time

# from evostra import EvolutionStrategy
from pytorch_es import EvolutionModule
from pytorch_es.utils.helpers import weights_init
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision

from agent import Agent

parser = argparse.ArgumentParser(description='Evolution Strategies')
parser.add_argument('-w', '--weights_path', type=str, default='results.pkl', help='Path to save final weights')
parser.add_argument('-c', '--disable_cuda', action='store_true', help='Whether or not to use CUDA')
parser.add_argument('-g', '--generations', type=int, default=1000, help='Number of generations ')
parser.add_argument('-s', '--print_steps', type=int, default=10, help='Test and print agent every p steps.')
parser.add_argument('-p', '--population', type=int, default=100, help='Population size.')
parser.add_argument('--seed', type=int, default=123, help='Random seed for atari_py')
parser.add_argument('--wrapper', type=str, default='gvgai', help='Game emulator wrapper framework')
parser.add_argument('--game', type=str, default='gvgai-cec1-lvl0-v0', help='Game identifier')  # default='space_invaders' gvgai-cec1-lvl0-v0
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--lr', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
args = parser.parse_args()

if args.wrapper == 'gvgai':
    from env_gvgai import Env
elif args.wrapper == 'gym':
    from env_gym import Env
    if args.game == 'gvgai-cec1-lvl0-v0':
        args.game = 'SpaceInvaders-v0'
else:
    print('Please choose a wrapper from [gvgai, gym]')
    exit()

print('Options: ')
for k, v in vars(args).items():
    print(' ' * 4 + k + ': ' + str(v))
print('\nGeneral configs')
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    print(' ' * 4 + 'Using GPU: Yes')
else:
    args.device = torch.device('cpu')
    print(' ' * 4 + 'Using GPU: No')


num_features = 16

model = (Agent(args)).model
env = Env(args)
max_episode_length = args.max_episode_length

def get_reward(weights, model, render=False):
    global env
    global max_episode_length

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    obs = env.reset()
    done = False
    total_reward = 0
    i = 0
    while not done and i < max_episode_length:
        if render:
            env.render()
        with torch.no_grad():
            prediction = cloned_model(obs.unsqueeze(0))
            action = prediction.data.max(1)[1].item()
        obs, reward, done = env.step(action)

        total_reward += reward 
        i += 1
    env.close()
    return total_reward


partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=args.population,
    sigma=0.01, learning_rate=args.lr, decay=0.9999,
    reward_goal=600, consecutive_goal_stopping=10, threadcount=1,
    cuda=(args.device == torch.device('cuda')), render_test=True, save_path=os.path.abspath(args.weights_path)
)

start = time.time()
final_weights = es.run(args.generations, print_step=args.print_steps)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))
print("Saved final weights to " + args.weights_path)


reward = partial_func(final_weights, render=True)

print("Reward from final weights: " + str(reward))
print("Time to completion: " + str(end))