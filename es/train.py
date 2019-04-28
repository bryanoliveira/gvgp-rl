import argparse
import copy
from functools import partial
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import time
import cv2
# from evostra import EvolutionStrategy
from pytorch_es import EvolutionModule
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from PIL import Image

gym_logger.setLevel(logging.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights_path', type=str, required=True, help='Path to save final weights')
parser.add_argument('-c', '--disable_cuda', action='store_true', help='Whether or not to use CUDA')
parser.set_defaults(cuda=False)

args = parser.parse_args()

_num_features = 16

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


class InvadersModel(nn.Module):
    def __init__(self, num_features):
        super(InvadersModel, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4096, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 8, num_features * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 16, 6, 4, 1, 0, bias=False),
            nn.Softmax(1)
        )

    def forward(self, input):
        main = self.main(input)
        return main


global_model = InvadersModel(_num_features)

if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = False  # Disable nondeterministic ops (not sure if critical but better safe than sorry)
    global_model = global_model.cuda()
    print(' ' * 4 + 'Using GPU: Yes')
else:
    args.device = torch.device('cpu')
    print(' ' * 4 + 'Using GPU: No')

env = gym.make("SpaceInvaders-v0")


def get_reward(weights, model, render=False):
    global env

    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data = weights[i]
        except:
            param.data = weights[i].data

    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
            time.sleep(0.005)

        image = transform(Image.fromarray(obs))
        image = image.unsqueeze(0)
        image = image.cuda()
        with torch.no_grad():
            prediction = cloned_model(Variable(image))
        action = prediction.data.cpu().numpy().argmax()
        obs, reward, done, _ = env.step(action)

        total_reward += reward
    env.close()
    return total_reward


partial_func = partial(get_reward, model=global_model)
mother_parameters = list(global_model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=100,
    sigma=0.01, learning_rate=0.001, decay=0.9999,
    reward_goal=600, consecutive_goal_stopping=10, threadcount=1,
    cuda=(args.device == torch.device('cuda')), render_test=True, save_path=os.path.abspath(args.weights_path)
)

start = time.time()
final_weights = es.run(10000, print_step=10)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath(args.weights_path), 'wb'))

result = partial_func(final_weights, render=True)
