import gym
import numpy as np
import ptan
from tensorboardX import SummaryWriter
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from algorithms._interface import RLInterface

class Model(nn.Module):
    def __init__(self, input_shape, n_actions, stack_frames=1):
        super(Model, self).__init__()

        # A shared convolution body
        self.conv = nn.Sequential(
            nn.Conv2d(stack_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)

        # First head is returning the policy with probability distribution over actions
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # Second head returns one single number (approximate state's value)
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Returns a tuple of two tensors: policy and value
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return Categorical(self.policy(conv_out)), self.value(conv_out)


class A2C(RLInterface):
    def __init__(
        self, 
        env_factory, 
        save_load_path = "trained_models", 
        skip_load = False,
        render = False,
        n_workers =  mp.cpu_count(),
        cuda = torch.cuda.is_available,
        gamma = 0.9, 
        max_eps = 10000,
        max_eps_length = 1000):

        super(A2C, self).__init__()

        self.device = torch.device('cuda' if cuda else 'cpu')
        print('Using ' + str(self.device))

        make_env = env_factory[0] if type(env_factory) is list else env_factory
        # self.envs = [make_env() for _ in range(n_workers)] TODO
        self.env = make_env()

        self.max_eps = max_eps
        self.max_eps_length = max_eps_length
        self.gamma = gamma

        self.input_shape = self.env.reset()[None, :].shape
        self.network = Model(self.input_shape, self.env.n_actions, 1).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)     

        print(self.network)

    def run(self):
        frame_idx = 0
        state = self.env.reset()

        while frame_idx < self.max_eps:
            log_probs = []
            values    = []
            rewards   = []
            masks     = []
            entropy = 0

            for _ in range(5):
                state = state[None, None, :]
                state = torch.FloatTensor(state).to(self.device)
                policy, value = self.network(state)

                action = policy.sample()
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())

                log_prob = policy.log_prob(action)
                entropy += policy.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))
                
                state = next_state
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(rewards.mean(), rewards[-1])
            
            next_state = torch.FloatTensor(next_state).to(self.device)
            _, next_value = self.network(next_state)
            returns = compute_returns(next_value, rewards, masks)
            
            log_probs = torch.cat(log_probs)
            returns   = torch.cat(returns).detach()
            values    = torch.cat(values)

            advantage = returns - values
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = (advantage**2).mean()         

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optmizer.step()   


    def compute_returns(next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]    
            returns.insert(0, R)
        return returns    



