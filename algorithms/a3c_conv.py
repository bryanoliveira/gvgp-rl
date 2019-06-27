import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from algorithms._interface import RLInterface
from utils import np_torch_wrap


class Model(nn.Module):
    def __init__(self, input_shape, n_actions, stack_frames=4):
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
        
        conv_out_size = int(np.prod(self.conv(torch.zeros(1, *input_shape)).size()))

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

        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        print(x.shape)
        # Returns a tuple of two tensors: policy and value
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(
        self, 
        worker_name,
        env_factory,
        f_checkpoint,
        f_sync, 
        res_queue, 
        global_ep_counter,
        update_global_delay=20,
        max_eps=10000,
        max_eps_length=1000,
        n_s=None,
        n_a=None,
        render=False):

        super(Worker, self).__init__()

        # callback global functions
        self.synchronize = f_sync  # push local gradients to global network
        self.checkpoint = f_checkpoint  # calculate statistics and save paramenters when improvements are achieved

        print("Worker w%i: " % worker_name, end="")
        self.env = env_factory()
        env_temp = env_factory()
        env_shape = env_temp.reset().shape
        self.local_network = Model(n_s if n_s is not None else env_shape, n_a if n_a is not None else self.env.n_actions, self.env.stack_frames)  # local network

        # local worker config
        self.name = 'w%i' % worker_name
        self.render = render
        self.max_eps = max_eps  # max episodes of all workers
        self.max_eps_length = max_eps_length
        self.update_global_delay = update_global_delay
        self.global_ep_counter = global_ep_counter
        self.res_queue = res_queue  # shared queue to store results

    def run(self):
        thread_step = 1  # initialize thread step counter
        while self.global_ep_counter.value < self.max_eps:  # repeat until T < Tmax
            # here we don't reset gradients or synchronize thread-specific parameters with
            # the global network - this will be treated in "self.synchronize" function 
            buffer_state, buffer_action, buffer_reward = [], [], []
            episode_reward = 0.
            episode_step = 0  # Tstart = T (this is equivalent, but easier to understand)
            state = self.env.reset()  # get state St
            while episode_step < self.max_eps_length:  # repeat until terminal or T-Tstart==Tmax
                if self.render and self.name == 'w0':
                    self.env.render()

                action = self.local_network.choose_action(np_torch_wrap(state[None, :]))  # perform At according to local policy
                new_state, reward, done, _ = self.env.step(action if action < self.env.n_actions else 0)  # receive reward Rt and new state St+1
                if done: reward = -1
                episode_reward += reward  # accumulate reward
                buffer_action.append(action)
                buffer_state.append(state)
                buffer_reward.append(reward)

                if thread_step % self.update_global_delay == 0 or done:  # update global and assign to local net
                    print("buffer ", np.array(buffer_state).shape)
                    # calculate R
                    # for... accumulate gradients ... end for
                    # perform asynchronous update on global network
                    # send all last states/actions/rewards function to calculate accumulated gradients and push it to the global network
                    self.synchronize(self.local_network, done, new_state, buffer_state, buffer_action, buffer_reward)
                    buffer_state, buffer_action, buffer_reward = [], [], []

                    if done: break  # done and print information

                state = new_state
                thread_step += 1  # t = t + 1
                episode_step += 1  # T = T + 1

            self.checkpoint(episode_reward, self.name)

        self.res_queue.put(None)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class A3C(RLInterface):
    def __init__(
        self, 
        env_factory, 
        save_load_path = "trained_models", 
        skip_load = False,
        render = False,
        n_workers = mp.cpu_count(), 
        gamma = 0.9, 
        update_global_delay = 20,
        max_eps = 10000,
        max_eps_length = 1000):

        super(A3C, self).__init__()

        self.name = "A3C_Conv"

        # init temp env to get it's properties
        print("A3C Global: ", end="")
        env = env_factory[0]() if type(env_factory) is list else env_factory()
        self.env_name = env.name
        self.env_shape = env.reset().shape
        
        # initialize global network
        self.global_network = Model(self.env_shape, env.n_actions, env.stack_frames)
        self.save_load_path = save_load_path
        if not skip_load:
            self.load()

        self.global_network.share_memory()  # share the global parameters in multiprocessing

        # configure shared parameters
        self.gamma = gamma
        self.optimizer = SharedAdam(self.global_network.parameters(), lr=0.0001)  # global optimizer
        self.current_max_reward = mp.Value('d', float("-inf"))  # max reward threshold
        self.global_ep_counter = mp.Value('i', 0)
        self.global_ep_reward = mp.Value('d', 0.)  # current episode reward
        self.res_queue = mp.Queue()  # queue to receive workers statistics

        # instantiate workers
        self.workers = [
            Worker(
                worker_name=i,
                # assign a random environment for each worker, if multiple envs are received 
                env_factory = env_factory[random.randint(0, len(env_factory) - 1)] if type(env_factory) is list else env_factory, 
                f_checkpoint = self.checkpoint,
                f_sync = self.sync,
                res_queue = self.res_queue,
                global_ep_counter = self.global_ep_counter, 
                update_global_delay = update_global_delay, 
                max_eps = max_eps,
                max_eps_length = 1000, 
                n_s = None,
                n_a = env.n_actions,
                render = render
            ) for i in range(n_workers)
        ]

    def run(self):
        # run workers and register statistics
        [w.start() for w in self.workers]
        res = []  # record episode reward to plot
        while True:
            r = self.res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]

        # show statistics
        plt.plot(res)
        plt.ylabel('Moving average episode reward')
        plt.xlabel('Step')
        plt.show()

    def sync(self, local_network, done, new_state, buffer_state, buffer_action, buffer_reward):
        # calculate R
        if done:
            R = 0.  # for terminal St
        else: # for non-terminal St // Bootstrap from last state
            R = local_network.forward(np_torch_wrap(new_state[None, :]))[-1].data.numpy()[0, 0]

        # for i E {t - 1, ..., Tstart}
        buffer_v_target = []
        for r in buffer_reward[::-1]:    # reverse buffer r
            R = r + self.gamma * R
            buffer_v_target.append(R)
        buffer_v_target.reverse()
        
        # accumulate gradients
        loss = local_network.loss_func(
            np_torch_wrap(np.array(buffer_state)),
            np_torch_wrap(np.array(buffer_action), dtype=np.int64) if buffer_action[0].dtype == np.int64 else np_torch_wrap(np.vstack(buffer_action)),
            np_torch_wrap(np.array(buffer_v_target)[:, None]))
        
        # perform asynchronous update of Θ using dΘ and of Θv using dΘv
        self.optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(local_network.parameters(), self.global_network.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()

        # synchronize thread-specific parameters Θ' = Θ and Θ'v = Θv
        local_network.load_state_dict(self.global_network.state_dict())

    def checkpoint(self, episode_reward, worker_name):
        # increment global episode counter
        with self.global_ep_counter.get_lock():
            self.global_ep_counter.value += 1

        # update global moving average reward
        with self.global_ep_reward.get_lock():
            if self.global_ep_reward.value == 0.:
                self.global_ep_reward.value = episode_reward
            else:
                self.global_ep_reward.value = self.global_ep_reward.value * 0.99 + episode_reward * 0.01

        # check if max reward threshold have been surpassed
        with self.current_max_reward.get_lock():
            if self.current_max_reward.value < episode_reward:
                self.current_max_reward.value = episode_reward
                self.save()

        self.res_queue.put(self.global_ep_reward.value)
        print(
            worker_name,
            "Ep:", self.global_ep_counter.value,
            "| Ep_r: %.0f" % self.global_ep_reward.value,
        )