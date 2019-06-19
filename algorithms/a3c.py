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


class A3C(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(A3C, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 800) # (X, S_DIM) * (S_DIM, 200)
        self.pi2 = nn.Linear(800, a_dim)
        self.v1 = nn.Linear(s_dim, 600)
        self.v2 = nn.Linear(600, 1)
        [(nn.init.normal_(layer.weight, mean=0., std=0.1), nn.init.constant_(layer.bias, 0.)) for layer in [self.pi1, self.pi2, self.v1, self.v2]]
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu6(self.v1(x))
        values = self.v2(v1)
        return logits, values

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
    def __init__(self, env_factory, gnet, opt, global_max_r, global_ep, global_ep_r, res_queue, update_global_delay, gamma, max_eps, name, max_eps_length=1000, n_s=None, n_a=None):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.global_max_r, self.g_ep, self.g_ep_r, self.res_queue = global_max_r, global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.env = env_factory()
        self.lnet = A3C(n_s if n_s is not None else self.env.n_obs, n_a if n_a is not None else self.env.n_actions)           # local network
        self.update_global_delay = update_global_delay
        self.gamma = gamma
        self.max_eps = max_eps
        self.max_eps_length = max_eps_length

    def run(self):
        total_step = 1
        while self.g_ep.value < self.max_eps:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            ep_length = 0
            while ep_length < self.max_eps_length:
                #if self.name == 'w0':
                #    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a if a < self.env.n_actions else 0)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.update_global_delay == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.gnet, self.global_max_r, self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, self.env.name)
                        break
                s = s_
                total_step += 1
                ep_length += 1

            record(self.gnet, self.global_max_r, self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name, self.env.name)

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


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.               # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in br[::-1]:    # reverse buffer r
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    loss = lnet.loss_func(
        v_wrap(np.vstack(bs)),
        v_wrap(np.array(ba), dtype=np.int64) if ba[0].dtype == np.int64 else v_wrap(np.vstack(ba)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(gnet, global_max_r, global_ep, global_ep_r, ep_r, res_queue, name, env_name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    with global_max_r.get_lock():
        if global_max_r.value < ep_r:
            global_max_r.value = ep_r
            save_model(gnet, env_name)

    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )

def save_model(model, env_name):
    torch.save(model.state_dict(), os.path.join('trained_models', env_name + '.pth'))
    print("Model saved.")

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print("Model " + path + " loaded.")

def run(env_factory, load_path = "trained_models", skip_load=False, update_global_delay=50, gamma=0.9, max_eps=4000):
    env = env_factory() if type(env_factory) is not list else env_factory[0]()

    gnet = A3C(env.n_obs, env.n_actions)  # global network
    load_path = os.path.join(load_path, env.name + '.pth')
    if not skip_load and os.path.isfile(load_path):
        load_model(gnet, load_path)

    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)  # global optimizer
    global_max_r, global_ep, global_ep_r, res_queue = mp.Value('d', float("-inf")), mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [
        Worker(env_factory[random.randint(0, len(env_factory) - 1)] if type(env_factory) is list else env_factory, 
                gnet, opt, global_max_r, global_ep, global_ep_r, res_queue, update_global_delay, gamma, max_eps, i, n_a=env.n_actions) 
        for i in range(mp.cpu_count())
    ]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()