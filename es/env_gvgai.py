from collections import deque
import random
import cv2
import torch

import gym
import gym_gvgai

# Predefined names referring to GVGAI framework
games = ['gvgai-cec1', 'gvgai-cec2', 'gvgai-cec3']
trainingLevels = ['lvl0-v0', 'lvl1-v0']


class Env:
    def __init__(self, args):
        self.device = args.device
        self.max_frames = args.max_episode_length
        self.env = gym_gvgai.make(args.game)
        actions = self.env.env.GVGAI.actions()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
        self.obs = self.env.reset
        self.frame = 0

    def _get_state(self):
        state = cv2.resize(self.obs, (84, 84), interpolation=cv2.INTER_LINEAR)
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        self._reset_buffer()
        self.obs = self.env.reset()
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.frame = 0
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            self.obs, treward, done, debug = self.env.step(action)
            reward += treward
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        self.frame += 1
        if self.frame >= self.max_frames:
            done = True
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
