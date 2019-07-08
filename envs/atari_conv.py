import gym
import cv2
import numpy as np
from envs._interface import EnvInterface
from envs.atari_wrappers import * 


class Env(EnvInterface):
    def __init__(self, env_name, stack_frames=4):  
        super(Env, self).__init__(env_name)

        self.stack_frames = stack_frames
        self.env = make_atari(env_name)
        self.env = wrap_deepmind(self.env, frame_stack=True,  pytorch_img=True)


    def reset(self):
        state = self.env.reset()
        return state.__array__()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        
        return state, reward, done, info

    def factory(env_name):
        return lambda : Env(env_name)