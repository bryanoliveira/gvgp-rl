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

    def _preprocess(self, state):
        state = cv2.resize(np.float32(state), (self._img_size, self._img_size), interpolation=cv2.INTER_LINEAR)
        state = cv2.cvtColor(np.float32(state), cv2.COLOR_BGR2GRAY)
        state = state / 255  # transforma a imagem num vetor e normaliza para 0~1
        return state
    