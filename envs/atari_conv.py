import gym
import cv2
import numpy as np
from envs._interface import EnvInterface


class Env(EnvInterface):
    def __init__(self, env_name):  
        super(Env, self).__init__(env_name)       

    def factory(env_name):
        return lambda : Env(env_name)

    def _preprocess(self, state):
        state = cv2.resize(np.float32(state), (self._img_size, self._img_size), interpolation=cv2.INTER_LINEAR)
        state = cv2.cvtColor(np.float32(state), cv2.COLOR_BGR2GRAY)
        state = state / 255  # transforma a imagem num vetor e normaliza para 0~1
        return state
    