import gym
from envs._interface import EnvInterface


class Env(EnvInterface):
    def __init__(self, env_name):
        super(Env, self).__init__(env_name)

    def factory(env_name):
        return lambda : Env(env_name)

    def _preprocess(self, state):
        return state