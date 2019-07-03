import gym
import logging


class EnvInterface:
    def __init__(self, env_name, _img_size=84):
        self._img_size = _img_size

        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.n_obs = self.env.observation_space.shape[0]
        self.name = env_name

        logging.info("Instantiated " + env_name)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = self._preprocess(state)
        return state, reward, done, info

    def reset(self):
        state = self._preprocess(self.env.reset())
        return state
    
    def render(self):
        self.env.render()

    def factory(env_name):
        raise NotImplementedError()
        return None

    def _preprocess(self, state):
        raise NotImplementedError()
        return None
    
    def close(self):
        self.env.close()

    def sample_action(self):
        return self.env.action_space.sample()