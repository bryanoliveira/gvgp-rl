import a3c
import gym

env = gym.make("Pong-v0")
state = env.reset()
processed = a3c.preprocess(state)
print(processed.shape, processed.size(0))
processed = processed.view(-1)
print(processed.shape)