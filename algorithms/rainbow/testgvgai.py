import gym
import gym_gvgai
import time
import random

env = gym_gvgai.make("gvgai-cec1-lvl0-v0")
obs = env.reset()

lt = time.time()
while True:
    action = random.randint(0, len(env.env.GVGAI.actions()) - 1)
    env.step(action)
    print(lt - time.time())
    lt = time.time()
