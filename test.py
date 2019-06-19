import gym_gvgai

env = gym_gvgai.make("gvgai-cec3-lvl0-v0")
obs = env.reset()
while True:
    env.render()
    _, r, done, _ = env.step(env.action_space.sample())
    print(r)
    if done:
        env.reset()