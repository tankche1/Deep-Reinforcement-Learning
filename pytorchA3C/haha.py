import gym
env = gym.make('Breakout-v3')
env.reset()
for i in range(10000):
    env.render()
    env.step(1)
