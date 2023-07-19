import gymnasium as gym
from gymnasium import spaces
import numpy as np
import itertools
import time
import torch
import pylab as plt
# %matplotlib inline
# %matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from SAC.sac import SACAgent

env_name = 'Pendulum-v1'
# env_name = 'CartPole-v1'

env = gym.make(env_name)

ac_space = 5
o_space = env.observation_space.shape[0]
print(ac_space)
print(o_space)
print(list(zip(env.observation_space.low, env.observation_space.high)))

q_agent = SACAgent(o_space, ac_space)

ob,_info = env.reset()
q_agent.select_action(ob)

stats = []
losses = []
max_episodes = 600
max_steps = 500
for i in range(max_episodes):
    # print("Starting a new episode")
    total_reward = 0
    ob, _info = env.reset()
    for t in range(max_steps):
        done = False
        a = q_agent.select_action(ob)
        (ob_new, reward, done, trunc, _info) = env.step(a)
        total_reward += reward
        q_agent.update(ob, a, reward, ob_new, done)
        ob = ob_new
        if done: break

    stats.append([i, total_reward, t + 1])

    if ((i - 1) % 20 == 0):
        print("{}: Done after {} steps. Reward: {}".format(i, t + 1, total_reward))