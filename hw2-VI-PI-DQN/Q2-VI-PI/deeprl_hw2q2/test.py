# -*- coding: utf-8 -*-
import gym
import rl
import numpy as np

env = gym.make('Deterministic-4x4-FrozenLake-v0')
gamma = 0.9

value_func, num_iter = rl.value_iteration_async_ordered(env, gamma)
print(num_iter)
print(value_func.reshape(4,4))

