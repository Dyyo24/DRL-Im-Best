# -*- coding: utf-8 -*-
import gym
import rl

env = gym.make('Deterministic-4x4-FrozenLake-v0')
gamma = 0.9

value_func, num_iter = rl.value_iteration_sync(env, gamma)
print(num_iter)
print(value_func)

