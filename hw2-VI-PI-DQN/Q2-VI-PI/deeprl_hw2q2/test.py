# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import time
import rl

env = gym.make('Deterministic-4x4-FrozenLake-v0')
gamma = 0.9

policy, value_func, num_impro_iter_total, num_eval_iter_total = rl.policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
print(num_impro_iter_total)
print(num_eval_iter_total)

