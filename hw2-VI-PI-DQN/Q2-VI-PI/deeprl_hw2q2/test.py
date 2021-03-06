# -*- coding: utf-8 -*-
import gym
import rl
import numpy as np
import time

env = gym.make('Deterministic-4x4-FrozenLake-v0')
gamma = 0.9

start = time.time()
# policy, value_func, num_impro_iter_total, num_eval_iter_total = rl.policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
# value_func, num_iter, policy = rl.value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3)
# =============================================================================
# 
# 
# 
# num_impro_iter_randperm = 0
# num_eval_iter_randperm = 0
# for i in range(10):
#     policy, value_func, num_impro_iter_total, num_eval_iter_total = rl.policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
#                                     tol=1e-3)
#     num_impro_iter_randperm += num_impro_iter_total
#     num_eval_iter_randperm += num_eval_iter_total
#     
# num_impro_iter_randperm /= 10
# num_eval_iter_randperm /= 10
# 
# policy, value_func, num_impro_iter_ordered, num_eval_iter_ordered = rl.policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
#                                    tol=1e-3)
# =============================================================================
value_func, num_iter_ordered = rl.value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3)
value_func, num_iter_randperm = rl.value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3)

end = time.time()
# Plotting policy letters for 2.2 & 2.6
# rl.display_policy_letters(env, policy)
# print(policy)
# Plotting heatmaps for 2.3 & 2.5
rl.value_func_heatmap(env, value_func)
# 2.7 runtime comparison
print('Time it took to run the algorithm: ',(end-start))