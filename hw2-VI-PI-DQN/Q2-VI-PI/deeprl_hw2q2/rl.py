# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import lake_envs as lake_env


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.
    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    num_iter = 0
    delta = 0

    while (True):
        num_iter+=1
        value_func_old = value_func.copy()
        for i in range(env.nS):
            v = value_func_old[i]
            prob, nextstate, reward, is_terminal = env.P[i][policy[i]][0]
            v_new = prob * (reward + gamma * value_func_old[nextstate])
            value_func[i] = v_new
            if delta < abs(v-v_new):
                delta = abs(v-v_new)
        if delta<tol:
            break
        if num_iter > max_iterations:
            break

    return value_func, num_iter


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function

    for num_iter in range(max_iterations):
        delta = 0
        for state in range(env.nS):
            # Run the Bellman Expectation Backup Operation
            value_old = value_func[state]
            prob, nextstate, reward, is_terminal = env.P[state][policy[state]][0]
            value_func[state] = prob * (reward + gamma * value_func[nextstate])
            delta = max(delta, abs(value_old - value_func[state]))
        # Check convergence
        if delta<tol:
            break
    num_iter += 1 # calculate the total number of value iteration
    return value_func, num_iter


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    
    for num_iter in range(max_iterations):
        delta = 0
        # generate random permutation of state
        randperm = np.arange(env.nS)
        np.random.shuffle(randperm)
        for state in randperm:
            # Run the Bellman Expectation Backup Operation
            value_old = value_func[state]
            prob, nextstate, reward, is_terminal = env.P[state][policy[state]][0]
            value_func[state] = prob * (reward + gamma * value_func[nextstate])
            delta = max(delta, abs(value_old - value_func[state]))
        # Check convergence
        if delta<tol:
            break
    num_iter += 1 # calculate the total number of value iteration
    return value_func, num_iter



def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_change = False
    for state in range(env.nS):
        q = np.zeros(env.nA)
        for action in range(env.nA):
            prob, nextstate, reward, is_terminal = env.P[state][action][0]
            q[action] = prob * (reward + gamma * value_func[nextstate])
        if policy[state] != np.argmax(q):
            policy_change = True
            policy[state] = np.argmax(q)
    return policy_change, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    policy_change = True
    num_impro_iter_total = 0
    num_eval_iter_total = 0
    while(policy_change):
        value_func, num_iter = evaluate_policy_sync(env, gamma, policy)
        policy_change, policy = improve_policy(env, gamma, value_func, policy)
        num_impro_iter_total += 1
        num_eval_iter_total += num_iter
    return policy, value_func, num_impro_iter_total, num_eval_iter_total

# Result: 14 improvement iteration and 119 evaluation iteration
def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    
    policy_change = True
    num_impro_iter_total = 0
    num_eval_iter_total = 0
    while(policy_change):
        value_func, num_iter = evaluate_policy_async_ordered(env, gamma, policy)
        policy_change, policy = improve_policy(env, gamma, value_func, policy)
        num_impro_iter_total += 1
        num_eval_iter_total += num_iter
    return policy, value_func, num_impro_iter_total, num_eval_iter_total

# Result: (14,119)/(14,115)/(14,119)/(14,117)/(14,118)/(14,113)/(14,119)/(14,119)/(14,118)/(14,116)
def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    
    policy_change = True
    num_impro_iter_total = 0
    num_eval_iter_total = 0
    while(policy_change):
        value_func, num_iter = evaluate_policy_async_randperm(env, gamma, policy)
        policy_change, policy = improve_policy(env, gamma, value_func, policy)
        num_impro_iter_total += 1
        num_eval_iter_total += num_iter
    return policy, value_func, num_impro_iter_total, num_eval_iter_total

def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    policy = np.zeros(env.nS, dtype='int')
    for num_iter in range(max_iterations):
        delta = 0
        value_func_old = value_func.copy()
        for state in range(env.nS):
            # Get the max value over all policies/actions
            values = np.zeros(env.nA)
            for action in range(env.nA):
                prob, nextstate, reward, is_terminal = env.P[state][action][0]
                values[action] = prob * (reward + gamma * value_func_old[nextstate])
            value_func[state] = np.max(values)
            policy[state] = policy[state] = np.argmax(values)
            delta = max(delta, abs(value_func_old[state] - value_func[state]))
        # Check convergence
        if delta<tol:
            break
    num_iter += 1 # calculate the total number of value iteration
    return value_func, num_iter, policy

# Result of one test: 15
def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    for num_iter in range(max_iterations):
        delta = 0
        for state in range(env.nS):
            # Get the max value over all policies/actions
            values = np.zeros(env.nA)
            for action in range(env.nA):
                prob, nextstate, reward, is_terminal = env.P[state][action][0]
                values[action] = prob * (reward + gamma * value_func[nextstate])
            value_old = value_func[state]
            value_func[state] = np.max(values)
            delta = max(delta, abs(value_old - value_func[state]))
        # Check convergence
        if delta<tol:
            break
    num_iter += 1 # calculate the total number of value iteration
    return value_func, num_iter

# Result of ten test: 8/7/10/10/10/10/8/11/12/10
def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    for num_iter in range(max_iterations):
        delta = 0
        randperm = np.arange(env.nS)
        np.random.shuffle(randperm)
        for state in randperm:
            # Get the max value over all policies/actions
            values = np.zeros(env.nA)
            for action in range(env.nA):
                prob, nextstate, reward, is_terminal = env.P[state][action][0]
                values[action] = prob * (reward + gamma * value_func[nextstate])
            value_old = value_func[state]
            value_func[state] = np.max(values)
            delta = max(delta, abs(value_old - value_func[state]))
        # Check convergence
        if delta<tol:
            break
    num_iter += 1 # calculate the total number of value iteration
    return value_func, num_iter


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)  # initialize value function
    return value_func, 0


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 2.2 & 2.6

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    from gym.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
    policy_letters = []
    action_names = {LEFT: 'LEFT', RIGHT: 'RIGHT', DOWN: 'DOWN', UP: 'UP'}
    
    for l in policy:
        policy_letters.append(action_names[l][0])
    
    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)
    

    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)
    
    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))
    
    for state in range(env.nS):
      for action in range(env.nA):
        for prob, nextstate, reward, is_terminal in env.P[state][action]:
            env.T[state, action, nextstate] = prob
            env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 2.3 & 2.5

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7,6)) 
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]), 
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels = np.arange(1, env.nrow+1)[::-1], 
                xticklabels = np.arange(1, env.nrow+1)) 
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None
