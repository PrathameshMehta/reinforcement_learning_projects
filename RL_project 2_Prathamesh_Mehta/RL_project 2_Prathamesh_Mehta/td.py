#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict

import numpy as np

"""
    Temporal Difference
    In this problem, you will implement an AI player for cliff-walking.
    The main goal of this problem is to get familiar with temporal difference algorithm.
    You could test the correctness of your code
    by typing 'nosetests -v td_test.py' in the terminal.
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
"""


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """

    g_i = np.argmax(Q[state])
    prob = np.ones(nA)*epsilon/nA  
    prob[g_i] += 1 - epsilon  
    return np.random.choice(np.arange(len(Q[state])), p=prob)


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hints:
    -----
    You could consider decaying epsilon, i.e. epsilon = 0.99*epsilon during each episode.
    """
    a_c = env.action_space.n
    Q = defaultdict(lambda: np.zeros(a_c))
    for i in range(n_episodes):
        epsilon = 0.99*epsilon  
        c_s = env.reset()  
        c_a = epsilon_greedy(Q, c_s, a_c, epsilon)  
        done = False
        while not done:  
            n_s, reward, done, _, _ = env.step(c_a)  
            n_a = epsilon_greedy(Q, n_s, a_c, epsilon)  
            td_tar = reward + gamma*Q[n_s][n_a]  
            td_err = td_tar - Q[c_s][c_a] 
            Q[c_s][c_a] = Q[c_s][c_a] + alpha*td_err  
            c_s = n_s 
            c_a = n_a  
    return Q


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    """
    a_c = env.action_space.n
    Q = defaultdict(lambda: np.zeros(a_c))
    for i in range(n_episodes):
        epsilon = 0.99*epsilon  
        c_s = env.reset()  
        done = False
        while not done:  
            c_a = epsilon_greedy(Q, c_s, a_c, epsilon)  
            n_s, reward, done, _, _ = env.step(c_a)          
            b_a = np.argmax(Q[n_s])
            td_tar = reward + gamma*Q[n_s][b_a] 
            td_err = td_tar - Q[c_s][c_a]  
            Q[c_s][c_a] += alpha*td_err  
            c_s = n_s  

    return Q
