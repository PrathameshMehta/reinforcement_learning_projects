import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
	Monte-Carlo
	In this problem, you will implememnt an AI player for Blackjack.
	The main goal of this problem is to get familar with Monte-Carlo algorithm.
	You could test the correctness of your code 
	by typing 'nosetests -v mc_test.python3' in the terminal.
	
	You don't have to follow the comments to write your code. They are provided
	as hints in case you need. 
'''
#-------------------------------------------------------------------------

def initial_policy(ob):
	"""A policy that sticks if the player score is >= 20 and his otherwise
	
	Parameters:
	-----------
	observation:
	Returns:
	--------
	action: 0 or 1
		0: STICK
		1: HIT
	"""
	print(ob)
	if(ob[0]>=20):
		ac = 0
	else:
		ac = 1	
	return ac 

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
	"""Given policy using sampling to calculate the value function 
		by using Monte Carlo first visit algorithm.
	
	Parameters:
	-----------
	policy: function
		A function that maps an obversation to action probabilities
	env: function
		OpenAI gym environment
	n_episodes: int
		Number of episodes to sample
	gamma: float
		Gamma discount factor
	Returns:
	--------
	V: defaultdict(float)
		A dictionary that maps from state to value
	"""
	# initialize empty dictionaries

	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# a nested dictionary that maps state -> value
	V = defaultdict(float)

	for i_episodes in range(n_episodes):
		previous_obs = env.reset()
		ep = []
		term = False
		while(term == False):
			action = policy(previous_obs) 
			new_obs, r, done, _, _ = env.step(action)
			ep.append([previous_obs,action,r])
			previous_obs = new_obs
			term = done
		Gval = defaultdict(float)
		i = len(ep)
		G = 0
		for [observ, action, r] in reversed(ep):
			G = gamma*G + r
			Gval[i] = G
			i -= 1
		i = 1
		s_v = []

		for [observ, action, r] in ep:
			if observ not in s_v:
				returns_count[observ] += 1 
				returns_sum[observ] += Gval[i]
				V[observ] = returns_sum[observ]/returns_count[observ]
				s_v.append(observ)
			i += 1 
	return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
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
	------
	With probability (1 − epsilon) choose the greedy action.
	With probability epsilon choose an action at random.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	sap = Q[state]
	b_pol = np.argmax(sap)
	print(epsilon)
	print(nA)
	pol = np.ones(nA, float)*(epsilon/nA)
	pol[b_pol] = (epsilon/nA) + 1 - epsilon

	action = np.random.choice(np.arange(len(sap)), p = pol)

	############################
	return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
	"""Monte Carlo control with exploring starts. 
		Find an optimal epsilon-greedy policy.
	
	Parameters:
	-----------
	env: function
		OpenAI gym environment
	n_episodes: int
		Number of episodes to sample
	gamma: float
		Gamma discount factor
	epsilon: float
		The probability to select a random action, range between 0 and 1
	Returns:
	--------
	Q: dict()
		A dictionary  that maps from state -> action-values,
		where Q[s][a] is the estimated action value corresponding to state s and action a.
	Hint:
	-----
	You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
	and episode must > 0.    
	"""
	#{

	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# a nested dictionary that maps state -> (action -> action-value)
	# e.g. Q[state] = np.darrary(nA)
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	
	############################
	# YOUR IMPLEMENTATION HERE #
	for i_episodes in range(n_episodes):
		if epsilon <= 0.05:
			epsilon = 0.05 
		else:
			epsilon = epsilon - (0.1/n_episodes)
		previous_obs = env.reset()
		term = False
		episode = []
		while(term == False):
			action = epsilon_greedy(Q, previous_obs, env.action_space.n, epsilon)
			new_obs, r, done, _, _ = env.step(action)
			episode.append([previous_obs, action, r])
			previous_obs = new_obs
			term = done
		G = 0
		Gval = defaultdict(float) 
		i = len(episode)
		for [observation, action, r] in reversed(episode):
			a_S = (observation,action) 
			G = gamma*G + r
			Gval[i] = G
			i -= 1 
		visited_a_S = []
		i = 1
		for [observation, action, r] in episode:
			a_S = (observation,action) 
			if a_S not in visited_a_S:
				returns_count[a_S] += 1
				returns_sum[a_S] += Gval[i]
				Q[observation][action] = returns_sum[a_S]/returns_count[a_S]
				visited_a_S.append(a_S)
			i += 1 	
	return Q