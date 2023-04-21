#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from collections import namedtuple
import math 
import csv
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""
import time
import os


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

GAMMA = 0.99

EPSILON = 1
EPS_START = EPSILON
EPS_END = 0.025
EPS_DECAY = 1000
batch_size = 32
ALPHA = 1e-5
T_U = 20000

# Parameters for Replay Buffer
CAPACITY = 5000 
memory = deque(maxlen=CAPACITY) 
storeEpsilon = []
StartLearning = 1000
LOAD = True


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.env = env
        s = env.reset()
        s = s.transpose(2,0,1)

        self.policy_net = DQN(s.shape, self.env.action_space.n) 
        self.target_net = DQN(s.shape, self.env.action_space.n) 
        self.target_net.load_state_dict(self.policy_net.state_dict())
        

        if USE_CUDA:
            print("Using CUDA . . .     ")
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()

        print('hyperparameters and network initialized')
                
        
        if args.test_dqn or LOAD == True:
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            cp = torch.load('trainData')
            self.policy_net.load_state_dict(cp['model_state_dict'])
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        observation = observation.transpose(2,0,1)
        
        if np.random.random() > EPSILON or test==True:
            observation   = Variable(torch.FloatTensor(np.float32(observation)).unsqueeze(0), volatile=True)
            q_value = self.policy_net.forward(observation)
            action  = q_value.max(1)[1].data[0]
            action = int(action.item())            
        else:
            action = random.randrange(4)
        
        ###########################
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        memory.append((state, action, reward, next_state, done))
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        state, action, reward, next_state, done = zip(*random.sample(memory, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
        
        ###########################
    def __len__(self):
        return len(self.buffer)       

    def optimize_model(self):

        states, actions, next_states, rewards, dones  = self.replay_buffer()

        s_v = Variable(torch.FloatTensor(np.float32(states)))
        n_s_v = Variable(torch.FloatTensor(np.float32(next_states)), volatile=True)
        a_v = Variable(torch.LongTensor(actions))
        r_v = Variable(torch.FloatTensor(rewards))
        done = Variable(torch.FloatTensor(dones))

        s_a_v = self.policy_net(s_v).gather(1, a_v.unsqueeze(1)).squeeze(1)
        next_state_values = self.target_net(n_s_v).max(1)[0]
        e_q_v = r_v + next_state_values * GAMMA * (1 - done) 

        loss = (s_a_v - Variable(e_q_v.data)).pow(2).mean()
        return  loss

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        opt = optim.Adam(self.policy_net.parameters(), lr=ALPHA)

        
        print('Gathering experiences ...')
        mScore = 0
        A_R = []
        AllS = []
        st = 1
        Ie = 0

        while mScore < 30:
                     
            state = self.env.reset()
            done = False
            epi_s = 0
            tBegin = time.time()
            done = False

            while not done:

                action = self.make_action(state)    
                nextState, reward, done, _ = self.env.st(action)
                self.push(state.transpose(2,0,1), action, nextState.transpose(2,0,1), reward, done)

                state = nextState   
                
                if len(memory) > StartLearning:
                    loss = self.optimize_model()
                    opt.zero_grad()
                    loss.backward()
                    opt.st()
                else:
                    Ie = 0
                    continue        

                
                EPSILON = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * st/EPS_DECAY)
                storeEpsilon.append(EPSILON)
                st += 1
                
                epi_s += reward

                if st % T_U == 0:
                    print('Updating Target Network . . .')
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            Ie += 1
            AllS.append(epi_s)
            mScore = np.mean(AllS[-100:])
            A_R.append(mScore)
            
            if len(memory) > StartLearning: 
                print('Episode: ', Ie, ' score:', epi_s, ' Avg Score:',mScore,' epsilon: ', EPSILON, ' t: ', time.time()-tBegin, ' loss:', loss.item())
            else:
                print('Gathering Data . . .')

            if Ie % 500 == 0:
                torch.save(self.policy_net.state_dict(),'pkdqn.pth')
        print('======== Complete ========')
        torch.save(torch.save(self.policy_net.state_dict(),'pkdqn.pth'))
        
        
        ###########################
