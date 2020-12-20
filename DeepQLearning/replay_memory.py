#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:49:51 2020

@author: samuel
"""

import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, action_space):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, action_space), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, action_space), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size, input_shape, action_space, prob_alpha):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, action_space), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, action_space), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.priorities = np.zeros((self.mem_size,), dtype = np.float32)
        self.prob_alpha = prob_alpha
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.priorities[index] = self.priorities.max() if self.mem_cntr > 0 else 1
        self.mem_cntr += 1

    def sample_buffer(self, batch_size, beta=0.4):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        prios = self.priorities[:max_mem]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        
        total = max_mem        
        
        batch = np.random.choice(max_mem, batch_size, p = probs,replace=False)
        
        weights = (total * probs[batch]) ** (-beta)
        weights /= weights.max()

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal, batch, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio