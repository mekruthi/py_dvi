

import collections
import copy
import numpy as np
import random
import sys

class LineMDP(object):
    """
    :description: A line mdp is just an x axis. Here the rewards are all -1 except for the last state on the right which is +1.
    Also assume a minimum position on the left of zero
    """

    def __init__(self, length):
        self.length = length
        self.states = range(length + 1)
        self.num_states = self.length + 1
        self.actions = [-1, 1]
        self.num_actions = 2
        self.exit_reward = 1
        self.move_reward = -1
        self.discount = 1.

    def next_states_probs(self, state_idx, action_idx):
        # convert indices to actual values 
        state = self.states[state_idx]
        action = self.actions[action_idx]

        # terminal
        if state == self.length:
            next_states, probs = np.array([]), np.array([])
        else:
            next_states, probs = np.array([max(0, state + action)]), np.array([1])

        return next_states, probs

    def reward(self, state_idx, action_idx):
        # convert indices to actual values 
        state = self.states[state_idx]
        action = self.actions[action_idx]

        next_state = max(0, state + action)
        if state == self.states[-1]:
            next_state = self.length + 1

        if next_state == self.length + 1:
            reward = 0
        elif next_state == self.length:
            reward = self.exit_reward
        else:
            reward = self.move_reward

        return reward