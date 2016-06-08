"""
Parallel implementation of value iteration.
"""

import numpy as np

import utils

class ParallelDiscreteValueIteration(object):

    def __init__(self, mdp, num_processes, max_iterations, min_residual):
        self.mdp = mdp
        self.num_processes = num_processes
        self.max_iterations = max_iterations
        self.min_residual = min_residual
        self.segmented_state_idxs = utils.segment_state_indices(self.mdp.num_states, self.num_processes)
        self.init_value_arrays()
        
    def init_value_arrays(self):
        # allocate the state value and state-action value arrays
        self.state_values = np.zeros((self.mdp.num_states))
        self.qvalues = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        
    def solve(self):
        # loop max iterations time performing a complete state-action pair update
        for idx in xrange(self.max_iterations):
            residual = self.solve_step()

            # break once converged
            if residual < self.min_residual:
                break

        # return the qvalues
        return self.qvalues

    def solve_step(self):
        # loop over chunks of the state space updating the value arrays
        max_residual = 0
        for state_idxs in self.segmented_state_idxs:
            cur_residual = self.solve_chunk(state_idxs)
            max_residual = max(max_residual, cur_residual)
        return max_residual

    def solve_chunk(self, state_idxs):
        # loops over states and actions updating value arrays
        max_residual = 0
        for state_idx in state_idxs:

            # track original value for residual computation
            original_state_value = self.state_values[state_idx]

            for action_idx in xrange(self.mdp.num_actions):

                # get the next states and their probs
                next_state_idxs, probs = self.mdp.next_states_probs(state_idx, action_idx)

                # always account for the reward for taking the action in this state
                state_action_reward = self.mdp.reward(state_idx, action_idx)
                new_value = state_action_reward

                # if we haven't reached a terminal state then also consider values of next states
                if len(next_state_idxs) > 0:
                    next_values = self.state_values[next_state_idxs] * probs * self.mdp.discount
                    new_value += np.sum(next_values)

                # update the qvalues
                self.qvalues[state_idx, action_idx] = new_value

                # update the state values
                # if this is the first action then set value to be for that action
                if action_idx == 0:
                    self.state_values[state_idx] = new_value
                else:
                    self.state_values[state_idx] = max(self.state_values[state_idx], new_value)

            # calculate and return residual
            cur_residual = abs(self.state_values[state_idx] - original_state_value)
            max_residual = max(cur_residual, max_residual)

        return max_residual
