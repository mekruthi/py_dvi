"""
Parallel implementation of value iteration.
"""

import multiprocessing as mp
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
        self.state_values = mp.Array('f', self.mdp.num_states, lock=False)
        self.qvalues = mp.Array('f', self.mdp.num_states * self.mdp.num_actions, lock=False)
        
    def solve(self):
        # create the pool of processes
        pool = mp.Pool(self.num_processes)

        # loop max iterations time performing a complete state-action pair update
        for idx in xrange(self.max_iterations):
            residual = self.solve_step(pool)

            # break once converged
            if residual < self.min_residual:
                break

        # return the qvalues
        qvalues = np.array(self.qvalues).reshape(self.mdp.num_states, self.mdp.num_actions)
        return qvalues

    def solve_step(self, pool):
        # loop over chunks of the state space updating the value arrays
        processes = []
        queue = mp.Queue()
        for state_idxs in self.segmented_state_idxs:
            p = mp.Process(target=self.solve_chunk, args=(self.mdp, self.state_values, self.qvalues, state_idxs, queue))
            p.start()
            processes.append(p)

        # retrieve max residual value across processes
        max_residual = 0
        for p in processes:
            max_residual = max(queue.get(), max_residual)
            p.join()

        return max_residual

    @staticmethod
    def solve_chunk(mdp, state_values, qvalues, state_idxs, queue):

        # loops over states and actions updating value arrays
        max_residual = 0
        for state_idx in state_idxs:

            # track original value for residual computation
            original_state_value = state_values[state_idx]

            for action_idx in xrange(mdp.num_actions):

                # get the next states and their probs
                next_state_idxs, probs = mdp.next_states_probs(state_idx, action_idx)

                # always account for the reward for taking the action in this state
                state_action_reward = mdp.reward(state_idx, action_idx)
                new_value = state_action_reward

                # if we haven't reached a terminal state then also consider values of next states
                if len(next_state_idxs) > 0:
                    for idx, next_state_idx in enumerate(next_state_idxs):
                        new_value += state_values[next_state_idx] * probs[idx] * mdp.discount

                # update the qvalues
                qvalues[state_idx * mdp.num_actions + action_idx] = new_value

                # update the state values
                # if this is the first action then set value to be for that action
                if action_idx == 0:
                    state_values[state_idx] = new_value
                else:
                    state_values[state_idx] = max(state_values[state_idx], new_value)

            # calculate and return residual
            cur_residual = abs(state_values[state_idx] - original_state_value)
            max_residual = max(cur_residual, max_residual)

        # insert max residual into queue for retrieve by solve_step
        queue.put(max_residual)
