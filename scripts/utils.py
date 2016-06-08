"""
Helper functions for dvi
"""

import numpy as np

def save_qvalues(qvalues, output_filepath):
    # save the qvalue array to file
    np.savez(output_filepath, qvalues=qvalues)

def segment_state_indices(num_states, num_processes):
    # segments the state indices into chunks to distribute between processes
    state_idxs = np.arange(num_states)
    num_uneven_states = num_states % num_processes
    if num_uneven_states == 0:
        segmented_state_idxs = state_idxs.reshape(num_processes, -1)
    else:
        segmented_state_idxs = state_idxs[:num_states - num_uneven_states].reshape(num_processes, -1).tolist()
        segmented_state_idxs[-1] = np.hstack((segmented_state_idxs[-1], state_idxs[-num_uneven_states:])).tolist()

    return segmented_state_idxs