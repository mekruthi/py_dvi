import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import utils

class TestSegmentStateIndices(unittest.TestCase):

    def test_segment_state_indices(self):
        num_states = 8
        num_processes = 3
        state_segments = utils.segment_state_indices(num_states, num_processes)
        expected = [[0, 1], [2, 3], [4, 5, 6, 7]]
        self.assertTrue(np.array_equal(state_segments, expected))

if __name__ == '__main__':
    unittest.main()