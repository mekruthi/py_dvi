import numpy as np
import os
import sys
import time
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import dvi
import mdps
import utils

class TestParallelDiscreteValueIteration(unittest.TestCase):

    def test_init(self):
        mdp = mdps.LineMDP(length=1001)
        num_processes = 1
        max_iterations = 1002
        min_residual = 1e-4
        solver = dvi.ParallelDiscreteValueIteration(mdp, num_processes, max_iterations, min_residual)

        start = time.time()
        qvalues = solver.solve()
        end = time.time()
        print 'total time: {}'.format(end - start)

        print qvalues
        output_filepath = '../policies/qvalues.npz'
        utils.save_qvalues(qvalues, output_filepath)

if __name__ == '__main__':
    unittest.main()