import cProfile
import multiprocessing as mp
import numpy as np
import os
import pstats
import StringIO
import sys
import time
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import dvi
import mdps
import utils

class TestParallelDiscreteValueIteration(unittest.TestCase):

    def test_solve(self):
        mdp = mdps.LineMDP(length=100 + 1)
        num_processes = 2
        max_iterations = 105
        min_residual = 1e-4
        verbose = 1
        solver = dvi.ParallelDiscreteValueIteration(mdp, num_processes, max_iterations, min_residual, verbose)

        start = time.time()
        pr = cProfile.Profile()
        pr.enable()
        qvalues = solver.solve()
        pr.disable()
        sortby = 'cumulative'
        s = StringIO.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        end = time.time()
        print 'total time: {}'.format(end - start)
        print s.getvalue()

        print qvalues
        output_filepath = '../policies/qvalues.npz'
        utils.save_qvalues(qvalues, output_filepath)

    def test_solve_chunk(self):
        mdp = mdps.LineMDP(length=100 + 1)
        num_processes = 1
        max_iterations = 1
        min_residual = 1e-4
        verbose = 1
        solver = dvi.ParallelDiscreteValueIteration(mdp, num_processes, max_iterations, min_residual, verbose)
        state_values = solver.state_values
        qvalues = solver.qvalues
        state_idxs = xrange(mdp.num_states)
        queue = mp.Queue()

        pr = cProfile.Profile()
        pr.enable()
        solver.solve_chunk(mdp, state_values, qvalues, state_idxs, queue)
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
        queue.get()




if __name__ == '__main__':
    unittest.main()

