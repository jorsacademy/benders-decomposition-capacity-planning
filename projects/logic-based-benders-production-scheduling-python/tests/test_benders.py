
import unittest, itertools, numpy as np
from lbbenders import *

class Tests(unittest.TestCase):
    def test_line_subproblem_exact_permutation(self):
        inst=generate_instance(2,5,2,3)
        jobs=(0,1,2)
        c,seq=exact_line_sequence(inst,0,jobs)
        vals=[]
        for perm in itertools.permutations(jobs):
            total=0
            for k,j in enumerate(perm):
                total+=inst.processing[j,0]
                if k: total+=inst.setup[0,inst.family[perm[k-1]],inst.family[j]]
            vals.append(total)
        self.assertEqual(c,min(vals))

    def test_benders_matches_full_assignment_enumeration(self):
        for seed in [3,4,5]:
            inst=generate_instance(seed,6,2,3)
            b=solve_logic_benders(inst)
            e=brute_force_global(inst)
            self.assertEqual(b.status,"OPTIMAL")
            self.assertAlmostEqual(b.objective,e[0],places=7)

    def test_infeasible_assignment_detected(self):
        inst=generate_instance(6,6,2,3)
        a=(0,)*inst.n_jobs
        s=solve_subproblem(inst,a)
        self.assertFalse(s.feasible)

    def test_master_assigns_each_job_once(self):
        inst=generate_instance(7,5,2,2)
        m=solve_master(inst,[],1000)
        self.assertEqual(len(m.assignment),inst.n_jobs)
        self.assertTrue(all(0<=x<inst.n_lines for x in m.assignment))

if __name__=="__main__":unittest.main()
