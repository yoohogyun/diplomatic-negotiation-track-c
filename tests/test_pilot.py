import unittest
from dataclasses import replace
import numpy as np
import pilot
from pilot.env import *

class PilotTests(unittest.TestCase):
    def test_bayes(self):
        self.assertAlmostEqual(posterior(.4,.85,.15)[0],.34/.43)
        self.assertEqual(posterior(.4,.5,.5)[0],.4)
        with self.assertRaises(ValueError): posterior(1,0,1)
    def test_hidden_type(self):
        e=NegotiationEnv(); a,_=e.reset(seed=1,options={'hard':False}); b,_=e.reset(seed=1,options={'hard':True})
        np.testing.assert_array_equal(a,b)
    def test_text_changes_acceptance_and_terms(self):
        e=NegotiationEnv(); e.reset(seed=1,options={'hard':True})
        e.step_text(render(6)); self.assertFalse(e.s.accepted) # 2 for 2
        e.reset(seed=1,options={'hard':True}); e.step_text(render(8)) # 3 for 2
        self.assertTrue(e.s.accepted); self.assertEqual(e.s.x,3)
        with self.assertRaises(ValueError): parse(render(6).replace(' if ',' unless '))
        with self.assertRaises(ValueError): parse(render(6).replace('will suspend','will not suspend'))
    def test_signal_cost_is_read_from_text(self):
        e=NegotiationEnv(); e.reset(seed=1); _,r1,_,_,_=e.step_text(render(1,Config(signal_cost=.1)))
        e.reset(seed=1); _,r2,_,_,_=e.step_text(render(1,Config(signal_cost=.2)))
        self.assertAlmostEqual(r1-r2,.1)
    def test_no_unilateral_ratification(self):
        e=NegotiationEnv(); e.reset(seed=1); _,_,d,_,i=e.step(9)
        self.assertFalse(d); self.assertNotEqual(i['outcome'],'agreement')
        e.step_text(render(7)) # self utility -0.04, opponent can accept
        _,_,d,_,i=e.step(9); self.assertNotEqual(i['outcome'],'agreement')
    def test_rewards_and_end(self):
        e=NegotiationEnv(); e.reset(seed=4,options={'hard':False}); costs=0.; rewards=0.; term=0.
        for a in [2,6,9]:
            _,r,d,_,i=e.step(a); costs+=sum(i['costs'].values()); rewards+=r; term+=i['terminal_utility']
        self.assertTrue(d); self.assertAlmostEqual(rewards,term-costs)
        with self.assertRaises(RuntimeError): e.step(10)
        e.reset(seed=4)
        for _ in range(5): _,_,d,_,i=e.step(0)
        self.assertTrue(d); self.assertEqual(i['outcome'],'exhausted')
    def test_rejection_is_information(self):
        e=NegotiationEnv(); e.reset(seed=1,options={'hard':True}); e.step(6)
        self.assertEqual(e.s.b,1.)
        e.reset(seed=1,options={'hard':False}); e.step(6); self.assertEqual(e.s.b,0.)
    def test_duplicate_question_and_promise(self):
        e=NegotiationEnv(); e.reset(seed=1); e.step(2); b=e.s.b; _,_,_,_,i=e.step(2)
        self.assertEqual(e.s.b,b); self.assertEqual(i['costs']['query'],e.cfg.query_cost)
        e.reset(seed=1); e.step(8); e.step(0); _,_,_,_,i=e.step(6)
        self.assertEqual(i['reversal'],1); self.assertGreater(i['costs']['reversal'],0)
    def test_planner_query_ablation(self):
        for c in [Config(),Config(rejection_cost=0),Config(reliability=.5)]:
            p=Planner(c); q=Planner(c,False)
            self.assertGreaterEqual(p.solve(Public(b=c.prior))[0]+1e-10,q.solve(Public(b=c.prior))[0])
    def test_gym_checker(self):
        from gymnasium.utils.env_checker import check_env
        check_env(NegotiationEnv(),skip_render_check=True)

if __name__=='__main__': unittest.main()
