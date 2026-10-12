import unittest,json
import numpy as np
from study import OUT
from study.env import LunaEnv
from pilot.env import Config

@unittest.skipUnless((OUT/'language_bank.json').exists(),'Wait for frozen Luna bank')
class StudyTests(unittest.TestCase):
    def test_checker(self):
        from gymnasium.utils.env_checker import check_env
        check_env(LunaEnv(),skip_render_check=True)
    def test_source_group_split(self):
        rows=json.loads((OUT/'contexts.json').read_text(encoding='utf-8')); seen={}
        for r in rows:
            for key in (r['event_group'],r['source']['url_or_doi'],r['source']['sentence']):
                if key in seen: self.assertEqual(seen[key],r['split'])
                seen[key]=r['split']
    def test_hidden_type(self):
        e=LunaEnv(); a,_=e.reset(seed=12,options={'hard':True}); b,_=e.reset(seed=12,options={'hard':False})
        np.testing.assert_equal(a,b)
    def test_interpretation_controls_transition(self):
        e=LunaEnv(); e.reset(seed=1,options={'hard':True,'variant':0}); s=e.s
        _,_,_,_,info=e.step_meaning({'kind':'offer','x':3,'y':2},'a synthetic offer parsed as 3 and 2',action=3)
        self.assertEqual(info['after']['x'],3); self.assertEqual(info['after']['y'],2)
        self.assertTrue(e.s.accepted) # actual terms, not intended action 3 = 1/1
    def test_common_reward_for_ablation(self):
        e=LunaEnv(remove_reward='rejection'); e.reset(seed=1,options={'config':Config(),'hard':True})
        _,r,_,_,i=e.step_meaning({'kind':'offer','x':1,'y':1},'rejected offer')
        self.assertAlmostEqual(r,i['common_reward']+i['costs']['rejection'])
    def test_all_bank_rows_complete(self):
        bank=json.loads((OUT/'language_bank.json').read_text(encoding='utf-8'))
        self.assertEqual(len(bank),396); self.assertEqual(len({z['id'] for z in bank}),396)
        for x in bank: self.assertIn(x['meaning']['kind'],['offer','query','signal','ambiguity','ratify','withdraw','invalid'])

if __name__=='__main__': unittest.main()
