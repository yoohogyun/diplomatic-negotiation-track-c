"""Prespecified follow-up after all 20,480-step PPO seeds ignored queries.
Three fresh seeds at 102,400 steps, same settings/condition mixture. This is
exploratory training-budget sensitivity, not a selected confirmatory result.
"""
import time,json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from pilot.env import *
from pilot.run import OUT,SEEDS,conditions,evaluate,save

class Curve(BaseCallback):
    def __init__(self): super().__init__(); self.rows=[]; self.ret=0.; self.length=0
    def _on_step(self):
        self.ret+=float(self.locals['rewards'][0]); self.length+=1
        if self.locals['dones'][0]:
            self.rows.append([self.num_timesteps,self.ret,self.length]); self.ret=0.; self.length=0
        return True

def main():
    path=OUT/'extension_protocol.json'
    if path.exists(): raise RuntimeError('Extension already started; do not overwrite')
    save(path,{'time':time.time(),'reason':'All three short PPO seeds ignored information; inspect learning-budget limitation.',
              'seeds':SEEDS,'steps':102400,'fresh_initialization':True,'other_settings':'unchanged','no_additional_tuning':'Stop after these three runs regardless of outcome.'})
    torch.set_num_threads(1); rows=[]; configs=list(dict.fromkeys(c for _,c in conditions()))
    for seed in SEEDS:
        cb=Curve(); m=PPO('MlpPolicy',NegotiationEnv(configs),seed=seed,n_steps=256,batch_size=64,n_epochs=10,
            learning_rate=3e-4,gamma=.95,gae_lambda=.95,ent_coef=.01,policy_kwargs={'net_arch':[64,64]},device='cpu',verbose=0)
        m.learn(total_timesteps=102400,callback=cb); m.save(OUT/f'ppo_long_{seed}'); save(OUT/f'curve_long_{seed}.json',cb.rows)
        for name,c in conditions():
            rows.append(dict(condition=name,policy='PPO_long',seed=seed,**evaluate(c,lambda o,s:int(m.predict(o,deterministic=True)[0]))))
            rows.append(dict(condition=name,policy='PPO_long_belief_mask_only',seed=seed,**evaluate(c,lambda o,s:int(m.predict(features(c,s,True),deterministic=True)[0]))))
        save(OUT/'extension_metrics.json',rows)
        print('Long seed',seed,[(r['condition'],r['policy'],round(r['return_'],5),r['query']) for r in rows if r['seed']==seed and r['condition'] in ('base','grid_r0.32_q0.9','no_agreement_hard')],flush=True)

if __name__=='__main__': main()
