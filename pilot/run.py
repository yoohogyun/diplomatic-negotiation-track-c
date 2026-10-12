"""Frozen exploratory pilot: exact information sweep + three-seed training.

Run: python -X utf8 -m pilot.run
No final-test claim: all configurations are development/exploration.
"""
import json, csv, time, pickle, hashlib
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from pilot import ROOT
from pilot.env import *

OUT=ROOT/'pilot_results'
SEEDS=(17,29,43)
STEPS=20480

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def conditions():
    cs=[(f'grid_r{r}_q{q}',Config(rejection_cost=r,reliability=q)) for r in (0,.04,.08,.16,.32) for q in (.5,.7,.9,1.)]
    cs += [('base',Config()),('cheap_query',Config(query_cost=.005)),('expensive_query',Config(query_cost=.08)),
           ('no_signal_effect',Config(signal_effect=0)),('strong_signal',Config(signal_effect=.25)),
           ('no_agreement_hard',Config(hard_cost=.9)),('low_prior',Config(prior=.2)),('high_prior',Config(prior=.8))]
    return cs

def heuristic(name,cfg,s):
    if s.accepted and own_utility(s.x,s.y)>=cfg.outside: return 9
    if s.t>=4: return 10
    if name=='withdraw': return 10
    if name=='signal_first' and not s.signaled: return 1
    if name=='query_first' and not s.queried: return 2
    if name=='robust': return 8
    # Strong adaptive heuristic: posterior expected terminal reward minus immediate
    # offer risks, with a sensible known-infeasible withdrawal rule.
    best=(cfg.outside-cfg.turn_cost,10)
    for a in range(3,9):
        m=parse(render(a)); u=own_utility(m['x'],m['y'])
        if u<cfg.outside: continue
        ph=accepts(cfg,s,m['x'],m['y'],True); ps=accepts(cfg,s,m['x'],m['y'],False)
        p=s.b*ph+(1-s.b)*ps
        v=p*(u-2*cfg.turn_cost)+(1-p)*(cfg.outside-2*cfg.turn_cost-cfg.rejection_cost-cfg.escalation_weight*(s.escalation+1)**2)
        if v>best[0]: best=(v,a)
    return best[1]

def episode(cfg,act,hard,u,seed=0):
    e=NegotiationEnv([cfg]); obs,_=e.reset(seed=seed,options={'hard':hard,'query_uniform':u})
    logs=[]; ret=0.; rev=0; invalid=0
    for _ in range(cfg.horizon):
        a=int(act(obs,e.s)); obs,r,done,_,info=e.step(a); info['action']=a; info['reward']=r
        logs.append(info); ret+=r; rev+=info['reversal']; invalid+=int(info['costs']['invalid']>0)
        if done: break
    agreement=info['outcome']=='agreement'
    own=info['terminal_utility']; other=other_utility(cfg,e.s.x,e.s.y,hard,e.s.signaled) if agreement else 0.
    return dict(return_=ret,agreement=float(agreement),own_utility=own,opponent_utility=other,
                total_utility=own+other,cost=own-ret,turns=len(logs),query=sum(z['action']==2 for z in logs),
                signal=sum(z['action']==1 for z in logs),reversals=rev,invalid_attempts=invalid,
                acceptance_violations=int(agreement and (other<-1e-9 or own<cfg.outside-1e-9)),
                max_escalation=e.s.escalation,mean_escalation=np.mean([z['after']['escalation'] for z in logs]),
                breakdown=float(not agreement),outcome=info['outcome'],trajectory=logs)

def evaluate(cfg,act,random_reps=1):
    records=[]
    for hard,p_type in [(False,1-cfg.prior),(True,cfg.prior)]:
        p_response=cfg.reliability if hard else 1-cfg.reliability
        for u,p in [(0.,p_response),(.999999,1-p_response)]:
            if p_type*p==0: continue
            for rep in range(random_reps):
                records.append((p_type*p/random_reps,episode(cfg,act,hard,u,rep)))
    keys=[k for k,v in records[0][1].items() if isinstance(v,(int,float,np.number))]
    return {k:float(sum(w*r[k] for w,r in records)) for k in keys}

class LinearBandit:
    def __init__(self,seed=0):
        self.rng=np.random.default_rng(seed); self.inv=np.array([np.eye(18)*10 for _ in range(11)]); self.weights=np.zeros((11,18))
    def predict(self,obs):
        x=np.r_[1.,obs]; return int(np.argmax(self.weights@x))
    def train(self,configs,steps):
        env=NegotiationEnv(configs); obs,_=env.reset(seed=int(self.rng.integers(100000)))
        for t in range(steps):
            a=int(self.rng.integers(11)) if self.rng.random()<.2 else self.predict(obs)
            x=np.r_[1.,obs]; new,r,done,_,_=env.step(a)
            px=self.inv[a]@x; denom=1+x@px
            self.weights[a]+=px/denom*(r-self.weights[a]@x)
            self.inv[a]-=np.outer(px,px)/denom
            obs=env.reset()[0] if done else new

def main():
    OUT.mkdir(exist_ok=True)
    spec={'status':'exploratory; no untouched test set','seeds':SEEDS,'steps_per_base_run':STEPS,
          'conditions':[(n,asdict(c)) for n,c in conditions()],
          'hypothesis':'Queries may have zero value when offers cheaply reveal type; value depends on rejection cost, reliability and signal effect.',
          'freeze_time':time.time(),'source_context':'C-012 reciprocal suspension; additional source functions audited separately',
          'evaluation':'Exact weighted enumeration of hidden type and one informative query response. Public offer responses deterministic.'}
    if (OUT/'protocol.json').exists(): raise RuntimeError('Do not overwrite a completed or interrupted pilot; inspect first.')
    save(OUT/'protocol.json',spec)
    rows=[]; sweep=[]; trajectory_cases=[]
    for name,c in conditions():
        p=Planner(c); nq=Planner(c,False); s=Public(b=c.prior)
        v,a=p.solve(s); nv,na=nq.solve(s)
        sweep.append(dict(condition=name,query_value=v-nv,planner_return=v,noquery_return=nv,first_action=NAMES[a],noquery_first=NAMES[na],**asdict(c)))
        for policy,act in [('Planner',lambda o,s:p.action(s)),('Planner_no_query',lambda o,s:nq.action(s))]+[(n,lambda o,s,n=n:heuristic(n,c,s)) for n in ('adaptive','query_first','signal_first','robust','withdraw')]:
            rows.append(dict(condition=name,policy=policy,seed=None,**evaluate(c,act)))
        rng=np.random.default_rng(12345)
        rows.append(dict(condition=name,policy='Random',seed=None,**evaluate(c,lambda o,s:int(rng.integers(11)),128)))
        if name in ('base','grid_r0_q0.9','grid_r0.32_q0.9','no_agreement_hard','strong_signal','grid_r0.08_q0.5'):
            for hard in (False,True):
                tr=episode(c,lambda o,s:p.action(s),hard,0.)
                trajectory_cases.append(dict(condition=name,hard_evaluation_only=hard,query_uniform=0.,config=asdict(c),policy='Planner',**tr))
    save(OUT/'information_sweep.json',sweep); save(OUT/'planner_cases.json',trajectory_cases)
    print('Exact sweep complete',[(s['condition'],round(s['query_value'],5),s['first_action']) for s in sweep if s['condition'] in ('base','grid_r0_q0.9','grid_r0.32_q0.9','strong_signal')],flush=True)
    # CPU + single thread makes the short pilot reproducible and avoids oversubscription.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    torch.set_num_threads(1)
    class Curve(BaseCallback):
        def __init__(self): super().__init__(); self.rows=[]; self.running=0.; self.length=0
        def _on_step(self):
            self.running+=float(self.locals['rewards'][0]); self.length+=1
            if self.locals['dones'][0]:
                self.rows.append([self.num_timesteps,self.running,self.length]); self.running=0.; self.length=0
            return True
    configs=list(dict.fromkeys(c for _,c in conditions()))
    for seed in SEEDS:
        b=LinearBandit(seed); b.train(configs,STEPS)
        with (OUT/f'bandit_{seed}.pkl').open('wb') as f: pickle.dump(b,f)
        for name,c in conditions(): rows.append(dict(condition=name,policy='Bandit',seed=seed,**evaluate(c,lambda o,s:b.predict(o))))
        env=NegotiationEnv(configs); cb=Curve()
        model=PPO('MlpPolicy',env,seed=seed,n_steps=256,batch_size=64,n_epochs=10,learning_rate=3e-4,
                  gamma=.95,gae_lambda=.95,ent_coef=.01,policy_kwargs={'net_arch':[64,64]},device='cpu',verbose=0)
        model.learn(total_timesteps=STEPS,callback=cb); model.save(OUT/f'ppo_{seed}')
        save(OUT/f'curve_{seed}.json',cb.rows)
        for name,c in conditions():
            rows.append(dict(condition=name,policy='PPO',seed=seed,**evaluate(c,lambda o,s:int(model.predict(o,deterministic=True)[0]))))
            # Evaluation-only belief masking: response/acceptance history still exists.
            rows.append(dict(condition=name,policy='PPO_belief_mask_only',seed=seed,**evaluate(c,lambda o,s:int(model.predict(features(c,s,True),deterministic=True)[0]))))
        save(OUT/'metrics.json',rows)
        print('Completed seed',seed,[(r['policy'],round(r['return_'],4),r['agreement']) for r in rows if r['condition']=='base' and r['seed']==seed],flush=True)
    save(OUT/'metrics.json',rows)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(OUT/'runtime.json',{'numpy':np.__version__,'torch':torch.__version__,'gymnasium':gym.__version__,
                           'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'pilot').glob('*.py')}})
    print('Pilot complete',flush=True)

if __name__=='__main__': main()
