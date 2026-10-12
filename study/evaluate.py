"""Freeze dev heuristic, then evaluate every saved policy with full weighted raw records."""
import json,gzip,time,pickle,sys
from functools import lru_cache
from dataclasses import asdict,replace
import numpy as np
import torch
from stable_baselines3 import PPO
from study import OUT
from study.env import LunaEnv
from study.train import Bandit
from pilot.env import Config,Public,observations,posterior,advance,own_utility,other_utility,accepts
from pilot.run import heuristic
from pilot.live import write

@lru_cache(maxsize=128)
def planner_for(cfg,encoded,allow_query=True):
    meanings=json.loads(encoded)
    @lru_cache(maxsize=None)
    def solve(s):
        values=[]
        for a,m in enumerate(meanings):
            if not allow_query and m['kind']=='query': continue
            v=0.
            for y,ph,ps in observations(cfg,s,m):
                prob=s.b*ph+(1-s.b)*ps
                if prob<=1e-12: continue
                b,_=posterior(s.b,ph,ps); n,r,d,_=advance(cfg,s,m,y,b)
                v+=prob*(r+(0 if d else solve(n)[0]))
            values.append((v,a))
        return max(values,key=lambda x:(round(x[0],12),-x[1]))
    return solve

def encoded_bank(env):
    return json.dumps([env.lookup[(env.context['id'],a,env.variant)]['meaning'] for a in range(11)],sort_keys=True)

def fixed(name,env):
    s=env.s;c=env.cfg
    if name=='probe_then_offer':
        if s.t==0:return 3
        if s.t==1:return 6 if s.b<.5 else (8 if c.hard_cost<=.5 else 10)
        return heuristic('adaptive',c,s)
    return heuristic(name,c,s)

def run_episode(env,actor,config,context,variant,hard,u,seed):
    o,_=env.reset(seed=seed,options={'config':config,'context':context,'variant':variant,'hard':hard,'query_uniform':u})
    traces=[]; ret=0.; reversals=0; invalid=0
    for _ in range(config.horizon):
        action=int(actor(o,env)); o,_,done,_,i=env.step(action); traces.append(i)
        ret+=i['common_reward']; reversals+=i['reversal']; invalid+=i['costs']['invalid']>0
        if done: break
    agree=i['outcome']=='agreement'; own=i['terminal_utility']
    other=other_utility(config,env.s.x,env.s.y,hard,env.s.signaled) if agree else 0.
    feasible=any(own_utility(x,y)>=config.outside and other_utility(config,x,y,hard,sign)>=-1e-9 for x in (1,2,3) for y in (1,2) for sign in (False,True))
    return {'net_utility':ret,'agreement':float(agree),'own_utility':own,'opponent_utility':other,'total_utility':own+other,
            'cost':own-ret,'breakdown':float(not agree),'length':len(traces),'queries':sum(i['meaning']['kind']=='query' for i in traces),
            'signals':sum(i['meaning']['kind']=='signal' for i in traces),'max_escalation':env.s.escalation,
            'mean_escalation':float(np.mean([i['after']['escalation'] for i in traces])),
            'reversals':reversals,'invalid_attempts':int(invalid),'acceptance_violations':int(agree and (other<-1e-9 or own<config.outside-1e-9)),
            'feasible':float(feasible),'outcome':i['outcome'],'trajectory':traces}

def assess(name,actor,split='test',seed=None,env_options=None,random_reps=1,save_raw=True,configs=None):
    env=LunaEnv(split,**(env_options or {})); configs=configs or env.configs
    records=[]; (OUT/'raw').mkdir(exist_ok=True)
    handle=gzip.open(OUT/'raw'/f'{split}_{name}_{seed}.jsonl.gz','wt',encoding='utf-8') if save_raw else None
    for ctx in env.contexts:
        for ci,cfg in enumerate(configs):
            totals={}
            for variant in range(3):
                for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
                    ph=cfg.reliability if hard else 1-cfg.reliability
                    for u,py in [(0.,ph),(.999999,1-ph)]:
                        if pt*py==0:continue
                        for repeat in range(random_reps):
                            w=pt*py/(3*random_reps)
                            z=run_episode(env,actor,cfg,ctx,variant,hard,u,123+repeat)
                            for k,v in z.items():
                                if isinstance(v,(int,float)):totals[k]=totals.get(k,0.)+w*v
                            if handle:
                                handle.write(json.dumps(dict(policy=name,seed=seed,context=ctx['id'],event_group=ctx['event_group'],condition=ci,
                                    config=asdict(cfg),variant=variant,weight=w,hidden_evaluation_only={'hard':hard,'query_uniform':u},**z),ensure_ascii=False)+'\n')
            records.append(dict(policy=name,seed=seed,split=split,context=ctx['id'],event_group=ctx['event_group'],condition=ci,**totals))
    if handle:handle.close()
    if save_raw:write(OUT/'case_metrics'/f'{split}_{name}_{seed}.json',records)
    return records

def mean_grouped(records):
    numeric=[k for k in records[0] if isinstance(records[0][k],(int,float)) and k not in ('seed','condition')]
    groups=sorted({r['event_group'] for r in records})
    return {k:float(np.mean([np.mean([r[k] for r in records if r['event_group']==g]) for g in groups])) for k in numeric}

def main():
    torch.set_num_threads(1)
    if not (OUT/'training_complete.json').exists(): raise RuntimeError('Training matrix incomplete')
    if (OUT/'test_opened.json').exists():raise RuntimeError('Test already evaluated; do not silently rerun selection')
    choices={}
    for name in ('adaptive','query_first','signal_first','robust','withdraw','probe_then_offer'):
        rs=assess(name,lambda o,e,n=name:fixed(n,e),split='dev')
        choices[name]=mean_grouped(rs)['net_utility']
    best=max(choices,key=choices.get);write(OUT/'heuristic_selection.json',{'dev_scores':choices,'selected':best,'time':time.time()})
    write(OUT/'test_opened.json',{'time':time.time(),'selection_frozen':True})
    summaries=[]
    actors=[('Heuristic',lambda o,e:fixed(best,e)),('Planner',lambda o,e:planner_for(e.cfg,encoded_bank(e))(e.s)[1]),
            ('Planner_no_query',lambda o,e:planner_for(e.cfg,encoded_bank(e),False)(e.s)[1])]
    for name,actor in actors:
        rs=assess(name,actor);summaries.append(dict(policy=name,seed=None,**mean_grouped(rs)))
        print('Evaluated',name,summaries[-1]['net_utility'],flush=True)
    for seed in (101,202,303):
        rng=np.random.default_rng(seed);rs=assess('Random',lambda o,e:int(rng.integers(11)),seed=seed,random_reps=4)
        summaries.append(dict(policy='Random',seed=seed,**mean_grouped(rs)))
    setattr(sys.modules['__main__'],'Bandit',Bandit)
    for seed in (17,29,43):
        with (OUT/f'models/bandit_{seed}.pkl').open('rb') as f:b=pickle.load(f)
        rs=assess('Bandit',lambda o,e:b.predict(o),seed=seed);summaries.append(dict(policy='Bandit',seed=seed,**mean_grouped(rs)))
    protocol=json.loads((OUT/'protocol.json').read_text())
    for run in protocol['runs']:
        for seed in run['seeds']:
            name=run['name']; model=PPO.load(OUT/f'models/{name}_{seed}',device='cpu')
            opts={'mask':run.get('mask'),'disable_query':run.get('disable_query',False)}
            rs=assess('PPO_'+name,lambda o,e:int(model.predict(o,deterministic=True)[0]),seed=seed,env_options=opts)
            summaries.append(dict(policy='PPO_'+name,seed=seed,**mean_grouped(rs)))
            if name=='base':
                rs=assess('PPO_eval_belief_mask',lambda o,e:int(model.predict(o,deterministic=True)[0]),seed=seed,env_options={'mask':'belief'})
                summaries.append(dict(policy='PPO_eval_belief_mask',seed=seed,**mean_grouped(rs)))
            write(OUT/'seed_metrics.json',summaries)
            print('Evaluated PPO',name,seed,round(summaries[-1]['net_utility'],5),flush=True)
    write(OUT/'seed_metrics.json',summaries)
    write(OUT/'evaluation_complete.json',{'time':time.time(),'all_prespecified_runs_included':True})

if __name__=='__main__':main()
