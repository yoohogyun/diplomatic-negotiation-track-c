"""Paired evaluation-only rule shifts for three frozen PPO seeds and public planners."""
import json,time
from dataclasses import asdict,replace
import torch
from stable_baselines3 import PPO
from study import OUT
from study.evaluate import assess,mean_grouped,planner_for,encoded_bank,fixed
from pilot.env import Config
from pilot.live import write

def main():
    torch.set_num_threads(1)
    families={'reliability':[.5,.7,.9,1.],'signal_effect':[0.,.1,.25],'hard_cost':[.4,.55,.9],
              'rejection_cost':[0.,.04,.08,.16,.32],'query_cost':[.01,.02,.08]}
    specs=[(field,value,replace(Config(),**{field:value})) for field,values in families.items() for value in values]
    write(OUT/'sensitivity_protocol.json',{'type':'paired evaluation-only changes; no retraining','reference':asdict(Config()),
        'changes':[(k,v) for k,v,_ in specs],'model_seeds':[17,29,43],'contexts':'same four test contexts','all_other_parameters_fixed':True})
    best=json.loads((OUT/'heuristic_selection.json').read_text())['selected']; result=[]
    actors=[('Planner',None,lambda o,e:planner_for(e.cfg,encoded_bank(e))(e.s)[1]),
            ('Planner_no_query',None,lambda o,e:planner_for(e.cfg,encoded_bank(e),False)(e.s)[1]),
            ('Heuristic',None,lambda o,e:fixed(best,e))]
    for seed in (17,29,43):
        m=PPO.load(OUT/f'models/base_{seed}',device='cpu'); actors.append(('PPO',seed,lambda o,e,m=m:int(m.predict(o,deterministic=True)[0])))
    for name,seed,actor in actors:
        rows=assess('sensitivity_'+name,actor,seed=seed,configs=[c for _,_,c in specs])
        for i,(field,value,cfg) in enumerate(specs):
            subset=[r for r in rows if r['condition']==i]
            result.append(dict(policy=name,seed=seed,field=field,value=value,**mean_grouped(subset)))
        print('Sensitivity evaluated',name,seed,flush=True)
    write(OUT/'sensitivity_metrics.json',result)

if __name__=='__main__':main()
