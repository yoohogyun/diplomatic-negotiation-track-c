"""Prespecified supplemental direct-Luna strategy baseline on a fixed test subset.
This isolates the prompted model as selector from PPO controlling Luna utterances.
"""
import json,time,hashlib
from dataclasses import asdict
import numpy as np
from study import OUT
from study.api import call,batch
from study.env import LunaEnv
from study.evaluate import run_episode
from pilot.language import schema
from pilot.live import write
from pilot.env import Config,NAMES

RULES='''You choose a strategy action in a SYNTHETIC diplomatic negotiation. Maximize expected undiscounted OWN net utility; agreement alone is not the objective. Only public state is available. Opponent is soft with type_cost .2 or hard with type_cost given in config. Belief b is probability of hard type. Offer x weeks of our specified measure conditional on y weeks of their measure. Own agreement utility=.5*y-.18*x. Opponent utility=.35*x-type_cost*y+signal_effect*signaled. They accept iff their utility>=0. Ratification requires observed acceptance and own utility>=outside. You can revise accepted proposals before ratification. Terminal utility is paid once. Withdrawal and exhaustion give outside. Every action costs turn_cost; a question adds query_cost, a signal adds signal_cost, a rejection adds rejection_cost and increases escalation by one. Each turn also costs escalation_weight*escalation^2. Reducing a public promised x costs reversal_cost. Invalid ratification costs .1. Signal makes its benefit active once. Query returns a noisy indication with reliability from config and updates b; repeated queries add cost but no information. Offers also update b through deterministic accept/reject. Ambiguity makes the next proposal provisional without removing old promises. Maximum horizon is five actions. All these numerical assumptions are synthetic. Choose one action index; do not supply hidden-state guesses as facts.'''

def worker(job):
    ctx,ci,cfg=job; env=LunaEnv('test'); cache={}; records=[]
    def actor(obs,e):
        public={'state':asdict(e.s),'config':asdict(cfg),'last_interpreted_function':e.last_kind}
        digest=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()[:18]
        if digest not in cache:
            z=call(f'policy_{ctx["id"]}_{ci}_{digest}',[{'role':'system','content':RULES},
              {'role':'user','content':json.dumps({'source_context':ctx['source']['sentence'],'our_measure':ctx['our_measure'],'their_measure':ctx['their_measure'],
                                                 'public':public,'actions':dict(enumerate(NAMES))},ensure_ascii=False)}],
              schema({'action':{'type':'number','minimum':0,'maximum':10}}))
            if z['action']!=int(z['action']): raise ValueError('Nonintegral action; no silent rounding')
            cache[digest]=int(z['action'])
        return cache[digest]
    for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
        ph=cfg.reliability if hard else 1-cfg.reliability
        for u,py in [(0.,ph),(.999999,1-ph)]:
            if pt*py<=0:continue
            z=run_episode(env,actor,cfg,ctx,0,hard,u,1)
            records.append(dict(policy='Direct_Luna',context=ctx['id'],event_group=ctx['event_group'],condition=ci,config=asdict(cfg),variant=0,
                        weight=pt*py,hidden_evaluation_only={'hard':hard,'query_uniform':u},**z))
    write(OUT/'luna_policy_cases'/f'{ctx["id"]}_{ci}.json',records)
    return {'context':ctx['id'],'condition':ci,'unique_calls':len(cache)}

def main():
    path=OUT/'luna_policy_protocol.json'
    if path.exists(): raise RuntimeError('Supplemental direct-Luna protocol already started')
    write(path,{'frozen_at':time.time(),'purpose':'Separate direct Luna strategy choice from PPO controlling a fixed Luna realization module.',
          'test_numeric_indices':list(range(16)),'contexts':'all four held-out document contexts','variant':0,'temperature':'model default',
          'generation_seed':'ledger request seed, saved per call','replication':'one cached response per unique public state; no model-seed uncertainty claim',
          'fair_comparison':'Compare other policies on exactly the same 64 synthetic cases/style; keep separate from full 256-case main result.',
          'not_a_training_change':True})
    deadline=time.time()+1800
    while not (OUT/'language_bank.json').exists():
        if time.time()>deadline:raise RuntimeError('Bank not ready')
        time.sleep(5)
    contexts=[c for c in json.loads((OUT/'contexts.json').read_text(encoding='utf-8')) if c['split']=='test']
    configs=[Config(**c) for c in json.loads((OUT/'numeric_conditions.json').read_text())['test'][:16]]
    results=batch([(c,i,cfg) for c in contexts for i,cfg in enumerate(configs)],worker,4)
    write(OUT/'luna_policy_complete.json',{'cases':len(results),'unique_calls':sum(x['unique_calls'] for x in results),'finished':time.time()})
    print('DIRECT LUNA DONE; outcomes retained for joint final analysis',flush=True)

if __name__=='__main__':main()
