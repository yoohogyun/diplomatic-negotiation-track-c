"""Train-only value gate, real Sol dev selection, frozen numerical pilot."""
import argparse, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3 import PPO
from routing import ROOT, OUT
from pilot.env import Config, NAMES
from pilot.live import write
from pilot.language import schema
from study.env import LunaEnv
from study.evaluate import run_episode, planner_for, encoded_bank
from study.luna_policy import RULES
from solstudy.api import call


def net():
    return torch.nn.Sequential(torch.nn.Linear(31,128),torch.nn.Tanh(),
        torch.nn.Linear(128,128),torch.nn.Tanh(),torch.nn.Linear(128,11))


def prepare():
    if (OUT/'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    numeric=json.loads((ROOT/'study_results/numeric_conditions.json').read_text())
    seen={json.dumps(c,sort_keys=True) for rows in numeric.values() for c in rows}
    seen.update(json.dumps(c,sort_keys=True) for c in json.loads((ROOT/'improvement_results/fresh_numeric_test.json').read_text()))
    rng=np.random.default_rng(2026101207);fresh=[]
    while len(fresh)<16:
        c=asdict(Config(prior=float(rng.choice([.2,.35,.5,.65,.8])),reliability=float(rng.choice([.6,.75,.9,1.])),
            query_cost=float(rng.choice([.01,.02,.04,.08])),rejection_cost=float(rng.choice([0.,.04,.08,.16,.32])),
            signal_effect=float(rng.choice([0.,.1,.25])),hard_cost=float(rng.choice([.4,.45,.55,.9]))))
        k=json.dumps(c,sort_keys=True)
        if k not in seen: fresh.append(c);seen.add(k)
    contexts=json.loads((ROOT/'study_results/contexts.json').read_text(encoding='utf-8'))
    test=[]
    for c in contexts:
        if c['split']=='test' and c['event_group'] not in [x['event_group'] for x in test]:test.append(c)
    assert len(test)==2
    write(OUT/'fresh_numeric_test.json',fresh)
    write(OUT/'protocol.json',{'frozen_at':time.time(),
        'question':'Can a public-state learned regret gate retain Sol medium utility with fewer reasoning calls?',
        'cheap_policy':'Existing Sol distilled_ppo seed17, fixed before new dev/test; no seed selection.',
        'gate':'Three train-only MLP Q estimators; max(mean Q)-mean Q[cheap action]. Call Sol if this exceeds threshold.',
        'gate_training':'Existing Sol teacher data (40706 rows); seeds71,73,79; 3000 Adam updates, batch512, lr.001, Q MSE. No new planner labels.',
        'threshold_candidates':[-1.,.01,.03,.06,.1,.2],
        'dev':'First 8 original dev configurations, both dev documents, style0; actual Sol medium fallback.',
        'selection':'Minimum expected Sol calls among candidates with dev utility >= always-Sol minus .01; ties by greater utility. Always-Sol threshold -1 guarantees feasible fallback.',
        'test_contexts':test,'test':'16 new numerical configs x 2 previously exposed source documents, style0. Full nonzero hidden/query branches probability-weighted. Not new historical holdout.',
        'comparators':['cheap only','always Sol medium','dev-selected hybrid','public planner'],
        'prompt':'Exact previous direct-policy RULES, model and reasoning budget unchanged; one cached model sample per identical public prompt shared between methods.',
        'resource_accounting':'Expected calls/tokens/API cost per episode include every logical invocation even cache hits; actual billed unique requests reported separately. Offline teacher/gate costs separate.',
        'limits':'Exploratory small pilot; no novelty, noninferiority, human-accuracy or real diplomatic success claim; no test-driven tuning.',
        'budget':'Same shared ledger; internal cumulative $9, user ceiling <$10; 4 concurrent case workers.'})
    print('Protocol frozen; test source sentences:',flush=True)
    for c in test:print(c['id'],c['source']['sentence'],flush=True)


def train():
    torch.set_num_threads(1)
    if (OUT/'gate_training.json').exists():raise RuntimeError('Gate training already completed')
    d=np.load(ROOT/'sol_improvement_results/teacher_data.npz');x=torch.tensor(d['obs']);y=torch.tensor(d['q'])
    curves={};started=time.time()
    for seed in (71,73,79):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);m=net();opt=torch.optim.Adam(m.parameters(),lr=.001);curve=[]
        for step in range(3000):
            ix=rng.integers(len(x),size=512);loss=((m(x[ix])-y[ix])**2).mean()
            opt.zero_grad();loss.backward();opt.step()
            if step%300==0:curve.append({'step':step,'mse':float(loss.detach())})
        torch.save(m.state_dict(),OUT/f'gate_{seed}.pt');curves[str(seed)]=curve
        print('Gate trained',seed,'last minibatch MSE',float(loss.detach()),flush=True)
    write(OUT/'gate_training.json',{'rows':len(x),'seconds':time.time()-started,'curves':curves,
        'input_data_sha256':hashlib.sha256((ROOT/'sol_improvement_results/teacher_data.npz').read_bytes()).hexdigest()})


def load_models():
    torch.set_num_threads(1)
    cheap=PPO.load(ROOT/'sol_improvement_results/models/distilled_ppo_17.zip',device='cpu')
    gates=[]
    for seed in (71,73,79):
        m=net();m.load_state_dict(torch.load(OUT/f'gate_{seed}.pt',weights_only=True));m.eval();gates.append(m)
    return cheap,gates


def worker(job,cheap,gates,thresholds):
    split,ctx,ci,cfg=job
    path=OUT/'cases'/f'{split}_{ctx["id"]}_{ci}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    bank=json.loads((ROOT/'sol_results/language_bank.json').read_text(encoding='utf-8'))
    env=LunaEnv(split,bank=bank);cache={};rows=[]
    methods=[('always_sol',-1.),('cheap',float('inf'))]+[(f'gate_{v}',v) for v in thresholds if v!=-1.]
    if split=='test':methods.extend([('initial_sol','initial'),('planner',None)])
    for name,threshold in methods:
        for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
            ph=cfg.reliability if hard else 1-cfg.reliability
            for u,py in [(0.,ph),(.999999,1-ph)]:
                if pt*py<=0:continue
                usage={'sol_calls':0.,'prompt_tokens':0.,'completion_tokens':0.,'reasoning_tokens':0.,'api_usd':0.};decisions=[]
                def actor(o,e):
                    if threshold is None:return planner_for(e.cfg,encoded_bank(e))(e.s)[1]
                    a=int(cheap.predict(o,deterministic=True)[0])
                    with torch.no_grad():q=torch.stack([m(torch.as_tensor(o)) for m in gates]).mean(0)
                    regret=float(q.max()-q[a]);route=(e.s.t==0) if threshold=='initial' else regret>threshold
                    decision={'public_state':asdict(e.s),'cheap_action':a,'predicted_regret':regret,'route_to_sol':route}
                    if route:
                        public={'state':asdict(e.s),'config':asdict(cfg),'last_interpreted_function':e.last_kind}
                        payload={'source_context':ctx['source']['sentence'],'our_measure':ctx['our_measure'],'their_measure':ctx['their_measure'],
                            'public':public,'actions':dict(enumerate(NAMES))}
                        messages=[{'role':'system','content':RULES},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
                        digest=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()[:24]
                        tag='routing_'+digest
                        recovery_path=OUT/'transport_recovery.json'
                        if recovery_path.exists():
                            recovery=json.loads(recovery_path.read_text())
                            tag=recovery.get('replacements',{}).get(tag,tag)
                        if digest not in cache:
                            result=call(tag,messages,schema({'action':{'type':'number','minimum':0,'maximum':10}}),effort='medium')
                            assert result['action']==int(result['action'])
                            raw=json.loads((ROOT/'sol_results/api'/f'sol_medium_{tag}.raw.json').read_text(encoding='utf-8'))
                            cache[digest]=(int(result['action']),raw['usage'])
                        a,tokens=cache[digest];usage['sol_calls']+=1
                        for k in ('prompt_tokens','completion_tokens'):usage[k]+=tokens.get(k,0)
                        usage['reasoning_tokens']+=tokens.get('completion_tokens_details',{}).get('reasoning_tokens',0)
                        usage['api_usd']+=tokens['cost'];decision['api_tag']='sol_medium_'+tag
                    decision['selected_action']=a;decisions.append(decision);return a
                z=run_episode(env,actor,cfg,ctx,0,hard,u,1)
                rows.append(dict(policy=name,split=split,context=ctx['id'],event_group=ctx['event_group'],condition=ci,config=asdict(cfg),variant=0,
                    weight=pt*py,hidden_evaluation_only={'hard':hard,'query_uniform':u},decisions=decisions,**usage,**z))
    write(path,rows);return rows


def summarize(rows):
    cases=[]
    for name,ctx,ci in sorted({(r['policy'],r['context'],r['condition']) for r in rows}):
        rs=[r for r in rows if (r['policy'],r['context'],r['condition'])==(name,ctx,ci)]
        assert abs(sum(r['weight'] for r in rs)-1)<1e-9
        numeric=[k for k,v in rs[0].items() if isinstance(v,(int,float)) and k not in ('condition','variant','weight')]
        cases.append(dict(policy=name,context=ctx,event_group=rs[0]['event_group'],condition=ci,
            **{k:sum(r['weight']*r[k] for r in rs) for k in numeric}))
    summary={}
    for name in sorted({r['policy'] for r in cases}):
        rs=[r for r in cases if r['policy']==name]
        summary[name]={k:float(np.mean([r[k] for r in rs])) for k in rs[0] if isinstance(rs[0][k],(int,float)) and k!='condition'}
    return cases,summary


def evaluate(split):
    if (OUT/'early_stop.json').exists():raise RuntimeError('Routing paid experiment stopped on user time constraint; inspect early_stop.json before any explicit resumption.')
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    contexts=json.loads((ROOT/'study_results/contexts.json').read_text(encoding='utf-8'))
    if split=='dev':
        contexts=[c for c in contexts if c['split']=='dev'];cfgs=json.loads((ROOT/'study_results/numeric_conditions.json').read_text())['dev'][:8]
        thresholds=protocol['threshold_candidates']
        if (OUT/'selection.json').exists():raise RuntimeError('Dev selection already frozen')
    else:
        contexts=protocol['test_contexts'];cfgs=json.loads((OUT/'fresh_numeric_test.json').read_text())
        thresholds=protocol['threshold_candidates']
        if (OUT/'test_summary.json').exists():raise RuntimeError('Test completed; no reselection')
        write(OUT/'test_started.json',{'time':time.time(),'selection_sha256':hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()})
    cheap,gates=load_models();jobs=[(split,c,i,Config(**cfg)) for c in contexts for i,cfg in enumerate(cfgs)];rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,j,cheap,gates,thresholds) for j in jobs]
        for i,f in enumerate(as_completed(futures)):
            rows.extend(f.result());print(split,'cases complete',i+1,'/',len(jobs),flush=True)
    cases,summary=summarize(rows);write(OUT/f'{split}_case_metrics.json',cases);write(OUT/f'{split}_summary.json',summary)
    if split=='dev':
        candidates=[('always_sol',-1.)]+[(f'gate_{v}',v) for v in thresholds if v!=-1.]
        eligible=[(n,t) for n,t in candidates if summary[n]['net_utility']>=summary['always_sol']['net_utility']-.01]
        name,t=min(eligible,key=lambda nt:(summary[nt[0]]['sol_calls'],-summary[nt[0]]['net_utility']))
        write(OUT/'selection.json',{'time':time.time(),'threshold':t,'policy':name,'summary':summary,'eligible':[n for n,t in eligible]})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','dev','test']);args=p.parse_args()
    if args.stage in ('dev','test'):evaluate(args.stage)
    else:globals()[args.stage]()
