"""Re-evaluate saved policies, preserve raw paths, and build pilot figures/report."""
import json,csv,hashlib,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from pilot import ROOT
from pilot.env import *
from pilot.run import OUT,SEEDS,conditions,evaluate,episode,heuristic,save,LinearBandit

def probe_heuristic(c,s):
    # A genuinely multi-step fixed rule: cheap proposal diagnoses type, then
    # exploit the public acceptance response. Do not ratify the diagnostic offer.
    if s.t==0: return 3
    if s.t==1:
        return 6 if s.b<.5 else (8 if c.hard_cost<=.5 else 10)
    return heuristic('adaptive',c,s)

def audit_language_transitions():
    from pilot.language import to_meaning
    rows=json.loads((OUT/'language_roundtrip.json').read_text(encoding='utf-8'))
    out=[]
    for z in rows:
        for hard in (False,True):
            e=NegotiationEnv(); e.reset(seed=1,options={'hard':hard}); e.step(8)
            actual=z['meaning']; expected=z['expected']; states=[]
            for m in (expected,actual):
                rs=observations(e.cfg,e.s,m)
                if m['kind']=='offer': response=rs[0 if accepts(e.cfg,e.s,m['x'],m['y'],hard) else 1]
                elif m['kind']=='query': response=rs[0 if hard else 1]
                else: response=rs[0]
                y,ph,ps=response; b,_=posterior(e.s.b,ph,ps)
                n,r,d,i=advance(e.cfg,e.s,m,y,b); states.append({'state':asdict(n),'reward':r,'done':d,'info':i})
            out.append({'id':z['id'],'hard_evaluation_only':hard,'match':states[0]==states[1],
                        'intended_transition':states[0],'interpreted_transition':states[1]})
    save(OUT/'language_transition_replay.json',out)
    return rows,out

def main():
    torch.set_num_threads(1)
    metrics=json.loads((OUT/'metrics.json').read_text())+json.loads((OUT/'extension_metrics.json').read_text())
    models={f'PPO_{s}':PPO.load(OUT/f'ppo_{s}',device='cpu') for s in SEEDS}
    models.update({f'PPO_long_{s}':PPO.load(OUT/f'ppo_long_{s}',device='cpu') for s in SEEDS})
    bandits={}
    # Pickle contains a __main__ class when trained via -m; load only our own file.
    import pickle,sys
    setattr(sys.modules['__main__'],'LinearBandit',LinearBandit)
    for seed in SEEDS:
        with (OUT/f'bandit_{seed}.pkl').open('rb') as f: bandits[seed]=pickle.load(f)
    transitions={}; weighted={}; validation=[]; cases=[]
    logpath=OUT/'trajectories.jsonl'
    with logpath.open('w',encoding='utf-8') as f:
        for name,c in conditions():
            p=Planner(c); nq=Planner(c,False)
            actors={'Planner':lambda o,s:p.action(s),'Planner_no_query':lambda o,s:nq.action(s),'probe_then_offer':lambda o,s:probe_heuristic(c,s)}
            actors.update({n:lambda o,s,n=n:heuristic(n,c,s) for n in ('adaptive','query_first','signal_first','robust','withdraw')})
            actors.update({n:lambda o,s,m=m:int(m.predict(o,deterministic=True)[0]) for n,m in models.items()})
            actors.update({f'Bandit_{seed}':lambda o,s,b=b:b.predict(o) for seed,b in bandits.items()})
            metrics.append(dict(condition=name,policy='probe_then_offer',seed=None,**evaluate(c,actors['probe_then_offer'])))
            exact=evaluate(c,actors['Planner'])['return_']; dp=p.solve(Public(b=c.prior))[0]
            validation.append({'condition':name,'source_id':'C-012','parameterized_case_not_independent_event':True,
                               'planner_matches_environment':abs(exact-dp)<1e-9,'difference':exact-dp})
            for policy,act in actors.items():
                if policy not in transitions: transitions[policy]=np.zeros((11,11),int); weighted[policy]=np.zeros((11,11),float)
                for hard,p_type in [(False,1-c.prior),(True,c.prior)]:
                    pr=c.reliability if hard else 1-c.reliability
                    for u,pq in [(0.,pr),(.999999,1-pr)]:
                        w=p_type*pq
                        if w==0: continue
                        z=episode(c,act,hard,u)
                        rec=dict(condition=name,policy=policy,weight=w,config=asdict(c),source_id='C-012',
                                 hidden_evaluation_only={'hard':hard,'query_uniform':u},**z)
                        f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                        for a,b in zip(z['trajectory'],z['trajectory'][1:]):
                            transitions[policy][a['action'],b['action']]+=1
                            weighted[policy][a['action'],b['action']]+=w
                        if policy in ('Planner','PPO_long_17','probe_then_offer') and name in ('base','grid_r0_q0.9','grid_r0.32_q0.9','no_agreement_hard') and u==0:
                            cases.append(rec)
    save(OUT/'environment_case_validation.json',validation); assert all(r['planner_matches_environment'] for r in validation)
    save(OUT/'discourse_cases.json',cases)
    transition_json={}
    for name,m in transitions.items():
        den=m.sum(axis=1,keepdims=True); probs=np.divide(m,den,out=np.zeros_like(m,dtype=float),where=den!=0)
        transition_json[name]={'counts':m.tolist(),'probabilities':[p.tolist() if n else None for p,n in zip(probs,den[:,0])],
                               'unobserved_rows':np.where(den[:,0]==0)[0].tolist(),'weighted_counts':weighted[name].tolist()}
    save(OUT/'transition_matrices.json',transition_json)
    save(OUT/'all_metrics.json',metrics)
    with (OUT/'all_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    sweep=json.loads((OUT/'information_sweep.json').read_text())
    rs=(0,.04,.08,.16,.32); qs=(.5,.7,.9,1.)
    grid=np.array([[next(z['query_value'] for z in sweep if z['condition']==f'grid_r{r}_q{q}') for q in qs] for r in rs])
    fig,ax=plt.subplots(figsize=(7.4,4.2)); im=ax.imshow(grid,cmap='Blues',vmin=0,vmax=.12,aspect='auto')
    ax.set(xticks=range(4),xticklabels=qs,yticks=range(5),yticklabels=rs,xlabel='Private-query reliability',ylabel='Cost per rejected public proposal',title='Value of allowing private questions\nExact public-belief planning; synthetic rules')
    for i in range(5):
        for j in range(4): ax.text(j,i,f'{grid[i,j]:.4f}',ha='center',va='center',color='white' if grid[i,j]>.07 else 'black')
    fig.colorbar(im,ax=ax,label='Expected net utility gain'); fig.tight_layout(); fig.savefig(OUT/'information_value.png',dpi=180); plt.close(fig)
    summaries=[]
    for condition in ('grid_r0_q0.9','base','grid_r0.32_q0.9','no_agreement_hard'):
        for policy in ('Planner','Planner_no_query','probe_then_offer','query_first','robust','Random','Bandit','PPO','PPO_long','PPO_long_belief_mask_only'):
            xs=[r for r in metrics if r['condition']==condition and r['policy']==policy]
            summaries.append(dict(condition=condition,policy=policy,n=len(xs),mean_return=float(np.mean([x['return_'] for x in xs])),
                seed_sd=float(np.std([x['return_'] for x in xs],ddof=1)) if len(xs)>1 else None,
                agreement=float(np.mean([x['agreement'] for x in xs])),query=float(np.mean([x['query'] for x in xs]))))
    save(OUT/'summary.json',summaries)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    policies=('Planner','Planner_no_query','probe_then_offer','PPO','PPO_long','Bandit')
    for ax,condition,title in zip(axes,('base','grid_r0.32_q0.9'),('Base: rejection cost 0.08','Higher rejection cost: 0.32')):
        vals=[next(x for x in summaries if x['condition']==condition and x['policy']==p) for p in policies]
        ax.bar(range(len(vals)),[x['mean_return'] for x in vals],yerr=[x['seed_sd'] or 0 for x in vals],capsize=3,color=['#3c7a89','#74a2b0','#9aa7b2','#c9985d','#a65942','#888888'])
        ax.set(xticks=range(len(vals)),xticklabels=['Planner','No query','Probe rule','PPO 20k','PPO 102k','Bandit'],ylim=(0,.65),ylabel='Expected net utility',title=title)
        ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Exploratory pilot; error bars = three training seeds, not historical uncertainty'); fig.tight_layout(); fig.savefig(OUT/'policy_comparison.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name in zip(axes,('Planner','PPO_long_17')):
        m=transitions[name]; d=m.sum(axis=1,keepdims=True)
        p=np.divide(m,d,out=np.full_like(m,np.nan,dtype=float),where=d!=0)
        im=ax.imshow(p,cmap='Blues',vmin=0,vmax=1); ax.set(xticks=range(11),yticks=range(11),xticklabels=NAMES,yticklabels=NAMES,title=name)
        ax.tick_params(axis='x',rotation=90,labelsize=8); ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Adjacent actions within episodes; white rows are unobserved\nUnweighted enumerated paths across exploratory settings')
    fig.tight_layout(); fig.savefig(OUT/'action_transitions.png',dpi=180); plt.close(fig)
    language,replays=audit_language_transitions()
    probes=json.loads((OUT/'semantic_probes.json').read_text()); audits=json.loads((OUT/'source_audit.json').read_text())
    ledger=json.loads((ROOT/'budget/pilot_20261012.json').read_text())
    info={'roundtrip_matches':sum(x['match'] for x in language),'roundtrip_total':len(language),'transition_matches':sum(x['match'] for x in replays),
          'transition_total':len(replays),'counterfactual_matches':sum(x['match'] for x in probes),'counterfactual_total':len(probes),
          'source_evidence_exact':sum(x['evidence_is_exact_substring'] for x in audits),'source_evidence_total':len(audits),
          'billed_usd':sum(x.get('cost',0) for x in ledger['records']),'unresolved_usd':sum(x['reserved'] for x in ledger['records'] if 'cost' not in x),
          'api_calls':len(ledger['records'])}
    save(OUT/'audit_summary.json',info)
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (x['condition'],x['policy'],f"{x['mean_return']:.4f}", '-' if x['seed_sd'] is None else f"{x['seed_sd']:.4f}",f"{x['agreement']:.1%}",f"{x['query']:.2f}"))+'</tr>' for x in summaries)
    htmltext='''<!doctype html><html><meta charset="utf-8"><title>Track C exploratory pilot</title><style>body{font:16px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 24px;color:#172b3a}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.note{background:#edf4f5;padding:16px;border-left:4px solid #3c7a89}code{font-size:14px}</style><h1>Track C: diplomatic reciprocal suspension</h1><p class="note">Exploratory pilot only. Real source text motivates discourse functions; numerical utilities, private types and responses are synthetic assumptions. No untouched historical test set or claim of actual diplomatic effects.</p><h2>Finding to investigate</h2><p>Questions need not be useful when offers cheaply reveal acceptance conditions. Exact belief planning identifies a boundary where rejection costs and query reliability make private inquiry valuable. Compare learned behavior with this attainable public-information benchmark; do not assume PPO will win.</p><img src="information_value.png"><img src="policy_comparison.png"><h2>Results</h2><p>Returns are undiscounted terminal utility minus all costs. Error bars describe training-seed variability. Planner knows public transition rules but never receives the true type.</p><table><tr><th>Condition</th><th>Policy</th><th>Return</th><th>Seed SD</th><th>Agreement</th><th>Queries</th></tr>'''+table+'''</table><h2>Language pilot</h2><pre>'''+html.escape(json.dumps(info,indent=2))+'''</pre><p>These are model-to-model roundtrip checks and authored synthetic semantic tests, not human-labeled source accuracy. In the source audit, “if” was sometimes overinterpreted as a reciprocal exchange; model labels require review. One excerpt was not an exact source substring.</p><h2>Discourse paths</h2><img src="action_transitions.png"><p>Full utterances, interpreted terms, observations, reward components and termination reasons are in trajectories.jsonl; selected paths in discourse_cases.json. Count matrices exclude cross-episode transitions. Multiple parameter settings from C-012 remain one historical motif.</p><h2>Limits and next decision</h2><p>Retain all pilot results. Formal submission work still needs event/document-group splits and untouched evaluation, independent human discourse review, full discount/entropy/reward ablations, richer source-grounded contexts, reproducibility by a teammate, and a four-page paper. Belief masking alone does not remove acceptance histories. The signal effect and rejection penalties require separate sensitivity analysis rather than historical interpretation.</p></html>'''
    (OUT/'report.html').write_text(htmltext,encoding='utf-8')
    print(json.dumps(info),flush=True)
    print(json.dumps([x for x in summaries if x['condition'] in ('base','grid_r0.32_q0.9')],indent=2),flush=True)

if __name__=='__main__': main()
