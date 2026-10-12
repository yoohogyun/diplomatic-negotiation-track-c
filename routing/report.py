"""Replay every pilot path, audit cached requests, and report paired resource outcomes."""
import hashlib, html, json, time
from dataclasses import asdict
import numpy as np
from routing import ROOT, OUT
from routing.run import load_models
from pilot.env import Config
from pilot.live import write, LEDGER
from study.env import LunaEnv


def main():
    selection=json.loads((OUT/'selection.json').read_text());summary=json.loads((OUT/'test_summary.json').read_text())
    cases=json.loads((OUT/'test_case_metrics.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    cheap,gates=load_models()
    import torch
    rows=[]
    for p in sorted((OUT/'cases').glob('*.json')):rows.extend(json.loads(p.read_text(encoding='utf-8')))
    contexts={c['id']:c for c in json.loads((ROOT/'study_results/contexts.json').read_text(encoding='utf-8'))}
    bank=json.loads((ROOT/'sol_results/language_bank.json').read_text(encoding='utf-8'));envs={s:LunaEnv(s,bank=bank) for s in ('dev','test')}
    public_requests={};turns=0;cheap_decisions=0
    for r in rows:
        env=envs[r['split']];o,_=env.reset(seed=1,options={'config':Config(**r['config']),'context':contexts[r['context']],
            'variant':0,**r['hidden_evaluation_only']})
        ret=0.;usage={'sol_calls':0.,'prompt_tokens':0.,'completion_tokens':0.,'reasoning_tokens':0.,'api_usd':0.}
        for i,trace in enumerate(r['trajectory']):
            assert asdict(env.s)==trace['before'];turns+=1
            if r['policy']!='planner':
                d=r['decisions'][i];assert d['public_state']==asdict(env.s)
                a=int(cheap.predict(o,deterministic=True)[0]);assert a==d['cheap_action'];cheap_decisions+=1
                with torch.no_grad():q=torch.stack([m(torch.as_tensor(o)) for m in gates]).mean(0)
                regret=float(q.max()-q[a]);assert abs(regret-d['predicted_regret'])<1e-6
                if r['policy']=='initial_sol':expected_route=env.s.t==0
                else:
                    threshold=-1. if r['policy']=='always_sol' else float('inf') if r['policy']=='cheap' else float(r['policy'].removeprefix('gate_'))
                    expected_route=regret>threshold
                assert d['route_to_sol']==expected_route
                if d['route_to_sol']:
                    tag=d['api_tag'];p=ROOT/'sol_results/api'/tag
                    request=json.loads(p.with_suffix('.request.json').read_text(encoding='utf-8'))
                    payload=json.loads(request['messages'][1]['content'])
                    assert set(payload)=={'source_context','our_measure','their_measure','public','actions'}
                    assert payload['public']=={'state':asdict(env.s),'config':r['config'],'last_interpreted_function':env.last_kind}
                    assert payload['source_context']==contexts[r['context']]['source']['sentence']
                    assert request['reasoning']['effort']=='medium'
                    parsed=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert parsed['parsed']['action']==d['selected_action']
                    raw=json.loads(p.with_suffix('.raw.json').read_text(encoding='utf-8'));tokens=raw['usage'];public_requests[tag]=tokens
                    usage['sol_calls']+=1
                    for k in ('prompt_tokens','completion_tokens'):usage[k]+=tokens.get(k,0)
                    usage['reasoning_tokens']+=tokens.get('completion_tokens_details',{}).get('reasoning_tokens',0)
                    usage['api_usd']+=tokens['cost']
                else:assert d['selected_action']==a
                assert trace['action']==d['selected_action']
            o,_,done,_,actual=env.step(trace['action']);assert actual==trace;ret+=actual['common_reward']
        assert done and abs(ret-r['net_utility'])<1e-9
        assert not r['acceptance_violations']
        for k,v in usage.items():assert abs(v-r[k])<1e-9
    fresh=json.loads((OUT/'fresh_numeric_test.json').read_text());old=json.loads((ROOT/'study_results/numeric_conditions.json').read_text())
    seen={json.dumps(c,sort_keys=True) for rs in old.values() for c in rs}
    seen.update(json.dumps(c,sort_keys=True) for c in json.loads((ROOT/'improvement_results/fresh_numeric_test.json').read_text()))
    assert len({json.dumps(c,sort_keys=True) for c in fresh})==16
    assert all(json.dumps(c,sort_keys=True) not in seen for c in fresh)
    start=json.loads((OUT/'test_started.json').read_text());assert selection['time']<start['time']
    assert hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()==start['selection_sha256']
    name=selection['policy'];hybrid=summary[name];baseline=summary['always_sol']
    ci_diffs=[]
    for ci in range(16):
        matched={p:np.mean([r['net_utility'] for r in cases if r['policy']==p and r['condition']==ci]) for p in (name,'always_sol')}
        ci_diffs.append(float(matched[name]-matched['always_sol']))
    rng=np.random.default_rng(17012026);boots=np.array(ci_diffs)[rng.integers(16,size=(10000,16))].mean(1)
    ledger=json.loads(LEDGER.read_text());records=[r for r in ledger['records'] if r['tag'].startswith('sol_medium_routing_')]
    spent=sum(r.get('cost',0) for r in ledger['records']);unresolved=[r['tag'] for r in ledger['records'] if r['status']!='validated']
    recovery_path=OUT/'transport_recovery.json'
    recoveries=json.loads(recovery_path.read_text()) if recovery_path.exists() else {'replacements':{}}
    allowed_pending={'sol_medium_'+t for t in recoveries['replacements']}
    assert set(unresolved)==allowed_pending
    reserved=sum(r['reserved'] for r in ledger['records'] if 'cost' not in r)
    assert spent+reserved<9.
    validated=[r for r in records if r['status']=='validated']
    assert len(validated)==len(public_requests)
    result={'selected':name,'threshold':selection['threshold'],'test_cells':32,'numeric_conditions':16,'source_documents':2,
        'utility_delta':hybrid['net_utility']-baseline['net_utility'],
        'utility_retention_ratio':hybrid['net_utility']/baseline['net_utility'],
        'calls_reduction_fraction':1-hybrid['sol_calls']/baseline['sol_calls'],
        'token_reduction_fraction':1-(hybrid['prompt_tokens']+hybrid['completion_tokens'])/(baseline['prompt_tokens']+baseline['completion_tokens']),
        'api_cost_reduction_fraction':1-hybrid['api_usd']/baseline['api_usd'],
        'paired_numeric_deltas':ci_diffs,'exploratory_numeric_cluster_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
        'uncertainty_scope':'16 numerical draws only, averaging two fixed exposed sources and one cached model draw per public prompt; not a historical or model-seed CI.',
        'actual_new_unique_calls':len(validated),'new_attempts_including_timeout':len(records),
        'actual_new_api_usd':sum(r.get('cost',0) for r in records),'cumulative_api_usd':spent,
        'unresolved_transport_reserve_usd':reserved,'conservative_cumulative_usd':spent+reserved,
        'gate_training':json.loads((OUT/'gate_training.json').read_text()),'test_summary':summary}
    result['secondary_frontier']={n:{'utility_delta':r['net_utility']-baseline['net_utility'],
        'calls_reduction_fraction':1-r['sol_calls']/baseline['sol_calls'],
        'api_cost_reduction_fraction':1-r['api_usd']/baseline['api_usd']} for n,r in summary.items() if n.startswith('gate_') or n=='initial_sol'}
    write(OUT/'findings.json',result)
    write(OUT/'audit.json',{'passed':True,'raw_paths_replayed':len(rows),'raw_turns_replayed':turns,
        'learned_decisions_reproduced':cheap_decisions,'unique_public_requests_verified':len(public_requests),
        'test_configuration_overlap':0,'selection_frozen_before_test':True,'acceptance_violations':0,
        'unresolved_transport_calls_retained':unresolved,'reserved_for_unresolved_usd':reserved})
    labels={'always_sol':'Always Sol medium','cheap':'Learned policy only','planner':'Public-rule planner','initial_sol':'Sol at first decision',name:'Selective Sol (development-selected)'}
    def table(data):
        s='<table><tr><th>Method</th><th>Net utility</th><th>Agreement</th><th>Sol calls/episode</th><th>Tokens/episode</th><th>Model-call $/episode</th><th>Queries/episode</th><th>Signals/episode</th></tr>'
        for n,r in data.items():
            s+=f'<tr><td>{html.escape(labels.get(n,n))}</td><td>{r["net_utility"]:.5f}</td><td>{r["agreement"]:.1%}</td><td>{r["sol_calls"]:.3f}</td><td>{r["prompt_tokens"]+r["completion_tokens"]:.1f}</td><td>{r["api_usd"]:.5f}</td><td>{r["queries"]:.3f}</td><td>{r["signals"]:.3f}</td></tr>'
        return s+'</table>'
    allocation={}
    for r in rows:
        if r['split']!='test' or not r['policy'].startswith('gate_'):continue
        for d in r['decisions']:
            key=r['policy']+':'+('accepted' if d['public_state']['accepted'] else 'not_accepted')
            z=allocation.setdefault(key,{'decisions':0.,'sol_calls':0.});z['decisions']+=r['weight']/32;z['sol_calls']+=r['weight']*d['route_to_sol']/32
    write(OUT/'routing_allocation.json',allocation)
    examples=[]
    for r in rows:
        if r['split']=='test' and r['policy'].startswith('gate_') and any(d['route_to_sol'] and d['cheap_action']!=d['selected_action'] for d in r['decisions']):
            examples.append(r)
    write(OUT/'routed_examples.json',examples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,key,xlabel in zip(axes,['sol_calls','api_usd'],['Expected Sol calls / episode','Expected API USD / episode']):
        gate_names=[f'gate_{t}' for t in [.2,.1,.06,.03,.01]]
        ax.plot([summary[n][key] for n in gate_names],[summary[n]['net_utility'] for n in gate_names],'-o',color='#176b87',label='Learned gate thresholds')
        for n in gate_names:
            ax.annotate(n.removeprefix('gate_'),(summary[n][key],summary[n]['net_utility']),xytext=(4,5),textcoords='offset points',fontsize=8)
        for n,color,marker in [('always_sol','#a53c32','s'),('cheap','#555555','x'),('initial_sol','#d28b00','D')]:
            ax.scatter(summary[n][key],summary[n]['net_utility'],color=color,marker=marker,label=n.replace('_',' '),zorder=4)
        ax.axhline(summary['planner']['net_utility'],color='#777777',linestyle=':',label='Known-rule planner')
        ax.set(xlabel=xlabel,ylabel='Expected net utility');ax.grid(alpha=.18)
    axes[0].legend(fontsize=8);fig.suptitle('Exploratory selective-reasoning pilot: 16 numeric settings, 2 fixed sources',fontsize=11)
    fig.savefig(OUT/'utility_cost.png',dpi=180);fig.savefig(OUT/'utility_cost.pdf');plt.close(fig)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Selective Sol pilot</title>
    <style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#18242c}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ccd5db;text-align:right;padding:9px}}td:first-child,th:first-child{{text-align:left}}h1,h2{{line-height:1.25}}.note{{background:#eff5f7;padding:18px}}blockquote{{border-left:3px solid #54798a;padding-left:18px}}</style>
    <h1>Selective Sol routing for diplomatic negotiation</h1>
    <p class="note">Net utility difference from always using Sol: {result['utility_delta']:+.5f}; reductions in calls: {result['calls_reduction_fraction']:.1%}, tokens: {result['token_reduction_fraction']:.1%}, and model-call cost: {result['api_cost_reduction_fraction']:.1%}. Agreement: {hybrid['agreement']:.1%}. Report all outcomes of the fixed pilot, including failures.</p>
    <h2>Evaluation on new numerical settings</h2>{table(summary)}<img src="utility_cost.png" alt="Utility versus Sol calls and API cost" style="width:100%">
    <p>The planned evaluation contains 16 new numerical settings across two previously used real documents, for 32 cells. Hidden-type and query-response branches are probability-weighted. Style 0 and the document/language bank were used previously. Numerical utilities and opponent responses are synthetic.</p>
    <h2>Method and development selection</h2><p>The existing distilled-plus-PPO policy at seed 17 is frozen. Three small MLPs predict public-state action values using only the existing 40,706 training states. Sol medium is called when the predicted loss of the original policy action exceeds a threshold. No planner is called at deployment. Training uses a public-rule planner teacher and the existing PPO policy, so this is not an equal-training-cost comparison.</p>
    <p>Development used Sol on eight numerical settings across two documents. Among candidates with utility loss at most 0.01 relative to always using Sol, threshold {selection['threshold']} minimizes calls. It was frozen before the new numerical evaluation and not changed based on test results. Development selected a zero-call candidate. Before testing, an addendum specified auxiliary curves for all existing thresholds and a first-decision-only Sol baseline. A favorable auxiliary threshold must not be reported as success of the development-selected method.</p>{table(selection['summary'])}
    <h2>Cost, checks, and interpretation</h2><p>Table costs are expected values per hypothetical deployment, counting cache hits as logical calls. The actual pilot reused duplicate public prompts: {len(validated)} successful responses cost ${result['actual_new_api_usd']:.6f}, with ${spent:.6f} cumulatively billed across responses. Interrupted requests remain unresolved, with an additional reserve of ${reserved:.6f}; the conservative cumulative upper bound is ${spent+reserved:.6f}. Deployment costs exclude language-bank construction, teacher computation, and training. They do not measure CPU runtime or end-to-end latency reductions.</p>
    <p>A separate process replayed {len(rows)} development/evaluation trajectories and {turns} transitions. Inputs contain only public state; the actual hidden type is never provided to the model. Requests, tokens, charges, call triggers, and acceptance conditions were audited.</p>
    <p>The paired bootstrap 95% interval for the utility difference across numerical settings is [{np.quantile(boots,.025):+.5f}, {np.quantile(boots,.975):+.5f}]. It reflects exploratory numerical variation conditional on two fixed documents and single cached responses. It does not establish historical generalization, model-sampling variability, or confirmatory non-inferiority. Routing itself is not claimed as a new algorithm.</p>
    <h2>Real source excerpts</h2>'''
    for ctx in protocol['test_contexts']:
        page+=f'<blockquote>{html.escape(ctx["source"]["sentence"])}</blockquote><p>{html.escape(ctx["source"]["doc_title"])} · {html.escape(str(ctx["source"]["date"]))}</p>'
    page+='</html>';(OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gate_training','test_summary')},indent=2),flush=True)


if __name__=='__main__':main()
