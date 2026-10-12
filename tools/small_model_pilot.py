"""Offline matched audit of Luna plus public-belief planning versus cached Sol."""
import json,time,collections
from pathlib import Path
from dataclasses import asdict
import pilot
import numpy as np
import pandas as pd
from pilot.env import Config
from pilot.live import write
from study.env import LunaEnv
from study.evaluate import run_episode,planner_for,encoded_bank

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'small_model_results'


def main():
    OUT.mkdir(exist_ok=True)
    write(OUT/'scope.json',{'time':time.time(),'reason':'User requested reusing existing evidence for small Luna plus strategy module; time constrained, no new paid experiment.',
        'comparison':'Same existing 64 conditions (16 numeric settings x 4 documents), style0, Luna execution bank.',
        'module':'Public-belief exact planning over declared opponent response model; no actual hidden type. Existing implementation now audited as a deployable known-model controller.',
        'limits':'Previously inspected conditions; this is an engineering pilot, not new held-out evidence. Planning is a strong known-rule benchmark, not a new algorithm. Cached language bank, not newly generated online dialogue.'})
    env=LunaEnv('test');configs=[Config(**c) for c in json.loads((ROOT/'study_results/numeric_conditions.json').read_text())['test'][:16]]
    rows=[];timings=[]
    for ctx in env.contexts:
        for ci,cfg in enumerate(configs):
            planner_for.cache_clear();start=time.perf_counter();calls=[0]
            def actor(o,e):
                calls[0]+=1
                return planner_for(e.cfg,encoded_bank(e))(e.s)[1]
            for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
                ph=cfg.reliability if hard else 1-cfg.reliability
                for u,py in [(0.,ph),(.999999,1-ph)]:
                    if pt*py<=0:continue
                    z=run_episode(env,actor,cfg,ctx,0,hard,u,123)
                    assert z['acceptance_violations']==0
                    rows.append(dict(policy='Luna_public_planning',context=ctx['id'],condition=ci,config=asdict(cfg),weight=pt*py,
                        hidden_evaluation_only={'hard':hard,'query_uniform':u},**z))
            timings.append({'context':ctx['id'],'condition':ci,'seconds':time.perf_counter()-start,'decisions':calls[0]})
    write(OUT/'planning_paths.json',rows);write(OUT/'planning_local_timing.json',timings)
    grouped=collections.defaultdict(lambda:collections.defaultdict(float))
    for r in rows:
        for k in ['net_utility','agreement','queries','signals','length']:grouped[(r['context'],r['condition'])][k]+=r['weight']*r[k]
    planning={k:float(np.mean([r[k] for r in grouped.values()])) for k in next(iter(grouped.values()))}
    controls=pd.read_csv(ROOT/'sol_results/matched_policy_controls.csv');direct=pd.read_csv(ROOT/'sol_results/direct_comparison.csv')
    previous=float(controls[(controls.bank=='Luna')&(controls.policy=='Planner_None')].net_utility.iloc[0]);assert abs(planning['net_utility']-previous)<1e-10
    learned=controls[(controls.bank=='Luna')&controls.policy.str.startswith('Selected_')]
    baseline=direct[direct.policy=='Sol_medium_luna'].iloc[0]
    table=[]
    for name,r,kind in [('Luna alone (none)',direct[direct.policy=='Luna_none_Luna_bank'].iloc[0],'API'),
        ('Luna + learned strategy (3 seeds)',learned[['net_utility','agreement','queries','signals','length']].mean(),'local neural policy'),
        ('Luna + public-belief planning',planning,'local known-rule planning'),('Sol medium',baseline,'API')]:
        table.append(dict(method=name,net_utility=float(r['net_utility']),agreement=float(r['agreement']),queries=float(r['queries']),signals=float(r['signals']),
            length=float(r['length']),strategic_api_calls_per_episode=0. if kind!='API' else float(r['length']),strategy_compute=kind))
    pd.DataFrame(table).to_csv(OUT/'matched_summary.csv',index=False)
    # Reconstruct actual logical Sol requests per episode; cache hits still count.
    from study.luna_policy import RULES
    from pilot.env import NAMES
    import hashlib
    costs=[]
    for p in (ROOT/'sol_results/direct_medium_luna_cases').glob('*.json'):
        for r in json.loads(p.read_text(encoding='utf-8')):
            usd=0.;tokens=0.;last='ambiguity'
            for tr in r['trajectory']:
                public={'state':tr['before'],'config':r['config'],'last_interpreted_function':last}
                digest=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()[:18]
                tag=f'sol_medium_policy_{r["context"]}_{r["condition"]}_{digest}'
                raw=json.loads((ROOT/'sol_results/api'/f'{tag}.raw.json').read_text(encoding='utf-8'))
                usd+=raw['usage']['cost'];tokens+=raw['usage']['prompt_tokens']+raw['usage']['completion_tokens'];last=tr['meaning']['kind']
            costs.append({'weight':r['weight'],'usd':usd,'tokens':tokens})
    findings={'matched_cases':64,'freshly_replayed_planning_paths':len(rows),'planning':planning,
        'learned_luna_utility':float(learned.net_utility.mean()),'sol_medium_utility':float(baseline.net_utility),
        'learned_retention_ratio':float(learned.net_utility.mean()/baseline.net_utility),
        'planning_minus_sol':planning['net_utility']-float(baseline.net_utility),
        'sol_strategic_calls_per_episode':float(baseline.length),
        'sol_strategic_api_usd_per_episode':sum(r['weight']*r['usd'] for r in costs)/64,
        'sol_strategic_tokens_per_episode':sum(r['weight']*r['tokens'] for r in costs)/64,
        'planning_local_seconds_total':sum(t['seconds'] for t in timings),
        'local_timing_scope':'64 complete enumerations of hidden/query branches, clearing planning cache before each cell; not a latency comparison with API.',
        'strategy_api_calls_for_controller':0,'new_api_calls_for_this_audit':0,
        'cost_scope':'Only strategy-selection API eliminated; cached language realization/interpretation and offline teacher/training costs are excluded. No end-to-end cost reduction percentage claim.'}
    write(OUT/'findings.json',findings)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>GPT-5.6 Luna strategy-module pilot</title><style>body{font:16px/1.65 system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#173440}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd}pre{white-space:pre-wrap;background:#eef4f5;padding:15px}</style><h1>GPT-5.6 Luna with a public-state strategic controller</h1>
    <p>In a synthetic negotiation grounded in supplied diplomatic sentences, Luna realizes utterances while a local controller selects queries, signals, concessions, and ratification. The controller uses action costs and beliefs updated from opponent responses.</p>'''
    page+=pd.DataFrame(table).round(5).to_html(index=False)
    page+=f'''<p>On the same existing 64 conditions, the learned policy achieves {findings['learned_retention_ratio']:.1%} of Sol medium net utility. Planning with the declared public transition rules differs from Sol by {findings['planning_minus_sol']:+.5f}. Replaying {len(rows)} trajectories reproduced the saved planning result.</p>
    <p>Sol action selection averages {findings['sol_strategic_calls_per_episode']:.3f} calls, {findings['sol_strategic_tokens_per_episode']:.1f} tokens, and ${findings['sol_strategic_api_usd_per_episode']:.5f} per negotiation. Both local controllers make zero model calls for action selection. These figures exclude shared utterance-bank generation/interpretation and teacher/training costs, so they do not measure an end-to-end service cost reduction.</p>
    <p>Real documents ground the expressions and context; utilities, types, and responses are synthetic. The planner knows the declared opponent response model exactly. It is a strong benchmark under known rules, not evidence of general diplomatic performance or a new algorithm. Luna alone uses none while Sol uses medium reasoning, so this is not an isolated model-size comparison. Evaluation uses a fixed model-generated/interpreted bank, not newly generated online dialogue.</p>
    <p>The paper investigates whether separating strategic control over public beliefs and action costs can improve negotiation utility and reduce strategy-selection model calls for a smaller language model. PPO is a required course baseline and one controller implementation, not a claimed new algorithm.</p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(findings,indent=2),flush=True)


if __name__=='__main__':main()
