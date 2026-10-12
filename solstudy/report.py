"""Matched model comparison and explicit evidence for/against the earlier storyline."""
import json,gzip,collections,html,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solstudy import OUT,ROOT,IMPROVED
from study import OUT as LUNA
from pilot.live import write,LEDGER

def matched_selected(label,path):
    import torch
    from stable_baselines3 import PPO
    from study.env import LunaEnv
    from study.evaluate import run_episode
    from pilot.env import Config
    torch.set_num_threads(1)
    directory=ROOT/'improvement_results' if label=='Luna' else IMPROVED
    selected=json.loads((directory/'selection.json').read_text())['selected']
    bank=json.loads((path/'language_bank.json').read_text(encoding='utf-8'));env=LunaEnv('test',bank=bank)
    configs=[Config(**c) for c in json.loads((path/'numeric_conditions.json').read_text())['test'][:16]]
    metrics=[]
    for seed in [17,29,43]:
        model=PPO.load(directory/f'models/{selected}_{seed}',device='cpu');records=[]
        for ctx in env.contexts:
            for ci,cfg in enumerate(configs):
                for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
                    ph=cfg.reliability if hard else 1-cfg.reliability
                    for u,py in [(0.,ph),(.999999,1-ph)]:
                        if pt*py<=0:continue
                        z=run_episode(env,lambda o,e:int(model.predict(o,deterministic=True)[0]),cfg,ctx,0,hard,u,123)
                        records.append(dict(policy=selected,seed=seed,event_group=ctx['event_group'],context=ctx['id'],condition=ci,weight=pt*py,**z))
        file=OUT/f'matched_selected_cases/{label}_{seed}.json';write(file,records)
        row=aggregate([file],label+'_selected_'+str(seed));row.update(bank=label,policy='Selected_'+selected+'_'+str(seed));metrics.append(row)
    return metrics

def aggregate(paths,label):
    cases=collections.defaultdict(lambda:collections.defaultdict(float));counts=0
    for p in paths:
        for r in json.loads(p.read_text(encoding='utf-8')):
            tr=r['trajectory'];assert r['acceptance_violations']==0
            assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
            assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
            for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
            repeated=any(a['response']=='reject' and a['meaning']['kind']=='offer' and a['meaning']==b['meaning'] for a,b in zip(tr,tr[1:]))
            row=cases[(r['event_group'],r['context'],r['condition'])];w=r['weight'];row['weight']+=w
            for k in ['net_utility','agreement','queries','signals','cost','length']:row[k]+=w*r[k]
            row['repeated_rejected_offer']+=w*repeated;counts+=1
    assert cases and all(abs(r['weight']-1)<1e-8 for r in cases.values())
    df=pd.DataFrame([dict(group=k[0],context=k[1],condition=k[2],**v) for k,v in cases.items()])
    df.to_csv(OUT/(label+'_case_metrics.csv'),index=False)
    metrics=df.groupby('group')[['net_utility','agreement','queries','signals','cost','length','repeated_rejected_offer']].mean().mean().to_dict()
    return dict(policy=label,cases=len(cases),raw_paths=counts,**metrics)

def main():
    direct=[aggregate(sorted((LUNA/'luna_policy_cases').glob('*.json')),'Luna_none_Luna_bank')]
    for mode in ['none_luna','medium_luna','none_sol']:
        assert (OUT/f'direct_{mode}_complete.json').exists(),mode+' incomplete'
        direct.append(aggregate(sorted((OUT/f'direct_{mode}_cases').glob('*.json')),'Sol_'+mode))
    direct_df=pd.DataFrame(direct);direct_df.to_csv(OUT/'direct_comparison.csv',index=False)
    assert all(x['cases']==64 for x in direct)
    mainrows=[]
    for label,path in [('Luna',LUNA),('Sol',OUT)]:
        for r in json.loads((path/'seed_metrics.json').read_text()):mainrows.append(dict(model=label,**r))
    md=pd.DataFrame(mainrows);md.to_csv(OUT/'all_seed_metrics.csv',index=False)
    cols=['net_utility','agreement','queries','signals','cost','length']
    ms=md.groupby(['model','policy'])[cols].mean();ms.to_csv(OUT/'main_comparison.csv')
    imp=[]
    for label,path in [('Luna',ROOT/'improvement_results'),('Sol',IMPROVED)]:
        chosen=json.loads((path/'selection.json').read_text())['selected']
        for r in json.loads((path/'seed_metrics.json').read_text()):imp.append(dict(model=label,selected_method=chosen,**r))
    ip=pd.DataFrame(imp);ip.to_csv(OUT/'improvement_all_seeds.csv',index=False)
    ips=ip.groupby(['model','policy'])[cols].mean();ips.to_csv(OUT/'improvement_comparison.csv')
    languages=[]
    for label,path in [('Luna',LUNA),('Sol',OUT)]:languages.append(dict(model=label,**json.loads((path/'language_summary.json').read_text())))
    ld=pd.DataFrame(languages);ld.to_csv(OUT/'language_comparison.csv',index=False)
    # Match policy controls to the original first-16-config, style-0 direct comparison.
    matched=[]
    for label,path in [('Luna',LUNA),('Sol',OUT)]:
        for name in ['Heuristic_None','Planner_None','PPO_base_17','PPO_base_29','PPO_base_43']:
            cs=collections.defaultdict(lambda:collections.defaultdict(float))
            with gzip.open(path/'raw'/('test_'+name+'.jsonl.gz'),'rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line)
                    if r['condition']>=16 or r['variant']!=0:continue
                    z=cs[(r['context'],r['condition'])]
                    for k in cols:z[k]+=3*r['weight']*r[k]
            matched.append(dict(bank=label,policy=name,**pd.DataFrame(cs.values()).mean().to_dict()))
        matched+=matched_selected(label,path)
    matched_df=pd.DataFrame(matched);matched_df.to_csv(OUT/'matched_policy_controls.csv',index=False)
    # Audit all local learned policy and direct trajectories; retain scope boundaries.
    n=0;turns=0
    for folder in [OUT,IMPROVED]:
        for p in (folder/'raw').glob('*.gz'):
            weights=collections.defaultdict(float)
            with gzip.open(p,'rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);tr=r['trajectory'];n+=1;turns+=len(tr)
                    assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
                    assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
                    assert r['acceptance_violations']==0
                    for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
                    weights[(r['context'],r['condition'])]+=r['weight']
            assert all(abs(w-1)<1e-8 for w in weights.values())
    ledger=json.loads(LEDGER.read_text());records=ledger['records'];sol=[r for r in records if r['tag'].startswith('sol_')]
    assert all(r['status']=='validated' and 'cost' in r for r in sol)
    raw_cost=0;usage=collections.defaultdict(lambda:collections.defaultdict(float))
    for p in (OUT/'api').glob('*.raw.json'):
        raw=json.loads(p.read_text());assert raw['model'].startswith('openai/gpt-5.6-sol');raw_cost+=raw['usage']['cost']
        role='direct_medium' if p.name.startswith('sol_medium_policy') else ('direct_none' if p.name.startswith('sol_none_policy') else 'language')
        u=raw['usage'];usage[role]['calls']+=1;usage[role]['cost']+=u['cost']
        for field in ['prompt_tokens','completion_tokens']:usage[role][field]+=u.get(field,0)
        usage[role]['reasoning_tokens']+=u.get('completion_tokens_details',{}).get('reasoning_tokens',0)
    assert abs(raw_cost-sum(r['cost'] for r in sol))<1e-8
    audit={'learned_policy_raw_paths':n,'learned_policy_raw_turns':turns,'direct_raw_paths':sum(r['raw_paths'] for r in direct[1:]),
      'checks_passed':True,'sol_calls':len(sol),'sol_cost':sum(r['cost'] for r in sol),'all_authorized_cost':sum(r.get('cost',0) for r in records),
      'unresolved_calls':sum('cost' not in r for r in records)}
    audit['account_usage_increment']=ledger['final_key_usage']['usage']-ledger['baseline_key_usage']['usage']
    audit['response_cost_minus_account_increment']=audit['all_authorized_cost']-audit['account_usage_increment']
    audit['account_counter_matches_response_sum']=abs(audit['response_cost_minus_account_increment'])<1e-6
    audit['budget_accounting']='Use full response cost sum conservatively; retain any account-counter difference without assuming its cause.'
    write(OUT/'audit.json',audit)
    usage_df=pd.DataFrame([dict(role=k,**v) for k,v in usage.items()]);usage_df.to_csv(OUT/'inference_usage.csv',index=False)
    dd={r['policy']:r for r in direct}
    matched_selected_mean=float(matched_df[(matched_df.bank=='Luna')&matched_df.policy.str.startswith('Selected_')].net_utility.mean())
    matched_sol_selected_mean=float(matched_df[(matched_df.bank=='Sol')&matched_df.policy.str.startswith('Selected_')].net_utility.mean())
    luna_bank=json.loads((LUNA/'language_bank.json').read_text(encoding='utf-8'));sol_bank={r['id']:r for r in json.loads((OUT/'language_bank.json').read_text(encoding='utf-8'))}
    semantic_differences=[r['id'] for r in luna_bank if r['split']=='test' and r['variant']==0 and r['meaning']!=sol_bank[r['id']]['meaning']]
    matched_heuristic=float(matched_df[(matched_df.bank=='Luna')&(matched_df.policy=='Heuristic_None')].net_utility.iloc[0])
    conclusions={'model_only_delta_none':dd['Sol_none_luna']['net_utility']-dd['Luna_none_Luna_bank']['net_utility'],
      'sol_medium_minus_none':dd['Sol_medium_luna']['net_utility']-dd['Sol_none_luna']['net_utility'],
      'sol_medium_minus_matched_selected_luna_mean':dd['Sol_medium_luna']['net_utility']-matched_selected_mean,
      'sol_medium_minus_matched_selected_sol_mean':dd['Sol_medium_luna']['net_utility']-matched_sol_selected_mean,
      'matched_luna_selected_mean':matched_selected_mean,'matched_sol_selected_mean':matched_sol_selected_mean,
      'test_style_zero_bank_semantic_differences':semantic_differences,
      'sol_medium_minus_matched_heuristic':dd['Sol_medium_luna']['net_utility']-matched_heuristic,
      'comparison_limit':'Medium changes reasoning and output budget; same prompts and fixed execution bank. Constructed PPO failures are not general LLM failures.'}
    write(OUT/'conclusions.json',conclusions)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    display=['Luna / none','Sol / none','Sol / medium','Sol / own bank']
    for ax,col,title in zip(axes,['net_utility','agreement'],['Matched net utility','Agreement rate']):
        ax.bar(range(4),direct_df[col],color=['#888888','#357f87','#1b565c','#a5bfc2']);ax.set(xticks=range(4),xticklabels=display,title=title);ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(OUT/'direct_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,['Luna','Sol']):
        p=ips.loc[model];chosen=json.loads(((ROOT/'improvement_results') if model=='Luna' else IMPROVED).joinpath('selection.json').read_text())['selected']
        names=['base','Heuristic',chosen,'Planner'];ax.bar(range(4),p.loc[names,'net_utility'],color=['#888','#7aa0a5','#246f73','#b8c6c6']);ax.set(title=model+' execution bank',xticks=range(4),xticklabels=['PPO','Heuristic','Selected method','Planner'],ylabel='Follow-up net utility');ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(OUT/'improvement_comparison.png',dpi=180);plt.close(fig)
    tablecols=['policy','net_utility','agreement','queries','signals','repeated_rejected_offer']
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>GPT-5.6 Sol replication</title><style>body{max-width:1150px;margin:35px auto;padding:0 24px;font:16px/1.6 system-ui;color:#16393f}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;background:#f0f5f5;padding:16px}</style><h1>GPT-5.6 Sol replication results</h1><p>The existing GPT-5.6 Luna conditions were repeated with GPT-5.6 Sol. PPO is an established algorithm; the PPO selector and distilled system are implementations for this environment. Direct model comparisons distinguish model behavior from weaknesses of a particular learned controller.</p><h2>Direct action selection under matched conditions</h2><p>The comparison uses 64 conditions, the same prompt, and style 0. The first three rows share the Luna execution bank. Sol none uses the same reasoning setting as Luna. Sol medium enables reasoning and raises the combined reasoning/output cap to 4,096 tokens, so its compute budget differs. Only the final row uses the Sol execution bank. One response per model and public state is cached; variation across model samples was not measured.</p><img src="direct_comparison.png">'''+direct_df[tablecols].round(5).to_html(index=False)+'''<h3>Policy controls on the same subset as direct models</h3>'''+matched_df.round(5).to_html(index=False)+'''<h2>Language checks</h2>'''+ld.drop(columns=['note']).to_html(index=False)+'''<p>These are consistency rates when the same model generates and interprets language. They are not accuracy against human labels. All mismatches remain in the transitions.</p><h2>Original full evaluation: learned policies by model</h2>'''+ms.round(5).to_html()+'''<h2>Replication of the improvement methods</h2><img src="improvement_comparison.png">'''+ips.round(5).to_html()+'''<p>Replications use the same training budget, three seeds, development selection, and follow-up numerical conditions. Changing the language model can change bank interpretations and transitions, so full-system outcomes must be distinguished from direct action-selection effects. Teacher computation using known rules is an additional resource; this is not an equal-resource model-free learning comparison.</p><h2>Validation and cost</h2><pre>'''+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'''</pre><p>This synthetic environment is grounded in real diplomatic documents. Its two test event groups and source texts were already used, so model replication is not new historical validation. The <a href="https://developers.openai.com/api/docs/models/gpt-5.6-sol">official Sol model documentation</a> and OpenRouter endpoint metadata were checked before execution. Actual charges are retained in raw responses and the private shared ledger.</p></html>'''
    narrative=f'''<h2>Implications of the matched comparison</h2><p>With reasoning set to none, the Sol-minus-Luna net utility difference is {conclusions['model_only_delta_none']:+.5f}. Enabling medium reasoning and increasing the output/reasoning budget improves Sol by {conclusions['sol_medium_minus_none']:+.5f} over none. On the same Luna bank and 64 conditions, Sol medium exceeds the three-seed mean of the selected Luna controller by {conclusions['sol_medium_minus_matched_selected_luna_mean']:+.5f} and the strong heuristic by {conclusions['sol_medium_minus_matched_heuristic']:+.5f}. Changes in model family and reasoning compute must therefore be distinguished.</p><p>The PPO results characterize the implemented learned controller. Strong direct Sol performance does not support generalizing the earlier failure to all capable LLMs. Distillation improves the learned policy, but this alone does not establish superiority over the stronger reasoning model.</p>'''
    page=page.replace('<h2>Language checks</h2>',narrative+'<h2>Language checks</h2>',1)
    page=page.replace('<h2>Language checks</h2>',f'<p>The improved controller retrained with the Sol bank averages {matched_sol_selected_mean:.5f} on the same 64 conditions; Sol medium direct selection exceeds it by {conclusions["sol_medium_minus_matched_selected_sol_mean"]:+.5f}. Across the evaluated contexts and style-0 actions, the banks have {len(semantic_differences)} interpretation differences.</p><h2>Language checks</h2>',1)
    page=page.replace('<h2>Validation and cost</h2>','<h2>Observed inference usage</h2>'+usage_df.round(6).to_html(index=False)+'<p>Charges are actual billed amounts for cached public-state responses. They do not represent total online deployment costs or all teacher/training computation.</p><h2>Validation and cost</h2>',1)
    (OUT/'report.html').write_text(page,encoding='utf-8')
    write(OUT/'completion.json',{'complete':True,'paper_written':False,'direct':direct,'audit':audit})
    print(json.dumps({'direct':direct,'audit':audit}),flush=True)

if __name__=='__main__':main()
