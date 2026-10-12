"""Build result tables, figures and discourse evidence. Does not write a paper."""
import json,csv,gzip,html,collections,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import OUT,ROOT
from study.env import RESPONSE_TEXT
from pilot.live import write
from pilot.env import NAMES

def main():
    rows=json.loads((OUT/'seed_metrics.json').read_text());df=pd.DataFrame(rows);df.to_csv(OUT/'seed_metrics.csv',index=False)
    summary=[]
    for name,g in df.groupby('policy',sort=False):
        row={'policy':name,'runs':len(g),'variation_unit':'evaluation RNG seeds' if name=='Random' else ('training seeds' if len(g)>1 else 'single fixed policy/run')}
        for k in ['net_utility','agreement','own_utility','opponent_utility','total_utility','cost','breakdown','length','queries','signals','max_escalation','mean_escalation','reversals','invalid_attempts','acceptance_violations','feasible']:
            row[k]=float(g[k].mean());row[k+'_sd']=float(g[k].std(ddof=1)) if len(g)>1 else None
        summary.append(row)
    write(OUT/'summary.json',summary);pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    # Direct Luna is compared on the exact fixed subset and style, not mixed into main estimates.
    direct=[]
    for p in sorted((OUT/'luna_policy_cases').glob('*.json')):direct+=json.loads(p.read_text(encoding='utf-8'))
    subset=collections.defaultdict(lambda:collections.defaultdict(float))
    for r in direct:
        for k in ['net_utility','agreement','queries','length','cost','acceptance_violations']:
            subset[('Direct_Luna',None,r['event_group'],r['context'],r['condition'])][k]+=r['weight']*r[k]
    for name in ['Heuristic_None','Planner_None','Planner_no_query_None','PPO_base_17','PPO_base_29','PPO_base_43','Bandit_17','Bandit_29','Bandit_43']:
        path=OUT/'raw'/('test_'+name+'.jsonl.gz')
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['condition']>=16 or r['variant']!=0:continue
                for k in ['net_utility','agreement','queries','length','cost','acceptance_violations']:
                    subset[(r['policy'],r['seed'],r['event_group'],r['context'],r['condition'])][k]+=3*r['weight']*r[k]
    sr=[dict(policy=k[0],seed=k[1],group=k[2],context=k[3],condition=k[4],**v) for k,v in subset.items()]
    sdf=pd.DataFrame(sr);sdf.to_csv(OUT/'direct_luna_matched_cases.csv',index=False)
    ss=sdf.groupby(['policy','seed'],dropna=False)[['net_utility','agreement','queries','length','cost','acceptance_violations']].mean().reset_index()
    ss.to_csv(OUT/'direct_luna_matched_summary.csv',index=False)
    bank=json.loads((OUT/'language_bank.json').read_text(encoding='utf-8'));probes=json.loads((OUT/'semantic_probes.json').read_text())
    # Every raw log supplies integer and probability-weighted transition counts.
    matrices={};cases=[];episode_count=0;turn_count=0;violations=0
    categories={}
    for path in sorted((OUT/'raw').glob('test_*.gz')):
        name=path.name.removesuffix('.jsonl.gz');count=np.zeros((11,11),int);weighted=np.zeros((11,11));candidate=[]
        with gzip.open(path,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);episode_count+=1;turn_count+=len(r['trajectory']);violations+=r['acceptance_violations']
                a=[i['action'] for i in r['trajectory']]
                for x,y in zip(a,a[1:]):count[x,y]+=1;weighted[x,y]+=r['weight']
                # Fixed, interpretable case categories; comparisons drawn from the same pool.
                if r['policy'] in ('Planner','Heuristic','PPO_base') and (r['seed'] in (None,17)):
                    if r['queries']>0 and r['agreement']:cat='question_then_agreement'
                    elif r['signals']>0 and r['agreement']:cat='signal_then_agreement'
                    elif r['outcome']=='withdrawal' and not r['feasible']:cat='appropriate_withdrawal'
                    elif r['outcome']=='exhausted':cat='opportunity_exhaustion'
                    elif r['invalid_attempts']>0:cat='invalid_language_or_action'
                    elif r['agreement'] and r['cost']>.15:cat='costly_agreement'
                    elif r['agreement']:cat='direct_agreement'
                    else:cat='other_breakdown'
                    if cat not in categories:categories[cat]=r
        den=count.sum(axis=1);matrix={'counts':count.tolist(),'weighted_counts':weighted.tolist(),
            'probabilities':[None if n==0 else (count[i]/n).tolist() for i,n in enumerate(den)]}
        matrices[name]=matrix
    extra=OUT/'direct_luna_failure_example.json'
    if extra.exists():categories['direct_luna_repeated_rejection']=json.loads(extra.read_text(encoding='utf-8'))
    for cat,r in categories.items():
        for t in r['trajectory']:t.setdefault('opponent_utterance',RESPONSE_TEXT[t['response']])
        cases.append(dict(category=cat,**r))
    write(OUT/'transition_matrices.json',matrices);write(OUT/'discourse_cases.json',cases)
    write(OUT/'evaluation_totals.json',{'raw_logged_episodes':episode_count,'logged_turns':turn_count,'acceptance_violations':violations,
        'interpretation':'Repeated synthetic rollouts; not independent historical cases. Includes random repetitions and evaluation sensitivity runs.'})
    # Main policy plot. Separate panel for semantics diagnostics.
    chosen=['Random','Heuristic','Bandit','PPO_base','Planner_no_query','Planner']
    vals=[next(r for r in summary if r['policy']==p) for p in chosen]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    axes[0].bar(range(6),[r['net_utility'] for r in vals],color=['#999999','#668b9d','#b2b2b2','#bb7847','#83a6a0','#27646b'])
    for i,r in enumerate(vals):
        if r['net_utility_sd'] is not None:axes[0].errorbar(i,r['net_utility'],yerr=r['net_utility_sd'],color='black',capsize=3)
    axes[0].set(xticks=range(6),xticklabels=['Random','Heuristic','Bandit','PPO','No-query\nplanner','Planner'],ylabel='Mean net utility',title='Held-out source-group evaluation')
    axes[0].tick_params(axis='x',rotation=20)
    axes[1].bar(range(6),[r['agreement']*100 for r in vals],color='#5c8c97');axes[1].set(xticks=range(6),xticklabels=['Random','Heuristic','Bandit','PPO','No-query\nplanner','Planner'],ylabel='Agreement (%)',ylim=(0,105),title='Agreement and utility are distinct')
    axes[1].tick_params(axis='x',rotation=20);fig.tight_layout();fig.savefig(OUT/'main_results.png',dpi=180);plt.close(fig)
    # Required ablations, compare the SAME base seed (17), not three-seed mean.
    baseline=next(r for r in rows if r['policy']=='PPO_base' and r['seed']==17)
    ab=[r for r in rows if r['seed']==17 and r['policy'].startswith('PPO_') and r['policy'] not in ('PPO_base','PPO_eval_belief_mask')]
    fig,ax=plt.subplots(figsize=(8.7,4.8));ax.barh([r['policy'].replace('PPO_','') for r in ab],[r['net_utility']-baseline['net_utility'] for r in ab],color='#668b9d');ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='Change in common net utility vs base seed 17',title='Prespecified single-seed ablations (exploratory)');fig.tight_layout();fig.savefig(OUT/'ablations.png',dpi=180);plt.close(fig)
    sm=pd.DataFrame(json.loads((OUT/'sensitivity_metrics.json').read_text()));sm.to_csv(OUT/'sensitivity_metrics.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(12,7.2));axes=axes.ravel()
    for ax,field,label in zip(axes,['reliability','rejection_cost','signal_effect','hard_cost','query_cost'],['Question reliability','Public rejection cost','Signal effect','Hard-type acceptance cost','Question cost']):
        for name,color in [('Planner','#27646b'),('Planner_no_query','#888888'),('PPO','#bb7847')]:
            g=sm[(sm.field==field)&(sm.policy==name)].groupby('value').net_utility.mean();ax.plot(g.index,g.values,'o-',label=name,color=color)
        ax.set(xlabel=label,ylabel='Net utility',title='Paired evaluation-only change')
    axes[4].legend(fontsize=8);axes[5].axis('off');fig.tight_layout();fig.savefig(OUT/'sensitivity.png',dpi=180);plt.close(fig)
    # Actual loss, entropy and returns from saved training records.
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for seed in (17,29,43):
        curve=json.loads((OUT/f'curves/base_{seed}.json').read_text());c=pd.DataFrame(curve)
        axes[0].plot(c.step,c['return'].rolling(200,min_periods=1).mean(),label=str(seed))
        log=pd.read_csv(OUT/f'training_logs/base_{seed}/progress.csv')
        axes[1].plot(log['time/total_timesteps'],log['train/value_loss'],label=str(seed))
        axes[2].plot(log['time/total_timesteps'],-log['train/entropy_loss'],label=str(seed))
    for ax,title in zip(axes,['Return (200-episode rolling mean)','Value loss','Policy entropy']):ax.set(title=title,xlabel='Training steps');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,name in zip(axes,['test_Heuristic_None','test_PPO_base_17']):
        m=matrices[name]['probabilities'];p=np.array([r if r is not None else [np.nan]*11 for r in m]);ax.imshow(p,cmap='Blues',vmin=0,vmax=1)
        ax.set(xticks=range(11),yticks=range(11),xticklabels=NAMES,yticklabels=NAMES,title=name.replace('test_',''));ax.tick_params(axis='x',rotation=90,labelsize=8);ax.tick_params(axis='y',labelsize=8)
    fig.suptitle('Within-episode action transitions; white = unobserved');fig.tight_layout();fig.savefig(OUT/'action_transitions.png',dpi=180);plt.close(fig)
    # Readable case walkthroughs, with a clear source/synthetic distinction.
    sections=[];contexts={c['id']:c for c in json.loads((OUT/'contexts.json').read_text(encoding='utf-8'))}
    for i,r in enumerate(cases,1):
        source=contexts[r['context']]['source']
        lines=[f"Case {i}: {r['category']} | {r['policy']} | context {r['context']} | synthetic condition {r['condition']}",
               f"Source: {source['doc_title']} ({source['date']}); candidate CSV row {source['csv_row']}",
               f"Source reference: {source['url_or_doi']}",f"Real source anchor: {source['sentence']}",
               f"Outcome {r['outcome']}; net utility {r['net_utility']:.4f}; agreement {r['agreement']}; source context is not an observed trajectory."]
        for t in r['trajectory']:
            lines+=[f"Turn {t['before']['t']+1}: {t['utterance']}",f"  Meaning: {t['meaning']}; response: {t['opponent_utterance']}",f"  Belief {t['before']['b']:.3f} -> {t['after']['b']:.3f}; reward {t['common_reward']:.4f}; costs {t['costs']}"]
        sections.append('\n'.join(lines))
    (OUT/'discourse_walkthroughs.txt').write_text('\n\n'.join(sections),encoding='utf-8')
    lang=json.loads((OUT/'language_summary.json').read_text());ledger=json.loads((ROOT/'budget/pilot_20261012.json').read_text())
    cost=sum(r.get('cost',0) for r in ledger['records']);unknown=sum(r['reserved'] for r in ledger['records'] if 'cost' not in r)
    table=pd.DataFrame(summary)[['policy','runs','net_utility','agreement','cost','queries','length']].round(4).to_html(index=False)
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>GPT-5.6 Luna experiment results</title><style>body{max-width:1120px;margin:32px auto;padding:0 24px;font:16px/1.6 system-ui;color:#18313b}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 10px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;font:13px/1.6 system-ui;background:#f1f5f6;padding:18px}.note{padding:16px;background:#edf4f5;border-left:4px solid #40818a}</style><h1>GPT-5.6 Luna: main experiment results</h1><p class="note">This report records the initial experiment stage. The current manuscript is in ../paper/main.pdf. The synthetic diplomatic negotiation environment is grounded in 12 real documents from six event groups: six training, two development, and four test documents. The test split contains only two historical event groups. Its 256 numerical cases are not independent historical negotiations.</p><h2>Policy comparison</h2><img src="main_results.png">'''+table+'''<h2>Direct Luna action selection: matched 64 conditions and utterance style</h2>'''+ss.round(4).to_html(index=False)+'''<p>Direct Luna receives the public rules in its prompt. PPO is a separate action selector trained with the same states and rewards. The experiment does not fine-tune Luna weights.</p><h2>Required ablations and information/rule sensitivity</h2><img src="ablations.png"><img src="sensitivity.png"><img src="learning_curves.png"><h2>Language checks</h2><pre>'''+html.escape(json.dumps(lang,ensure_ascii=False,indent=2))+'''</pre><p>Mismatches were retained in transitions without correction or replacement by intended actions. These checks do not measure accuracy against human labels. In the condition-first style, some immediate withdrawals were interpreted as conditional withdrawals.</p><h2>Discourse trajectories</h2><img src="action_transitions.png">'''+''.join('<pre>'+html.escape(s)+'</pre>' for s in sections)+f'''<h2>Reproduction, cost, and review status</h2><p>At this experiment stage, cumulative additional model-response cost was ${cost:.6f}, with ${unknown:.6f} reserved for unresolved requests. Raw logs, models, run settings, and requests/responses are retained; the private cost ledger is excluded from the release. The current manuscript contains six discourse cases, and the later Sol replication is available in ../sol_results/. Independent human validation is not claimed. Actual teammate reproduction is recorded separately in ../submission/TEAM_RECORD.txt.</p></html>'''
    main_by={r['policy']:r for r in summary};ppo=main_by['PPO_base'];planner=main_by['Planner'];nq=main_by['Planner_no_query']
    diagnostics=json.loads((OUT/'diagnostics_by_group_style.json').read_text())
    direct_diag=[r for r in diagnostics if r['policy']=='Direct_Luna']
    repeated=float(np.mean([r['repeated_rejected_offer'] for r in direct_diag]))
    observed=f'''<h2>Observed results</h2><p>The experiment completed 14 training runs and 1,433,600 environment interactions. Across three seeds, PPO achieves mean net utility {ppo['net_utility']:.4f} (SD {ppo['net_utility_sd']:.4f}) and agreement {ppo['agreement']:.1%}. It underperforms the strong heuristic, and base PPO uses neither queries nor signals. For the planner with access to public rules and the utterance bank, allowing queries improves utility by {planner['net_utility']-nq['net_utility']:.5f}. The planner is a benchmark with known model dynamics, not an equal-resource comparison of learning efficiency.</p><p>On the matched 64 conditions and style, direct Luna achieves net utility {float(ss.loc[ss.policy=='Direct_Luna','net_utility'].iloc[0]):.4f}, compared with {float(ss.loc[ss.policy=='PPO_base','net_utility'].mean()):.4f} for the three-seed PPO mean. In probability-weighted direct Luna trajectories, {repeated:.1%} repeat the immediately preceding rejected offer. Results use one prompt and one cached response per state; they do not characterize general Luna capabilities or performance under alternative prompts.</p><p>Error bars represent standard deviations across training seeds or Random evaluation RNG seeds, not confidence intervals over historical events. Auxiliary ablations use seed 17 and should be compared with its corresponding base policy. Sensitivity tests change one public rule at evaluation time after training. Results concern fixed synthetic utilities and source-grounded hypothetical scenarios; real diplomatic outcomes were not validated.</p>'''
    text=text.replace('<h2>Policy comparison</h2>',observed+'<h2>Policy comparison</h2>',1)
    (OUT/'report.html').write_text(text,encoding='utf-8')
    write(OUT/'finalization.json',{'time':time.time(),'paper_written':False,'models_expanded':False,'api_cost_so_far':cost,'unresolved':unknown,'discourse_cases':len(cases)})
    print('RESULTS READY',json.dumps({'main':[(r['policy'],round(r['net_utility'],4),round(r['agreement'],4)) for r in summary if r['policy'] in chosen],'luna_matched':ss.to_dict('records'),'cases':len(cases),'cost':cost},allow_nan=True),flush=True)

if __name__=='__main__':main()
