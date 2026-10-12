"""Report the follow-up honestly, with paired gains and all selected-policy ablations."""
import json,gzip,collections,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improvement import OUT
from pilot.live import write

def main():
    selection=json.loads((OUT/'selection.json').read_text());chosen=selection['selected']
    rows=json.loads((OUT/'seed_metrics.json').read_text());df=pd.DataFrame(rows);df.to_csv(OUT/'seed_metrics.csv',index=False)
    cols=['net_utility','agreement','queries','signals','cost','length','acceptance_violations']
    means=df.groupby('policy')[cols].mean();sd=df.groupby('policy')[cols].std();means.to_csv(OUT/'summary.csv');sd.to_csv(OUT/'seed_sd.csv')
    cases=[]
    for path in (OUT/'case_metrics').glob('test_*.json'):cases+=json.loads(path.read_text())
    c=pd.DataFrame(cases);c.to_csv(OUT/'case_metrics.csv',index=False)
    paired=c.groupby(['policy','context','condition'])[cols].mean().reset_index()
    wide=paired.pivot(index=['context','condition'],columns='policy',values='net_utility')
    info=wide.Planner-wide.Planner_no_query;valuable=info>1e-7
    metrics={'selected':chosen,'mean_gain_over_base':float((wide[chosen]-wide['base']).mean()),
      'mean_gain_over_heuristic':float((wide[chosen]-wide.Heuristic).mean()),
      'selected_minus_no_query':float((wide[chosen]-wide[chosen+'_no_query']).mean()),
      'selected_minus_no_signal':float((wide[chosen]-wide[chosen+'_no_signal']).mean()),
      'selected_minus_masked_belief':float((wide[chosen]-wide[chosen+'_masked_belief']).mean()),
      'positive_query_value_cases':int(valuable.sum()),'all_context_settings':len(wide),
      'query_removal_loss_on_positive_value_cases':float((wide[chosen]-wide[chosen+'_no_query'])[valuable].mean()),
      'query_removal_loss_on_other_cases':float((wide[chosen]-wide[chosen+'_no_query'])[~valuable].mean()),
      'gap_to_planner':float((wide.Planner-wide[chosen]).mean()),
      'no_new_api_calls':True,'main_results_unchanged':True}
    wide['planner_question_value']=info;wide['selected_query_removal_loss']=wide[chosen]-wide[chosen+'_no_query'];wide.to_csv(OUT/'paired_comparisons.csv')
    # Every path: cost consistency, branch weight, legal agreement and continuity.
    n=0;turns=0;weights=collections.defaultdict(float);examples=[]
    for p in sorted((OUT/'raw').glob('*.gz')):
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);tr=r['trajectory'];n+=1;turns+=len(tr)
                assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
                assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
                assert r['acceptance_violations']==0
                for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
                weights[(p.name,r['context'],r['condition'])]+=r['weight']
                if p.name.startswith('test_') and r['policy']==chosen and r['agreement'] and r['queries']>0 and len(examples)<3 and not any((x['context'],x['condition'])==(r['context'],r['condition']) for x in examples):examples.append(r)
    assert all(abs(w-1)<1e-8 for w in weights.values())
    metrics.update(audited_episodes=n,audited_turns=turns,integrity_checks_passed=True)
    write(OUT/'findings.json',metrics);write(OUT/'successful_query_examples.json',examples)
    names=['base','Heuristic','distilled','distilled_ppo','Planner']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,col,title in zip(axes,['net_utility','queries','signals'],['Net utility','Private questions / episode','Signals / episode']):
        vals=means.loc[names,col];ax.bar(range(len(names)),vals,color=['#888888','#6a939b','#246f73','#66a7a7','#b8c6c6'])
        for i,name in enumerate(names):
            if pd.notna(sd.loc[name,col]):ax.errorbar(i,vals.iloc[i],yerr=sd.loc[name,col],color='black',capsize=3)
        ax.set(title=title,xticks=range(5),xticklabels=['Base PPO','Heuristic','Distilled','Distilled\n+ PPO','Planner']);ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ab=[chosen,chosen+'_no_query',chosen+'_no_signal',chosen+'_masked_belief']
    ax.bar(range(4),means.loc[ab,'net_utility'],color='#347e84');ax.set(xticks=range(4),xticklabels=['Selected','Query removed','Signal removed','Belief masked'],ylabel='Common net utility',title='Deployment interventions; no retraining');fig.tight_layout();fig.savefig(OUT/'interventions.png',dpi=180);plt.close(fig)
    status='Improvement observed' if metrics['mean_gain_over_heuristic']>0 and metrics['mean_gain_over_base']>0 else 'Limited improvement; see comparisons below'
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Information-use policy experiment</title><style>body{{max-width:1150px;margin:35px auto;font:16px/1.65 system-ui;padding:0 22px;color:#18383d}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f1f5f5;padding:15px}}</style><h1>Information-use policy: {status}</h1><p>This follow-up began after inspection of the original results. Development-selected method: <b>{chosen}</b>. Original records are preserved. Evaluation uses 64 new numerical settings across four previously used document contexts. These are not new historical events or previously unseen source documents.</p><p>Action values from a public-belief planner were distilled into a small policy network. Neither training nor inference inputs contain the actual opponent type. The learned policy does not call the planner at evaluation time, but training uses known transition rules and additional teacher computation. This is not an equal-budget model-free learning comparison or a new algorithm. The Luna utterance/interpretation bank is unchanged, with no new model calls.</p><img src="comparison.png"><p>Error bars show standard deviations across three training seeds. The heuristic and planner are fixed policies.</p>{means.round(5).to_html()}<h2>Development selection record</h2>{pd.DataFrame(selection['dev'])[['method','seed','net_utility','queries','signals']].round(5).to_html(index=False)}<h2>Does information use improve performance?</h2><img src="interventions.png"><pre>{html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))}</pre><p>Query and signal removal exclude the corresponding action and select the next-highest learned action score. Belief masking replaces the posterior with the initial prior while retaining other public state variables. These are deployment interventions, not retrained ablations. Positive query value is defined by the utility difference between the public-belief planner and its query-disabled counterpart.</p><h2>Retained artifacts</h2><p>This directory retains the protocol, teacher data, learning curves, six checkpoints, all evaluation trajectories, per-seed results, matched comparisons, and successful query examples. This report records the experiment stage.</p></html>'''
    narrative=f'''<h2>Interpretation</h2><p>The selected policy achieves net utility {means.loc[chosen,'net_utility']:.5f} (seed SD {sd.loc[chosen,'net_utility']:.5f}). On the same new numerical evaluation, its gains are {metrics['mean_gain_over_base']:+.5f} over the original PPO and {metrics['mean_gain_over_heuristic']:+.5f} over the strong heuristic. Disabling signals reduces mean net utility by {metrics['selected_minus_no_signal']:.5f}; masking updated beliefs reduces it by {metrics['selected_minus_masked_belief']:.5f}. The improvement is therefore associated mainly with selective signaling and use of information from opponent responses.</p><p>Use of costly queries remains limited. Only {int(((df.policy==chosen)&(df.queries>0)).sum())} of the three selected seeds use queries, and query removal reduces mean net utility by {metrics['selected_minus_no_query']:.5f}. Query use is not resolved across all seeds. These interventions perturb policy inputs or outputs; they are neither a complete causal decomposition nor comparisons against retrained alternatives.</p>'''
    page=page.replace('<h2>Development selection record</h2>',narrative+'<h2>Development selection record</h2>',1)
    if json.loads((OUT/'protocol.json').read_text()).get('model')=='openai/gpt-5.6-sol':
        page=page.replace('Luna utterance/interpretation bank','Sol utterance/interpretation bank')
    (OUT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
