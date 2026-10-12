"""Freeze groups and protocol before generating/training/evaluating."""
import csv,json,hashlib,time
import numpy as np
from dataclasses import asdict,replace
from study import ROOT,OUT
from pilot.env import Config
from pilot.live import write

def main():
    if (OUT/'protocol.json').exists(): raise RuntimeError('Protocol already frozen')
    rows=list(csv.DictReader((ROOT/'data/source/sentences/candidates.csv').open(encoding='utf-8-sig')))
    selectors=[
      ('kr1','peninsula','train','14733','in exchange','suspend large-scale military exercises','suspend nuclear and missile tests'),
      ('kr2','peninsula','train','16629','reciprocal','suspend large-scale military exercises','suspend nuclear and missile tests'),
      ('ss1','south_sudan','train','10890','signing','maintain a cessation of hostilities','maintain a cessation of hostilities'),
      ('ss2','south_sudan','train','13454','ceasefire','halt offensive military operations','halt offensive military operations'),
      ('ss3','south_sudan','train','9467','demilitarized','keep military forces outside the demilitarized zone','keep police forces outside the demilitarized zone'),
      ('ip1','kashmir','train','10527','ceasefire','halt firing along the line of control','halt firing along the line of control'),
      ('ga1','gaza','dev','Global Partners','at least six weeks','maintain a ceasefire','maintain humanitarian access'),
      ('ga2','gaza','dev','March 25 2024','linked to hostages','maintain a ceasefire','maintain humanitarian access'),
      ('sy1','syria','test','12850','seriously implement','maintain a cessation of hostilities','maintain a cessation of hostilities'),
      ('sy2','syria','test','8727','In our talks, Turkey','keep forces outside disputed positions','keep forces outside agreed observation areas'),
      ('ir1','iran','test','2020/644','reversible','suspend the specified sanctions','suspend the specified nuclear activities'),
      ('ir2','iran','test','10285','This round of talks offers','suspend the specified sanctions','suspend the specified nuclear activities')]
    contexts=[]
    for cid,group,split,title,phrase,ours,theirs in selectors:
        matches=[(i,r) for i,r in enumerate(rows) if title in r['doc_title'] and phrase in r['sentence']]
        assert len(matches)==1,(cid,len(matches))
        i,r=matches[0]
        contexts.append(dict(id=cid,event_group=group,split=split,source=dict(r,csv_row=i+2),our_measure=ours,their_measure=theirs,
          interpretation='Researcher-authored bilateral hypothetical inspired by this public discourse; durations, reciprocity, types and payoffs are synthetic.',
          prior_exposure='Some sources were inspected in pilot/planning. Held out from training, not claimed researcher-blind or historically untouched.'))
    write(OUT/'contexts.json',contexts)
    # Exact normalized text/document/event isolation, before any transformation.
    for field in ('event_group',):
        groups={s:{x[field] for x in contexts if x['split']==s} for s in ('train','dev','test')}
        assert not(groups['train']&groups['dev'] or groups['train']&groups['test'] or groups['dev']&groups['test'])
    rng=np.random.default_rng(20261012)
    configs={}
    for split,n in [('train',96),('dev',32),('test',64)]:
        values=[]
        for i in range(n):
            c=Config(prior=float(rng.choice([.2,.35,.5,.65,.8])),reliability=float(rng.choice([.6,.75,.9,1.])),
               query_cost=float(rng.choice([.01,.02,.04,.08])),rejection_cost=float(rng.choice([0.,.04,.08,.16,.32])),
               signal_effect=float(rng.choice([0.,.1,.25])),hard_cost=float(rng.choice([.4,.45,.55,.9])))
            values.append(asdict(c))
        configs[split]=values
    write(OUT/'numeric_conditions.json',configs)
    protocol={'version':'luna-main-v1','frozen_at':time.time(),'model':'openai/gpt-5.6-luna','concurrency':4,
      'language_variants_per_context_action':3,'generation_calls':12*11*3,'interpretation_calls':12*11*3,
      'parameterized_eval_cases':{'dev':2*32,'test':4*64},'historical_event_groups':6,
      'seeds':[17,29,43],'steps':102400,'evaluation_random_seeds':[101,202,303],'random_repetitions_per_cell':4,
      'base_ppo':{'gamma':.95,'ent_coef':.01,'learning_rate':.0003,'n_steps':256,'batch_size':64,'n_epochs':10,'net_arch':[64,64]},
      'runs':[{'name':'base','seeds':[17,29,43]},{'name':'gamma08','seeds':[17],'gamma':.8},{'name':'gamma099','seeds':[17],'gamma':.99},
              {'name':'entropy0','seeds':[17],'ent_coef':0.},{'name':'entropy01','seeds':[17],'ent_coef':.1},
              {'name':'no_rejection_penalty','seeds':[17],'remove_reward':'rejection'},
              {'name':'no_reversal_penalty','seeds':[17],'remove_reward':'reversal'},
              {'name':'belief_mask','seeds':[17],'mask':'belief'},
              {'name':'query_disabled','seeds':[17],'disable_query':True}],
      'heuristic_selection':'Select one fixed heuristic by mean dev net utility before opening test aggregate results.',
      'ppo_selection':'Final fixed-budget checkpoint for every seed; no test-based selection.',
      'execution':'All policies share the frozen Luna-generated, Luna-interpreted utterance bank. Interpretation errors are charged and preserved, never replaced with intended labels.',
      'oracle':'Exact public-belief model planner is an upper benchmark only for canonical language; not presented as an exact optimum for a fallible bank.',
      'hypotheses':['Question value depends on rejection cost and reliability after accounting for information from offers.',
                    'High deal rates can coexist with unused information and avoidable utility loss.',
                    'Faithful utterance realization is separate from selecting a useful strategy.'],
      'statistics':'Aggregate response branches and language variants within synthetic cases, then contexts within event groups. Three-seed SD measures training variability; only two held-out event groups, so no historical population CI.',
      'human_review':'Agent/model annotation is not human gold; prepare a blinded review sheet. Teammate reproduction remains a human task.',
      'stop_rule':'Complete fixed run matrix and report all failures; no tuning after test evaluation.'}
    write(OUT/'protocol.json',protocol)
    print(json.dumps({'contexts':len(contexts),'groups':6,'generation_calls':396,'training_runs':11,'heldout_contexts':4}),flush=True)

if __name__=='__main__': main()
