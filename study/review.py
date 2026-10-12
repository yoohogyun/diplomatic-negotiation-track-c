"""Prepare evidence review, boundary checks and a blinded human-review sheet."""
import json,csv,hashlib
from dataclasses import asdict
from study import ROOT,OUT
from study.env import LunaEnv
from study.evaluate import planner_for,encoded_bank,run_episode
from pilot.env import Config
from pilot.live import write

def main():
    contexts=json.loads((OUT/'contexts.json').read_text(encoding='utf-8'))
    bank=json.loads((OUT/'language_bank.json').read_text(encoding='utf-8'))
    validations=[]
    for c in contexts:
        e=LunaEnv(c['split'])
        for hard in (False,True):
            z=run_episode(e,lambda o,e:planner_for(e.cfg,encoded_bank(e))(e.s)[1],Config(),c,0,hard,0.,123)
            reward_sum=sum(i['common_reward'] for i in z['trajectory'])
            cost_sum=sum(sum(i['costs'].values()) for i in z['trajectory'])
            validations.append({'context':c['id'],'source_row':c['source']['csv_row'],'hard_evaluation_only':hard,
                   'length':z['length'],'outcome':z['outcome'],'reward_sum_correct':abs(reward_sum-(z['own_utility']-cost_sum))<1e-10,
                   'acceptance_violations':z['acceptance_violations'],'text_path':z['trajectory']})
    write(OUT/'source_case_validation.json',validations)
    mismatches=[r for r in bank if not r['exact_meaning_match']]
    reviews=[]
    for z in mismatches:
        intended=z['intended']['kind']; actual=z['meaning']['kind']
        if intended=='withdraw' and 'if ' in z['utterance'].lower():
            note='Generation changed an unconditional exit into a conditional statement; invalid interpretation is plausible. Fronted-condition style can conflict with preserving the requested speech act.'
        elif intended=='offer':
            note='Inspect for interpreter false rejection versus changed terms/actor roles. Exact roundtrip mismatch alone cannot attribute the error to generator or interpreter.'
        else:note='Requires semantic review; no claim of a gold label.'
        reviews.append({'id':z['id'],'utterance':z['utterance'],'intended':z['intended'],'interpreted':z['meaning'],
                        'agent_review':note,'review_status':'Codex analysis; not independent human annotation'})
    write(OUT/'language_error_review.json',reviews)
    # Three sources per event group is impossible for sparse groups; explicitly
    # use all 12 source anchors + all mismatches + 33 deterministic controls.
    sample=mismatches+[z for z in bank if z['exact_meaning_match']][:33]
    sheet=[]; gold=[]
    for i,z in enumerate(sample):
        rid=f'U{i+1:03d}'; sheet.append({'review_id':rid,'type':'synthetic_utterance','text':z['utterance'],
            'human_function':'','our_weeks':'','their_weeks':'','conditionality':'','unsupported_detail':'','notes':''})
        gold.append({'review_id':rid,'bank_id':z['id'],'intended':z['intended'],'model_interpretation':z['meaning']})
    for c in contexts:
        sheet.append({'review_id':'S_'+c['id'],'type':'real_source','text':c['source']['sentence'],
            'human_function':'','our_weeks':'','their_weeks':'','conditionality':'','unsupported_detail':'','notes':''})
    with (OUT/'human_review_blank.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(sheet[0]));w.writeheader();w.writerows(sheet)
    write(OUT/'human_review_key.json',gold)
    write(OUT/'review_status.json',{'source_anchors':12,'environment_validations':len(validations),'blinded_items':len(sheet),
           'completed_human_reviews':0,'all_case_checks_passed':all(x['reward_sum_correct'] and x['acceptance_violations']==0 and x['length']<=5 for x in validations),
           'notes':'Selection enriches errors and is not a prevalence sample. No human inter-annotator agreement claim.'})
    print('Source and transition checks complete',len(validations),'human review sheet',len(sheet))

if __name__=='__main__':main()
