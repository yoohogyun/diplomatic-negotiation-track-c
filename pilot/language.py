"""Small real-source audit and fresh Luna generation/interpretation pilot.

Model annotations are proposals for review, never human ground truth. All
generated dialogue is explicitly synthetic. No source text supplies utilities.
"""
import csv,json,hashlib
from pilot import ROOT
from pilot.env import *
from pilot.live import call,write,DIR,LEDGER

def schema(fields):
    return {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
STR={'type':'string'}
NUM={'type':['number','null']}
SEM=schema({'kind':{'type':'string','enum':['offer','query','signal','ambiguity','ratify','withdraw','invalid']},'x':NUM,'y':NUM,'cost':NUM,'conditional':{'type':'string','enum':['yes','no','not_applicable']}})

def interpret(tag,text):
    return call(tag,[{'role':'system','content':'Interpret only the supplied synthetic diplomatic utterance. x=our military exercise suspension duration in weeks, y=their nuclear/missile test suspension in weeks; cost=explicit signal resource cost. Use null for unspecified numbers. An offer is valid only when our suspension is conditional on their suspension. Negated commitments, unless, missing reciprocity, or numbers outside x=1,2,3 and y=1,2 must be invalid. Do not repair or infer missing terms. Signal is explicit resource expenditure; query asks privately about acceptance; ambiguity withholds commitment; ratify finalizes an accepted proposal; withdraw exits talks.'},
                     {'role':'user','content':text}],SEM)

def to_meaning(z):
    k=z['kind']
    if k=='offer':
        if z['conditional']!='yes' or z['x'] not in (1,2,3) or z['y'] not in (1,2): return {'kind':'invalid'}
        return {'kind':'offer','x':int(z['x']),'y':int(z['y'])}
    if k=='signal':
        if z['cost'] is None or not 0<=z['cost']<=1: return {'kind':'invalid'}
        return {'kind':'signal','cost':z['cost']}
    return {'kind':k}

def main():
    rows=[]
    for path in ('all_250.csv','condition_supplement.csv'):
        rows+=list(csv.DictReader((ROOT/'data/source/sentences'/path).open(encoding='utf-8-sig')))
    ids=['C-012','S1-019','C-006','C-007','C-011','C-023','S3-043','S1-025','S3-009','S4-018','S4-040','S5-015']
    chosen=[next(r for r in rows if r['id']==i) for i in ids]
    candidates=list(csv.DictReader((ROOT/'data/source/sentences/candidates.csv').open(encoding='utf-8-sig')))
    for idx in (1186,1773,1774,1785,1793,1873):
        chosen.append(dict(candidates[idx],id=f'candidate_row_{idx+2}'))
    write(ROOT/'pilot_results/source_selection.json',{'status':'purposeful exploratory selection; not independent crises or held-out test',
             'source_sentences':chosen,'unique_document_keys':len({r['url_or_doi'] for r in chosen}),
             'main_simulator_context_id':'C-012','warning':'Remaining sources support discourse functions only; not additional historical realizations of this simulator.'})
    audits=[]
    audit_schema=schema({'function':{'type':'string','enum':['conditional_exchange','condition','commitment','noncommitment','private_channel_mention','background','other']},'evidence':STR,'condition':STR,'limitation':STR})
    for r in chosen:
        z=call('audit_'+r['id'],[{'role':'system','content':'Audit a real diplomatic source sentence. Return a cautious primary discourse function, an exact short evidence substring, explicit condition or NONE, and what cannot be inferred from this sentence. Do not infer hidden motives, utilities, actual negotiation outcomes or secret dialogue. A reporter-facing refusal to discuss private conversations is not itself private dialogue. Noncommitment is not proof of strategic intent. Keep each field short.'},
              {'role':'user','content':json.dumps({k:r[k] for k in ('doc_title','date','sentence')},ensure_ascii=False)}],audit_schema)
        audits.append({'source_id':r['id'],'annotation':z,'evidence_is_exact_substring':z['evidence'] in r['sentence'],'label_status':'model proposed; no human gold'})
    write(ROOT/'pilot_results/source_audit.json',audits)
    print('Source audit completed',len(audits),flush=True)
    language=[]
    for variant in range(2):
        for a in range(11):
            canonical=render(a); expected=parse(canonical)
            source=chosen[0] if expected['kind']=='offer' else chosen[(a+variant)%len(chosen)]
            prompt='Write ONE synthetic diplomatic sentence expressing exactly the supplied meaning. Preserve numbers, reciprocal if-condition, private/public status, resource cost and degree of commitment. Do not add facts or promises. Use generic parties. Source is inspiration for diplomatic register only, not a quotation or historical event. '+('Use a formal communique register.' if variant==0 else 'Use direct language in a negotiation room.')
            generated=call(f'gen_{variant}_{a}',[{'role':'system','content':prompt},{'role':'user','content':json.dumps({'meaning_sentence':canonical,'source_sentence':source['sentence']},ensure_ascii=False)}],schema({'utterance':STR}))['utterance']
            interpreted=interpret(f'interpret_{variant}_{a}',generated); actual=to_meaning(interpreted)
            language.append({'id':f'{variant}_{a}','source_id':source['id'],'canonical':canonical,'generated':generated,'expected':expected,'interpreted':interpreted,'meaning':actual,'match':expected==actual})
    write(ROOT/'pilot_results/language_roundtrip.json',language)
    # Counterfactuals are authored synthetic diagnostic pairs; tests are not natural-source labels.
    probes=[]
    for a in (4,6,8):
        original=render(a)
        for mode,text in [('original',original),('negated',original.replace('will suspend','will not suspend')),
                          ('unless',original.replace(' if ',' unless ')),('unconditional',original.split(' if ')[0]+'.'),
                          ('changed_number',original.replace('for 2 weeks.','for 1 weeks.'))]:
            expected=parse(text) if mode in ('original','changed_number') else {'kind':'invalid'}
            z=interpret(f'probe_{a}_{mode}',text); actual=to_meaning(z)
            probes.append({'action':a,'mode':mode,'text':text,'expected':expected,'actual':actual,'match':actual==expected})
    write(ROOT/'pilot_results/semantic_probes.json',probes)
    ledger=json.loads(LEDGER.read_text()); print(json.dumps({'roundtrip_matches':sum(z['match'] for z in language),'roundtrip_n':len(language),
              'probe_matches':sum(z['match'] for z in probes),'probe_n':len(probes),'billed_cost':sum(r.get('cost',0) for r in ledger['records']),
              'reserved_unresolved':sum(r['reserved'] for r in ledger['records'] if 'cost' not in r)}),flush=True)

if __name__=='__main__': main()
