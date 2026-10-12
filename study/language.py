import json
from study import ROOT,OUT
from study.api import call,batch
from pilot.live import write
from pilot.language import schema,STR,NUM
from pilot.env import NAMES

SEM=schema({'kind':{'type':'string','enum':['offer','query','signal','ambiguity','ratify','withdraw','invalid']},
            'x':NUM,'y':NUM,'cost':NUM,'conditional':{'type':'string','enum':['yes','no','not_applicable']}})

def intended(a):
    if 3<=a<=8:
        x,y=map(int,NAMES[a].split('_')[1:]); return {'kind':'offer','x':x,'y':y}
    if a==1: return {'kind':'signal','cost':.06}
    return {'kind':NAMES[a]}

def text_request(c,a):
    m=intended(a); k=m['kind']
    if k=='offer': return f'We will {c["our_measure"]} for {m["x"]} weeks IF you {c["their_measure"]} for {m["y"]} weeks. Our commitment depends on yours.'
    if k=='signal': return 'We now commit resources at a cost of 0.06 utility units to a monitored confidence-building measure.'
    if k=='query': return 'Privately ask whether the counterpart requires a larger reciprocal concession to accept. Make no offer or public commitment.'
    if k=='ambiguity': return 'State that our next proposal is provisional and makes no new binding public commitment. Do not erase earlier commitments.'
    if k=='ratify': return 'Ratify the currently accepted agreement, subject to the recorded bilateral conditions. Do not invent terms.'
    return 'Withdraw from these negotiations without proposing or ratifying an agreement.'

def interpret(tag,c,text):
    return call(tag,[{'role':'system','content':
       'Interpret the supplied synthetic diplomatic utterance, not its presumed intent. x=weeks of OUR measure, y=weeks of THEIR reciprocal measure, cost=explicit resource expenditure. Use null for unstated numbers. Classify offer only if our measure is explicitly conditional on theirs and x in {1,2,3}, y in {1,2}. Negated offers, unless-conditions, unilateral promises, missing reciprocity and out-of-range terms are invalid; do not repair them. Query seeks conditions privately. Signal commits resources at a stated cost. Ambiguity makes the next proposal provisional. Ratify finalizes an accepted agreement. Withdraw exits talks. Return only the schema.'},
       {'role':'user','content':json.dumps({'our_measure':c['our_measure'],'their_measure':c['their_measure'],'utterance':text},ensure_ascii=False)}],SEM)

def meaning(z):
    if z['kind']=='offer':
        if z['conditional']!='yes' or z['x'] not in (1,2,3) or z['y'] not in (1,2): return {'kind':'invalid'}
        return {'kind':'offer','x':int(z['x']),'y':int(z['y'])}
    if z['kind']=='signal':
        if z['cost'] is None or not 0<=z['cost']<=1: return {'kind':'invalid'}
        return {'kind':'signal','cost':z['cost']}
    return {'kind':z['kind']}

def generate(job):
    c,a,v=job
    style=['formal diplomatic communique','direct private negotiating-room statement','concise statement with a fronted condition or reordered clauses'][v]
    tag=f'{c["id"]}_{a}_{v}'
    z=call('gen_'+tag,[{'role':'system','content':
        'Write one SYNTHETIC diplomatic utterance, not a quotation or reconstruction. Preserve ALL required conditions, numbers, actor roles, channel and commitment strength. Use generic we/you, not actual country names. A historical source only supplies discourse context. Do not copy unsupported historical details. Return an utterance under 65 words in a '+style+' style.'},
        {'role':'user','content':json.dumps({'source':c['source']['sentence'],'synthetic_required_meaning':text_request(c,a)},ensure_ascii=False)}],schema({'utterance':STR}))
    text=z['utterance']; parsed=interpret('parse_'+tag,c,text); actual=meaning(parsed)
    return {'id':tag,'context_id':c['id'],'split':c['split'],'action':a,'variant':v,'utterance':text,
            'parsed':parsed,'meaning':actual,'intended':intended(a),'exact_meaning_match':actual==intended(a)}

def probe(job):
    c,a,mode=job; base=text_request(c,a).split(' Our commitment')[0]
    if mode=='valid': text=base; expected=intended(a)
    elif mode=='number':
        target=2 if intended(a)['y']==1 else 1
        text=base.rsplit('for ',1)[0]+f'for {target} weeks.'; expected=dict(intended(a),y=target)
    elif mode=='negated': text=base.replace('We will ','We will not ',1); expected={'kind':'invalid'}
    elif mode=='unless': text=base.replace(' IF ',' UNLESS '); expected={'kind':'invalid'}
    else: text=base.split(' IF ')[0]+'.'; expected={'kind':'invalid'}
    z=interpret(f'probe_{c["id"]}_{a}_{mode}',c,text); actual=meaning(z)
    return {'context':c['id'],'split':c['split'],'action':a,'mode':mode,'text':text,'expected':expected,'actual':actual,'match':actual==expected}

def main():
    contexts=json.loads((OUT/'contexts.json').read_text(encoding='utf-8'))
    jobs=[(c,a,v) for c in contexts for a in range(11) for v in range(3)]
    bank=batch(jobs,generate,4); bank.sort(key=lambda z:z['id']); write(OUT/'language_bank.json',bank)
    print('BANK READY',len(bank),'matches',sum(x['exact_meaning_match'] for x in bank),flush=True)
    jobs=[(c,a,mode) for c in contexts for a in (4,6,8) for mode in ('valid','number','negated','unless','unconditional')]
    probes=batch(jobs,probe,4); write(OUT/'semantic_probes.json',probes)
    write(OUT/'language_summary.json',{'bank_n':len(bank),'bank_match':sum(x['exact_meaning_match'] for x in bank),
             'probes_n':len(probes),'probes_match':sum(x['match'] for x in probes),
             'note':'Same-model generation/interpretation consistency, not human gold accuracy.'})
    print('LANGUAGE DONE',flush=True)

if __name__=='__main__': main()
