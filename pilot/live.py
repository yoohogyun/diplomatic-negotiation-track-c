"""Serial, budget-reserved OpenRouter transport. Credentials never logged."""
import json, os, time, hashlib
from pathlib import Path
import requests
from pilot import ROOT
from infra.openrouter import build_request,check_metadata,validate_response,save_new

DIR=ROOT/'pilot_results'/'api'
LEDGER=ROOT/'budget'/'pilot_20261012.json'

def key():
    if os.environ.get('OR_KEY'): return os.environ['OR_KEY']
    for p in (ROOT/'.env',ROOT.parent/'.env'):
        if p.exists():
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                k,sep,v=line.strip().partition('=')
                if k=='OR_KEY' and sep: return v.strip().strip('\"').strip("'")
    raise RuntimeError('OR_KEY missing')

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); q.replace(p)

def preflight():
    DIR.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/luna_standard.json').read_text())
    meta=requests.get(cfg['metadata_url'],timeout=30); meta.raise_for_status(); meta=meta.json()
    entry=check_metadata(meta,build_request(cfg,[{'role':'user','content':'Return a diplomatic sentence.'}],0))
    write(DIR/'metadata_live.json',meta)
    r=requests.get('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+key()},timeout=30)
    r.raise_for_status(); d=r.json()['data']
    usage={k:d.get(k) for k in ('usage','usage_daily','limit','limit_remaining','is_free_tier')}
    if not LEDGER.exists():
        write(LEDGER,{'authorized_increment_usd':9.0,'user_ceiling_exclusive_usd':10.,'old_carryover_preserved':True,
              'baseline_key_usage':usage,'records':[],'note':'New authorization applies only to requests started in this pilot. Old carryover is untouched.'})
    print(json.dumps({'model':cfg['request']['model'],'tag':entry['tag'],'pricing':entry['pricing'],'usage':usage}))

def call(tag,messages,schema):
    """No retries; reserve $0.03 before each <=24 KB/512-output-token call."""
    DIR.mkdir(parents=True,exist_ok=True); out=DIR/(tag+'.json')
    if out.exists(): return json.loads(out.read_text(encoding='utf-8'))['parsed']
    cfg=json.loads((ROOT/'configs/luna_standard.json').read_text())
    cfg['request']['response_format']['json_schema']={'name':'pilot_record','strict':True,'schema':schema}
    ledger=json.loads(LEDGER.read_text()); reserve=.03
    if any(r['tag']==tag for r in ledger['records']): raise RuntimeError('Unresolved/failed call: no automatic retry')
    spent=sum(r.get('cost',r['reserved']) for r in ledger['records'])
    if spent+reserve>ledger['authorized_increment_usd']: raise RuntimeError('Pilot budget limit')
    body=build_request(cfg,messages,len(ledger['records']))
    check_metadata(json.loads((DIR/'metadata_live.json').read_text()),body)
    rec={'tag':tag,'reserved':reserve,'status':'reserved','started':time.time()}; ledger['records'].append(rec); write(LEDGER,ledger)
    # Persist request without headers before sending; retain reservation on ambiguity.
    write(DIR/(tag+'.request.json'),body)
    r=requests.post(cfg['endpoint_url'],headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'},json=body,timeout=90)
    raw=r.json(); write(DIR/(tag+'.raw.json'),raw)
    cost=raw.get('usage',{}).get('cost')
    if isinstance(cost,(int,float)) and cost>=0: rec['cost']=cost
    rec['status']='received'; rec['http_status']=r.status_code; write(LEDGER,ledger)
    r.raise_for_status()
    parsed=validate_response(raw,schema)
    write(out,{'parsed':parsed,'response_id':raw.get('id'),'cost':cost}); rec['status']='validated'; write(LEDGER,ledger)
    return parsed

def finish():
    r=requests.get('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+key()},timeout=30)
    r.raise_for_status(); d=r.json()['data']; ledger=json.loads(LEDGER.read_text())
    ledger['final_key_usage']={k:d.get(k) for k in ('usage','usage_daily','limit','limit_remaining')}
    ledger['billed_increment_usd']=sum(x.get('cost',0) for x in ledger['records'])
    ledger['unresolved_reservations_usd']=sum(x['reserved'] for x in ledger['records'] if 'cost' not in x)
    ledger['finished_at']=time.time(); write(LEDGER,ledger)
    print(json.dumps({k:ledger[k] for k in ('billed_increment_usd','unresolved_reservations_usd','final_key_usage')}))

if __name__=='__main__':
    import sys
    finish() if '--finish' in sys.argv else preflight()
