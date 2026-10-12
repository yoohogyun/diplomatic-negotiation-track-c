"""Explicit Sol authorization, shared cumulative budget and no automatic retries."""
import json,time
import requests
from solstudy import OUT,ROOT
from study.api import LOCK,batch
from pilot.live import LEDGER,key,write
from infra.openrouter import build_request,check_metadata,validate_response

def call(tag,messages,schema,effort='none'):
    tag='sol_'+effort+'_'+tag;path=OUT/'api'/(tag+'.json')
    with LOCK:
        if path.exists():
            request=json.loads((OUT/'api'/(tag+'.request.json')).read_text(encoding='utf-8'))
            assert request['messages']==messages,'Cached request content changed'
            return json.loads(path.read_text(encoding='utf-8'))['parsed']
        ledger=json.loads(LEDGER.read_text())
        if any(r['tag']==tag for r in ledger['records']):raise RuntimeError('Unresolved/failed request; no automatic retry: '+tag)
        cfg=json.loads((ROOT/'configs/sol_standard.json').read_text())
        cfg['request']['reasoning']['effort']=effort
        cfg['request']['max_tokens']=4096 if effort=='medium' else 512
        cfg['request']['response_format']['json_schema']={'name':'study_record','strict':True,'schema':schema}
        body=build_request(cfg,messages,len(ledger['records']))
        check_metadata(json.loads((OUT/'api/metadata_live.json').read_text()),body)
        # Conservative byte-count upper bound plus framing allowance; output includes reasoning.
        prices=body['provider']['max_price']
        reserve=(len(json.dumps(messages,ensure_ascii=False).encode('utf-8'))+1024)*prices['prompt']/1e6+body['max_tokens']*prices['completion']/1e6
        spent=sum(r.get('cost',r['reserved']) for r in ledger['records'])
        if spent+reserve>ledger['authorized_increment_usd']:raise RuntimeError('Cumulative budget ceiling; no new request sent')
        record={'tag':tag,'reserved':reserve,'status':'reserved','started':time.time(),'model':body['model'],'effort':effort}
        ledger['records'].append(record);write(LEDGER,ledger);write(OUT/'api'/(tag+'.request.json'),body)
    response=requests.post(cfg['endpoint_url'],headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'},json=body,timeout=180)
    raw=response.json();write(OUT/'api'/(tag+'.raw.json'),raw)
    with LOCK:
        ledger=json.loads(LEDGER.read_text());record=next(r for r in ledger['records'] if r['tag']==tag)
        cost=raw.get('usage',{}).get('cost')
        if isinstance(cost,(int,float)) and cost>=0:record['cost']=cost
        record.update(status='received',http_status=response.status_code);write(LEDGER,ledger)
    response.raise_for_status();parsed=validate_response(raw,schema,expected_model='openai/gpt-5.6-sol')
    with LOCK:
        write(path,{'parsed':parsed,'response_id':raw.get('id'),'cost':cost})
        ledger=json.loads(LEDGER.read_text());next(r for r in ledger['records'] if r['tag']==tag)['status']='validated';write(LEDGER,ledger)
    return parsed
