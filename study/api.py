"""Four concurrent transports, serial durable reservations and settlements."""
import threading,json,time,os
if os.name=='nt': import msvcrt
else: import fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from study import ROOT,OUT
from pilot.live import key,write,LEDGER
from infra.openrouter import build_request,check_metadata,validate_response

class BudgetLock:
    """Thread and process exclusion for a shared incremental budget ledger."""
    def __init__(self): self.thread=threading.Lock()
    def __enter__(self):
        self.thread.acquire()
        try:
            self.file=(ROOT/'budget'/'study_api.lock').open('a+b')
            if self.file.tell()==0:self.file.write(b'0');self.file.flush()
            while True:
                try:
                    self.file.seek(0)
                    if os.name=='nt':msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
                    else:fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except OSError:time.sleep(.1)
            return self
        except Exception:
            self.thread.release();raise
    def __exit__(self,*exc):
        self.file.seek(0)
        if os.name=='nt':msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close();self.thread.release()
LOCK=BudgetLock()
API=OUT/'api'

def call(tag,messages,schema):
    tag='study_'+tag; path=API/(tag+'.json')
    with LOCK:
        if path.exists(): return json.loads(path.read_text(encoding='utf-8'))['parsed']
        ledger=json.loads(LEDGER.read_text())
        if any(r['tag']==tag for r in ledger['records']): raise RuntimeError('Existing unresolved call '+tag)
        cfg=json.loads((ROOT/'configs/luna_standard.json').read_text())
        cfg['request']['response_format']['json_schema']={'name':'study_record','strict':True,'schema':schema}
        body=build_request(cfg,messages,len(ledger['records']))
        check_metadata(json.loads((ROOT/'pilot_results/api/metadata_live.json').read_text()),body)
        reserve=.03; used=sum(r.get('cost',r['reserved']) for r in ledger['records'])
        if used+reserve>ledger['authorized_increment_usd']: raise RuntimeError('Budget ceiling reached')
        record={'tag':tag,'reserved':reserve,'status':'reserved','started':time.time()}
        ledger['records'].append(record); write(LEDGER,ledger)
        write(API/(tag+'.request.json'),body)
    response=requests.post(cfg['endpoint_url'],headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'},json=body,timeout=90)
    raw=response.json(); write(API/(tag+'.raw.json'),raw)
    with LOCK:
        ledger=json.loads(LEDGER.read_text()); record=next(x for x in ledger['records'] if x['tag']==tag)
        cost=raw.get('usage',{}).get('cost')
        if isinstance(cost,(float,int)) and cost>=0: record['cost']=cost
        record.update(status='received',http_status=response.status_code); write(LEDGER,ledger)
    response.raise_for_status(); parsed=validate_response(raw,schema)
    with LOCK:
        write(path,{'parsed':parsed,'response_id':raw.get('id'),'cost':cost})
        ledger=json.loads(LEDGER.read_text()); next(x for x in ledger['records'] if x['tag']==tag)['status']='validated'; write(LEDGER,ledger)
    return parsed

def batch(items,fn,workers=4):
    results=[]; failures=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fn,x):x for x in items}
        for i,f in enumerate(as_completed(futures),1):
            try: results.append(f.result())
            except Exception as e:
                failures.append({'item':str(futures[f])[:160],'error':str(e)[:300]})
                # No retries; cancel not-yet-started jobs after an error.
                for pending in futures: pending.cancel()
                break
            if i%25==0: print('Luna completed',i,'/',len(items),flush=True)
    if failures:
        write(OUT/'api_failures.json',failures); raise RuntimeError('API batch failed; inspect preserved records')
    return results
