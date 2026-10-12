"""Fresh-process, saved-checkpoint and raw-log reproduction; not a teammate sign-off."""
import json,gzip,hashlib,sys
import numpy as np
import torch
from stable_baselines3 import PPO
from study import OUT,ROOT
from study.env import LunaEnv
from study.evaluate import run_episode
from pilot.env import Config,own_utility,other_utility
from pilot.live import write

def main():
    torch.set_num_threads(1)
    model=PPO.load(OUT/'models/base_17',device='cpu');env=LunaEnv('test')
    contexts={x['id']:x for x in env.contexts}; source=OUT/'raw/test_PPO_base_17.jsonl.gz'
    matches=[];checked=0;violations=0
    with gzip.open(source,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); costs=sum(sum(i['costs'].values()) for i in r['trajectory'])
            assert abs(r['net_utility']-(r['own_utility']-costs))<1e-8
            assert len(r['trajectory'])<=5
            for before,after in zip(r['trajectory'],r['trajectory'][1:]):assert before['after']==after['before']
            checked+=1; violations+=r['acceptance_violations']
            if len(matches)<24:
                hidden=r['hidden_evaluation_only'];c=Config(**r['config'])
                again=run_episode(env,lambda o,e:int(model.predict(o,deterministic=True)[0]),c,contexts[r['context']],r['variant'],hidden['hard'],hidden['query_uniform'],123)
                identical=abs(again['net_utility']-r['net_utility'])<1e-10 and [(i['action'],i['meaning'],i['response']) for i in again['trajectory']]==[(i['action'],i['meaning'],i['response']) for i in r['trajectory']]
                assert identical
                matches.append({'context':r['context'],'condition':r['condition'],'variant':r['variant'],'same_path_and_return':identical})
    result={'process':'fresh Python invocation with reloaded saved model','reproduced_paths':len(matches),'all_reproduced':all(r['same_path_and_return'] for r in matches),
            'raw_paths_reward_checked':checked,'acceptance_violations':violations,'teammate_reproduction':False,
            'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'study').glob('*.py'))+[OUT/'protocol.json',OUT/'language_bank.json']}}
    write(OUT/'reproduction_record.json',result);print(json.dumps({k:v for k,v in result.items() if k!='hashes'}))

if __name__=='__main__':main()
