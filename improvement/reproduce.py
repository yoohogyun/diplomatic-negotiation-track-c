"""Reload selected learned policy and reproduce every saved seed-17 test path."""
import json,gzip,hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from improvement import OUT
from study.env import LunaEnv
from study.evaluate import run_episode
from pilot.env import Config
from pilot.live import write

def main():
    torch.set_num_threads(1);name=json.loads((OUT/'selection.json').read_text())['selected']
    model=PPO.load(OUT/f'models/{name}_17',device='cpu');env=LunaEnv('test');contexts={c['id']:c for c in env.contexts}
    n=0;questions=0;signals=0
    with gzip.open(OUT/f'raw/test_{name}_17.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);h=r['hidden_evaluation_only']
            z=run_episode(env,lambda o,e:int(model.predict(o,deterministic=True)[0]),Config(**r['config']),contexts[r['context']],r['variant'],h['hard'],h['query_uniform'],123)
            assert abs(z['net_utility']-r['net_utility'])<1e-9
            assert [(t['action'],t['meaning'],t['response']) for t in z['trajectory']]==[(t['action'],t['meaning'],t['response']) for t in r['trajectory']]
            n+=1;questions+=z['queries'];signals+=z['signals']
    result={'selected':name,'seed':17,'reproduced_paths':n,'query_turns_in_reproduced_paths':questions,'signal_turns_in_reproduced_paths':signals,
            'all_match':True,'new_process':True,'model_sha256':hashlib.sha256((OUT/f'models/{name}_17.zip').read_bytes()).hexdigest()}
    write(OUT/'reproduction.json',result);print(json.dumps(result))

if __name__=='__main__':main()
