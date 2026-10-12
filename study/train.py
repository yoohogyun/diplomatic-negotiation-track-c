"""Run the frozen policy matrix. Final checkpoint only; no test selection."""
import json,time,pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from gymnasium.utils.env_checker import check_env
from study import OUT
from study.env import LunaEnv
from pilot.live import write

class Bandit:
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed); self.inv=np.array([np.eye(32)*10 for _ in range(11)]); self.weights=np.zeros((11,32))
    def predict(self,obs): return int(np.argmax(self.weights@np.r_[1.,obs]))
    def train(self,env,steps,seed):
        o,_=env.reset(seed=seed); stats=[]; ret=0.; length=0
        for t in range(steps):
            a=int(self.rng.integers(11)) if self.rng.random()<.2 else self.predict(o)
            x=np.r_[1.,o]; nxt,r,d,_,info=env.step(a); px=self.inv[a]@x; denom=1+x@px
            self.weights[a]+=px/denom*(r-self.weights[a]@x); self.inv[a]-=np.outer(px,px)/denom
            ret+=r; length+=1
            if d: stats.append([t+1,ret,length]); ret=0.; length=0
            o=env.reset()[0] if d else nxt
        return stats

class Record(BaseCallback):
    def __init__(self): super().__init__(); self.episodes=[]; self.running=0.; self.length=0; self.actions=np.zeros(11,int)
    def _on_step(self):
        i=self.locals['infos'][0]; self.running+=i['common_reward']; self.length+=1; self.actions[i['action']]+=1
        if self.locals['dones'][0]:
            self.episodes.append({'step':self.num_timesteps,'return':self.running,'length':self.length,'agreement':i['outcome']=='agreement'})
            self.running=0.; self.length=0
        return True

def train_run(job):
    run,seed,protocol=job; torch.set_num_threads(1)
    name=f'{run["name"]}_{seed}'; start=time.time()
    cfg=dict(protocol['base_ppo']); net=cfg.pop('net_arch')
    cfg.update({k:run[k] for k in ('gamma','ent_coef') if k in run})
    env=LunaEnv(mask=run.get('mask'),remove_reward=run.get('remove_reward'),disable_query=run.get('disable_query',False))
    model=PPO('MlpPolicy',env,seed=seed,**cfg,policy_kwargs={'net_arch':net},device='cpu',verbose=0)
    model.set_logger(configure(str(OUT/'training_logs'/name),['csv']))
    record=Record(); model.learn(total_timesteps=protocol['steps'],callback=record); model.save(OUT/'models'/name)
    write(OUT/'curves'/(name+'.json'),record.episodes)
    write(OUT/'training_logs'/name/'metadata.json',{'seconds':time.time()-start,'action_counts':record.actions.tolist(),'config':run,'seed':seed})
    return name,round(time.time()-start,1)

def main():
    torch.set_num_threads(1)
    deadline=time.time()+1800
    while not (OUT/'language_bank.json').exists():
        if time.time()>deadline: raise RuntimeError('Language bank not ready')
        time.sleep(5)
    protocol=json.loads((OUT/'protocol.json').read_text()); (OUT/'models').mkdir(exist_ok=True)
    check_env(LunaEnv(),skip_render_check=True)
    with (OUT/'training_start.json').open('x') as f: json.dump({'time':time.time(),'steps':protocol['steps'],'parallel_processes':2},f)
    for seed in protocol['seeds']:
        b=Bandit(seed); stats=b.train(LunaEnv(),protocol['steps'],seed)
        with (OUT/f'models/bandit_{seed}.pkl').open('wb') as f: pickle.dump(b,f)
        write(OUT/f'curves/bandit_{seed}.json',stats)
        print('Bandit finished',seed,flush=True)
    jobs=[(run,seed,protocol) for run in protocol['runs'] for seed in run['seeds']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for name,seconds in pool.map(train_run,jobs): print('PPO finished',name,'seconds',seconds,flush=True)
    write(OUT/'training_complete.json',{'time':time.time(),'all_prespecified_runs_complete':True})

if __name__=='__main__': main()
