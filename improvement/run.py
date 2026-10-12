"""Public-belief planning distillation and PPO fine-tuning, train/dev selected."""
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from improvement import OUT
from study import OUT as MAIN
from study.env import LunaEnv
from study.evaluate import planner_for,encoded_bank,run_episode,assess,mean_grouped,fixed
from study.train import Record
from pilot.env import Config,observations,posterior,advance
from pilot.live import write

def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Follow-up protocol already exists')
    OUT.mkdir(exist_ok=True)
    old=json.loads((MAIN/'numeric_conditions.json').read_text())
    seen={json.dumps(c,sort_keys=True) for rows in old.values() for c in rows}
    rng=np.random.default_rng(2026101202);fresh=[]
    while len(fresh)<64:
        c=asdict(Config(prior=float(rng.choice([.2,.35,.5,.65,.8])),reliability=float(rng.choice([.6,.75,.9,1.])),
          query_cost=float(rng.choice([.01,.02,.04,.08])),rejection_cost=float(rng.choice([0.,.04,.08,.16,.32])),
          signal_effect=float(rng.choice([0.,.1,.25])),hard_cost=float(rng.choice([.4,.45,.55,.9]))))
        key=json.dumps(c,sort_keys=True)
        if key not in seen:seen.add(key);fresh.append(c)
    write(OUT/'fresh_numeric_test.json',fresh)
    write(OUT/'protocol.json',{'time':time.time(),'reason':'User requested a policy that successfully uses information, beyond diagnosing failure.',
      'methods':['distilled','distilled_ppo'],'seeds':[17,29,43],'distillation_updates':2400,'batch_size':256,
      'teacher':'Exact public-belief planner on TRAIN contexts and numerical settings only. No true-type input to teacher or student.',
      'teacher_targets':'Action values distilled with softmax temperature .03; actor cross entropy plus .5 value MSE.',
      'data':'One expert and one .35-random exploration path per train context/config/style/nonzero latent branch. Full labels at visited public states.',
      'ppo_steps':102400,'ppo_config':'Same base PPO settings and architecture as original study, initialized with distillation.',
      'selection':'Choose distilled versus distilled_ppo by 3-seed average ORIGINAL DEV net utility, final checkpoint only.',
      'final_eval':'64 newly drawn unique numeric configurations on the same four test source contexts. Sources/language bank previously inspected; this is a follow-up numerical holdout, not new historical evidence.',
      'ablations':'At final evaluation, remove query or signal by selecting next-highest permitted learned logit; mask posterior separately. These are deployment interventions, not retrained ablations.',
      'comparators':['Original PPO 3 seeds','Dev-selected original heuristic','Public bank-aware planner','No-query planner'],
      'resource_caveat':'Teacher uses known simulator and extra computation; gains do not imply equal-budget model-free RL superiority.',
      'stop':'Report both candidates and all seeds, including failure. No final-test-driven tuning.',
      'api':'No new paid requests: reuse frozen Luna language bank; do not change any interpretation.'})
    print('FOLLOW-UP PROTOCOL FROZEN',flush=True)

def collect():
    if (OUT/'teacher_data.npz').exists():raise RuntimeError('Teacher data already exists')
    env=LunaEnv('train');rng=np.random.default_rng(8317);xx=[];qq=[];vv=[];cache={};action_counts=np.zeros(11,int)
    for ci,cfg in enumerate(env.configs):
        for ctx in env.contexts:
            for style in range(3):
                encoded=json.dumps([env.lookup[(ctx['id'],a,style)]['meaning'] for a in range(11)],sort_keys=True)
                solve=planner_for(cfg,encoded);meanings=json.loads(encoded)
                for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
                    ph=cfg.reliability if hard else 1-cfg.reliability
                    for u,py in [(0.,ph),(.999999,1-ph)]:
                        if pt*py<=0:continue
                        for explore in (False,True):
                            o,_=env.reset(seed=1,options={'config':cfg,'context':ctx,'variant':style,'hard':hard,'query_uniform':u})
                            for t in range(5):
                                key=(cfg,encoded,env.s)
                                if key not in cache:
                                    vals=[]
                                    for m in meanings:
                                        q=0.
                                        for y,lh,ls in observations(cfg,env.s,m):
                                            prob=env.s.b*lh+(1-env.s.b)*ls
                                            if prob<=1e-12:continue
                                            b,_=posterior(env.s.b,lh,ls);n,r,d,_=advance(cfg,env.s,m,y,b)
                                            q+=prob*(r+(0 if d else solve(n)[0]))
                                        vals.append(q)
                                    cache[key]=np.asarray(vals,np.float32)
                                q=cache[key];xx.append(o.copy());qq.append(q);vv.append(float(q.max()))
                                best=solve(env.s)[1];action_counts[best]+=1
                                action=int(rng.integers(11)) if explore and rng.random()<.35 else best
                                o,_,done,_,_=env.step(action)
                                if done:break
        cache.clear();planner_for.cache_clear()
        if (ci+1)%16==0:print('Teacher configs',ci+1,'/',len(env.configs),'states',len(xx),flush=True)
    np.savez_compressed(OUT/'teacher_data.npz',obs=np.array(xx),q=np.array(qq),v=np.array(vv))
    write(OUT/'teacher_data_metadata.json',{'states':len(xx),'optimal_action_counts':action_counts.tolist(),
      'contexts':[c['id'] for c in env.contexts],'configs':len(env.configs),'hidden_type_in_input':False,
      'sha256':hashlib.sha256((OUT/'teacher_data.npz').read_bytes()).hexdigest()})

def train_one(seed):
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    protocol=json.loads((OUT/'protocol.json').read_text());config=json.loads((MAIN/'protocol.json').read_text())['base_ppo'].copy();net=config.pop('net_arch')
    model=PPO('MlpPolicy',LunaEnv(),seed=seed,**config,policy_kwargs={'net_arch':net},device='cpu',verbose=0)
    data=np.load(OUT/'teacher_data.npz');x=torch.as_tensor(data['obs']);q=torch.as_tensor(data['q']);v=torch.as_tensor(data['v'])
    probs=torch.softmax((q-q.max(dim=1,keepdim=True).values)/.03,dim=1)
    opt=torch.optim.Adam(model.policy.parameters(),lr=.001);curve=[]
    for step in range(protocol['distillation_updates']):
        idx=rng.integers(len(x),size=protocol['batch_size']);distribution=model.policy.get_distribution(x[idx]).distribution
        ce=-(probs[idx]*torch.log_softmax(distribution.logits,dim=1)).sum(dim=1).mean()
        mse=((model.policy.predict_values(x[idx]).flatten()-v[idx])**2).mean();loss=ce+.5*mse
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);opt.step()
        if step%100==0:curve.append({'update':step,'cross_entropy':float(ce.detach()),'value_mse':float(mse.detach())})
    (OUT/'models').mkdir(exist_ok=True);model.save(OUT/f'models/distilled_{seed}');write(OUT/f'curves/distilled_{seed}.json',curve)
    print('Distillation finished',seed,flush=True)
    model.set_logger(configure(str(OUT/f'logs/distilled_ppo_{seed}'),['csv']));rec=Record()
    model.learn(total_timesteps=protocol['ppo_steps'],callback=rec);model.save(OUT/f'models/distilled_ppo_{seed}')
    write(OUT/f'curves/distilled_ppo_{seed}.json',rec.episodes)
    print('Fine-tuning finished',seed,flush=True)
    return seed

def train():
    with (OUT/'training_started.json').open('x') as f:json.dump({'time':time.time()},f)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(train_one,[17,29,43]))
    write(OUT/'training_complete.json',{'time':time.time()})

def evaluate():
    # Route all follow-up logs to the separate folder; original metrics are immutable.
    import study.evaluate as ev
    ev.OUT=OUT;torch.set_num_threads(1)
    if (OUT/'selection.json').exists():raise RuntimeError('Selection already frozen')
    dev=[]
    for method in ('distilled','distilled_ppo'):
        for seed in (17,29,43):
            m=PPO.load(OUT/f'models/{method}_{seed}',device='cpu')
            rs=assess(method,lambda o,e,m=m:int(m.predict(o,deterministic=True)[0]),split='dev',seed=seed)
            dev.append(dict(method=method,seed=seed,**mean_grouped(rs)))
    scores={method:float(np.mean([r['net_utility'] for r in dev if r['method']==method])) for method in ('distilled','distilled_ppo')}
    selected=max(scores,key=scores.get);write(OUT/'selection.json',{'time':time.time(),'dev':dev,'mean_scores':scores,'selected':selected})
    print('DEV SELECTION',selected,scores,flush=True)
    configs=[Config(**c) for c in json.loads((OUT/'fresh_numeric_test.json').read_text())]
    summaries=[];best=json.loads((MAIN/'heuristic_selection.json').read_text())['selected']
    actors=[('Heuristic',None,lambda o,e:fixed(best,e),{}),('Planner',None,lambda o,e:planner_for(e.cfg,encoded_bank(e))(e.s)[1],{}),
      ('Planner_no_query',None,lambda o,e:planner_for(e.cfg,encoded_bank(e),False)(e.s)[1],{})]
    for method in ('base','distilled','distilled_ppo'):
        for seed in (17,29,43):
            m=PPO.load((MAIN if method=='base' else OUT)/f'models/{method}_{seed}',device='cpu')
            actors.append((method,seed,lambda o,e,m=m:int(m.predict(o,deterministic=True)[0]),{}))
            if method==selected:
                for blocked,label in [(2,'no_query'),(1,'no_signal')]:
                    def act(o,e,m=m,blocked=blocked):
                        with torch.no_grad():p=m.policy.get_distribution(torch.as_tensor(o).reshape(1,-1)).distribution.logits.flatten().clone()
                        p[blocked]=-torch.inf;return int(torch.argmax(p))
                    actors.append((method+'_'+label,seed,act,{}))
                actors.append((method+'_masked_belief',seed,lambda o,e,m=m:int(m.predict(o,deterministic=True)[0]),{'mask':'belief'}))
    for name,seed,actor,options in actors:
        rs=assess(name,actor,seed=seed,configs=configs,env_options=options)
        row=dict(policy=name,seed=seed,**mean_grouped(rs));summaries.append(row)
        write(OUT/'seed_metrics.json',summaries)
        print('FOLLOW-UP TEST',name,seed,round(row['net_utility'],5),'query',round(row['queries'],3),'signal',round(row['signals'],3),flush=True)
    write(OUT/'evaluation_complete.json',{'time':time.time(),'selected':selected,'all_runs_reported':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','collect','train','evaluate']);args=parser.parse_args()
    globals()[args.stage]()
