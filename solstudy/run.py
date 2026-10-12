"""Model-only replication; original Luna protocols and outputs are never overwritten."""
import argparse,json,time,hashlib,pickle,shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from solstudy import OUT,ROOT,IMPROVED
from study import OUT as LUNA
from pilot.live import write

def configure():
    import study.env as env,study.train as train,study.evaluate as ev,study.sensitivity as sens,study.review as rev
    env.OUT=OUT;train.OUT=OUT;ev.OUT=OUT;sens.OUT=OUT;rev.OUT=OUT

def prepare():
    if (OUT/'protocol.json').exists():raise RuntimeError('Sol protocol already frozen')
    cfg=json.loads((ROOT/'configs/luna_standard.json').read_text())
    cfg['request']['model']='openai/gpt-5.6-sol';cfg['metadata_url']='https://openrouter.ai/api/v1/models/openai/gpt-5.6-sol/endpoints'
    cfg['metadata_snapshot']='sol_results/api/metadata_live.json';cfg['request']['provider']['max_price']={'prompt':2.,'completion':10.}
    write(ROOT/'configs/sol_standard.json',cfg)
    from infra.openrouter import build_request,check_metadata
    check_metadata(json.loads((OUT/'api/metadata_live.json').read_text()),build_request(cfg,[{'role':'user','content':'Return a diplomatic sentence.'}],0))
    for file in ['contexts.json','numeric_conditions.json']:shutil.copyfile(LUNA/file,OUT/file)
    p=json.loads((LUNA/'protocol.json').read_text());p.update(version='sol-replication-v1',frozen_at=time.time(),model='openai/gpt-5.6-sol')
    p['replication']={'reason':'User explicitly requested testing the same setup with GPT-5.6 Sol.',
      'primary_direct':'Same 64 cases, prompt and reasoning none as Luna, on the unchanged Luna bank to isolate selector model.',
      'secondary_direct':'Sol reasoning medium, same 64 cases on Luna bank; max output including reasoning 4096.',
      'own_bank_direct':'Sol reasoning none on Sol bank, same cases. Cache reused only if public prompt exactly matches.',
      'language':'396 Sol generations and 396 Sol interpretations, 180 same probes; no correction of errors.',
      'learning':'Replicate all 14 main learned runs and six distillation/fine-tuning checkpoints, same seeds, budgets and dev selection.',
      'evaluation':'Same original test for main results; same follow-up 64 numeric settings for improvement. Both are previously exposed, a model replication rather than new blind evidence.',
      'stop':'No hyperparameter/prompt tuning to favor Sol or the proposed method. Report contrary results.',
      'budget':'Same shared ledger; cumulative internal ceiling $9, user exclusive ceiling $10; no reset.',
      'interpretation':'PPO is an established algorithm, but the configured negotiation policy and distillation system are our constructed baselines. Do not claim a general LLM failure from their weaknesses.'}
    write(OUT/'protocol.json',p);print('SOL PROTOCOL FROZEN',flush=True)

def language():
    import study.language as lang
    from solstudy.api import call
    lang.OUT=OUT;lang.call=call;lang.main()

def direct(mode):
    configure()
    from study.env import LunaEnv
    from study.evaluate import run_episode
    from study.luna_policy import RULES
    from pilot.env import Config,NAMES
    from pilot.language import schema
    from solstudy.api import call,batch
    effort='medium' if mode=='medium_luna' else 'none';bank_source=OUT if mode=='none_sol' else LUNA
    bank=json.loads((bank_source/'language_bank.json').read_text(encoding='utf-8'))
    contexts=[c for c in json.loads((OUT/'contexts.json').read_text(encoding='utf-8')) if c['split']=='test']
    configs=[Config(**c) for c in json.loads((OUT/'numeric_conditions.json').read_text())['test'][:16]]
    marker=OUT/f'direct_{mode}_protocol.json'
    if not marker.exists():write(marker,{'time':time.time(),'mode':mode,'numeric_indices':list(range(16)),'contexts':[c['id'] for c in contexts],'style':0,'effort':effort,'bank':str(bank_source),'single_response_cache':True})
    def worker(job):
        ctx,ci,cfg=job;env=LunaEnv('test',bank=bank);cache={};records=[]
        def actor(o,e):
            public={'state':asdict(e.s),'config':asdict(cfg),'last_interpreted_function':e.last_kind}
            digest=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()[:18]
            if digest not in cache:
                z=call(f'policy_{ctx["id"]}_{ci}_{digest}',[{'role':'system','content':RULES},{'role':'user','content':json.dumps({'source_context':ctx['source']['sentence'],'our_measure':ctx['our_measure'],'their_measure':ctx['their_measure'],'public':public,'actions':dict(enumerate(NAMES))},ensure_ascii=False)}],schema({'action':{'type':'number','minimum':0,'maximum':10}}),effort=effort)
                if z['action']!=int(z['action']):raise ValueError('Noninteger action')
                cache[digest]=int(z['action'])
            return cache[digest]
        for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
            ph=cfg.reliability if hard else 1-cfg.reliability
            for u,py in [(0.,ph),(.999999,1-ph)]:
                if pt*py<=0:continue
                z=run_episode(env,actor,cfg,ctx,0,hard,u,1)
                records.append(dict(policy='Direct_Sol_'+mode,context=ctx['id'],event_group=ctx['event_group'],condition=ci,config=asdict(cfg),variant=0,weight=pt*py,hidden_evaluation_only={'hard':hard,'query_uniform':u},**z))
        write(OUT/f'direct_{mode}_cases/{ctx["id"]}_{ci}.json',records)
        return len(cache)
    counts=batch([(c,i,cfg) for c in contexts for i,cfg in enumerate(configs)],worker,4)
    write(OUT/f'direct_{mode}_complete.json',{'time':time.time(),'cases':64,'unique_states':sum(counts)})
    print('SOL DIRECT COMPLETE',mode,flush=True)

def train_worker(job):
    configure()
    from study.train import train_run
    return train_run(job)

def train():
    configure()
    import torch
    from study.train import Bandit
    from study.env import LunaEnv
    torch.set_num_threads(1);p=json.loads((OUT/'protocol.json').read_text());(OUT/'models').mkdir(exist_ok=True)
    with (OUT/'training_start.json').open('x') as f:json.dump({'time':time.time()},f)
    for seed in p['seeds']:
        b=Bandit(seed);stats=b.train(LunaEnv(),p['steps'],seed)
        with (OUT/f'models/bandit_{seed}.pkl').open('wb') as f:pickle.dump(b,f)
        write(OUT/f'curves/bandit_{seed}.json',stats);print('SOL Bandit finished',seed,flush=True)
    jobs=[(r,s,p) for r in p['runs'] for s in r['seeds']]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(train_worker,jobs):print('SOL PPO finished',result,flush=True)
    write(OUT/'training_complete.json',{'time':time.time(),'all_prespecified_runs_complete':True})

def improve_configure():
    configure()
    import improvement.run as imp
    imp.OUT=IMPROVED;imp.MAIN=OUT
    return imp

def improve_worker(seed):return improve_configure().train_one(seed)

def improve():
    imp=improve_configure();IMPROVED.mkdir(exist_ok=True)
    if not (IMPROVED/'protocol.json').exists():
        p=json.loads((ROOT/'improvement_results/protocol.json').read_text());p.update(time=time.time(),model='openai/gpt-5.6-sol',reason='Exact-method replication under Sol language bank.')
        write(IMPROVED/'protocol.json',p);shutil.copyfile(ROOT/'improvement_results/fresh_numeric_test.json',IMPROVED/'fresh_numeric_test.json')
    imp.collect()
    with (IMPROVED/'training_started.json').open('x') as f:json.dump({'time':time.time()},f)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(improve_worker,[17,29,43]))
    write(IMPROVED/'training_complete.json',{'time':time.time()})

def offline(stage):
    configure()
    if stage=='evaluate':
        import study.evaluate as x;x.main()
    elif stage=='sensitivity':
        import study.sensitivity as x;x.main()
    elif stage=='review':
        import study.review as x;x.main()
    elif stage=='reproduce':
        import study.reproduce as x;x.OUT=OUT;x.main()
    elif stage=='improve_evaluate':improve_configure().evaluate()
    elif stage=='improve_report':
        import improvement.report as x;x.OUT=IMPROVED;x.main()
    elif stage=='improve_reproduce':
        import improvement.reproduce as x;x.OUT=IMPROVED;x.main()

def local_pipeline():
    deadline=time.time()+3600
    while not (OUT/'language_bank.json').exists():
        if time.time()>deadline:raise RuntimeError('Sol bank not ready')
        time.sleep(5)
    configure()
    from study.env import LunaEnv
    from gymnasium.utils.env_checker import check_env
    check_env(LunaEnv(),skip_render_check=True)
    train()
    for stage in ['evaluate','sensitivity','review','reproduce']:offline(stage)
    improve()
    for stage in ['improve_evaluate','improve_report','improve_reproduce']:offline(stage)
    write(OUT/'local_pipeline_complete.json',{'time':time.time()});print('SOL LOCAL PIPELINE COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','language','none_luna','medium_luna','none_sol','train','evaluate','sensitivity','review','reproduce','improve','improve_evaluate','improve_report','improve_reproduce','local_pipeline']);a=p.parse_args()
    if a.stage in ('none_luna','medium_luna','none_sol'):direct(a.stage)
    elif a.stage in ('prepare','language','train','improve','local_pipeline'):globals()[a.stage]()
    else:offline(a.stage)
