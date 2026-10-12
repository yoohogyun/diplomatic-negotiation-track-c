"""Offline reproduction in child processes; preserve all archived experiment files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'verification'
STAGES={
    'quick':['tests','main_replay'],
    'main':['main_replay','main_audit'],
    'improved':['improved_replay'],
    'sol':['sol_replay','sol_improved_replay'],
    'all':['tests','main_replay','main_audit','improved_replay','sol_replay','sol_improved_replay'],
}

def worker(stage):
    import socket
    def disabled(*args,**kwargs):raise RuntimeError('Network disabled during offline reproduction')
    socket.socket.connect=disabled
    socket.socket.connect_ex=disabled
    socket.create_connection=disabled
    import pilot  # loads NumPy before PyTorch
    import requests
    requests.sessions.Session.request=disabled
    import tempfile
    temp=OUT/'tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp)
    import pilot.live as live
    original_write=live.write
    def redirected(p,obj):
        relative=Path(p).resolve().relative_to(ROOT)
        original_write(OUT/stage/relative,obj)
    live.write=redirected
    import torch
    torch.set_num_threads(1)
    if stage=='tests':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        original_write(OUT/'tests.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'network_disabled':True})
        if not result.wasSuccessful():raise SystemExit(1)
    elif stage=='main_replay':
        from study.reproduce import main
        main()
    elif stage=='main_audit':
        from study.audit import main
        main()
    elif stage=='improved_replay':
        from improvement.reproduce import main
        main()
    elif stage in ('sol_replay','sol_improved_replay'):
        from solstudy.run import offline
        offline('reproduce' if stage=='sol_replay' else 'improve_reproduce')
    else:raise ValueError(stage)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=list(STAGES)+['worker'])
    p.add_argument('--task')
    args=p.parse_args();OUT.mkdir(exist_ok=True)
    if args.stage=='worker':return worker(args.task)
    from tools.verify_release import verify
    integrity=verify()
    results=[]
    env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1')
    env.pop('OR_KEY',None)
    for stage in STAGES[args.stage]:
        print('Running offline:',stage,flush=True)
        command=[sys.executable,'-X','utf8','-m','tools.reproduce_release','worker','--task',stage]
        with (OUT/(stage+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append({'stage':stage,'returncode':r.returncode,'log':'verification/'+stage+'.log'})
        print(stage, 'PASS' if r.returncode==0 else 'FAIL',flush=True)
        if r.returncode:break
    summary={'stage':args.stage,'integrity':integrity,'checks':results,'all_passed':len(results)==len(STAGES[args.stage]) and all(x['returncode']==0 for x in results),'paid_calls':0,'teammate_reproduction':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if not summary['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
