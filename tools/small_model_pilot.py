"""Offline matched audit of Luna plus public-belief planning versus cached Sol."""
import json,time,collections
from pathlib import Path
from dataclasses import asdict
import pilot
import numpy as np
import pandas as pd
from pilot.env import Config
from pilot.live import write
from study.env import LunaEnv
from study.evaluate import run_episode,planner_for,encoded_bank

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'small_model_results'


def main():
    OUT.mkdir(exist_ok=True)
    write(OUT/'scope.json',{'time':time.time(),'reason':'User requested reusing existing evidence for small Luna plus strategy module; time constrained, no new paid experiment.',
        'comparison':'Same existing 64 conditions (16 numeric settings x 4 documents), style0, Luna execution bank.',
        'module':'Public-belief exact planning over declared opponent response model; no actual hidden type. Existing implementation now audited as a deployable known-model controller.',
        'limits':'Previously inspected conditions; this is an engineering pilot, not new held-out evidence. Planning is a strong known-rule benchmark, not a new algorithm. Cached language bank, not newly generated online dialogue.'})
    env=LunaEnv('test');configs=[Config(**c) for c in json.loads((ROOT/'study_results/numeric_conditions.json').read_text())['test'][:16]]
    rows=[];timings=[]
    for ctx in env.contexts:
        for ci,cfg in enumerate(configs):
            planner_for.cache_clear();start=time.perf_counter();calls=[0]
            def actor(o,e):
                calls[0]+=1
                return planner_for(e.cfg,encoded_bank(e))(e.s)[1]
            for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
                ph=cfg.reliability if hard else 1-cfg.reliability
                for u,py in [(0.,ph),(.999999,1-ph)]:
                    if pt*py<=0:continue
                    z=run_episode(env,actor,cfg,ctx,0,hard,u,123)
                    assert z['acceptance_violations']==0
                    rows.append(dict(policy='Luna_public_planning',context=ctx['id'],condition=ci,config=asdict(cfg),weight=pt*py,
                        hidden_evaluation_only={'hard':hard,'query_uniform':u},**z))
            timings.append({'context':ctx['id'],'condition':ci,'seconds':time.perf_counter()-start,'decisions':calls[0]})
    write(OUT/'planning_paths.json',rows);write(OUT/'planning_local_timing.json',timings)
    grouped=collections.defaultdict(lambda:collections.defaultdict(float))
    for r in rows:
        for k in ['net_utility','agreement','queries','signals','length']:grouped[(r['context'],r['condition'])][k]+=r['weight']*r[k]
    planning={k:float(np.mean([r[k] for r in grouped.values()])) for k in next(iter(grouped.values()))}
    controls=pd.read_csv(ROOT/'sol_results/matched_policy_controls.csv');direct=pd.read_csv(ROOT/'sol_results/direct_comparison.csv')
    previous=float(controls[(controls.bank=='Luna')&(controls.policy=='Planner_None')].net_utility.iloc[0]);assert abs(planning['net_utility']-previous)<1e-10
    learned=controls[(controls.bank=='Luna')&controls.policy.str.startswith('Selected_')]
    baseline=direct[direct.policy=='Sol_medium_luna'].iloc[0]
    table=[]
    for name,r,kind in [('Luna alone (none)',direct[direct.policy=='Luna_none_Luna_bank'].iloc[0],'API'),
        ('Luna + learned strategy (3 seeds)',learned[['net_utility','agreement','queries','signals','length']].mean(),'local neural policy'),
        ('Luna + public-belief planning',planning,'local known-rule planning'),('Sol medium',baseline,'API')]:
        table.append(dict(method=name,net_utility=float(r['net_utility']),agreement=float(r['agreement']),queries=float(r['queries']),signals=float(r['signals']),
            length=float(r['length']),strategic_api_calls_per_episode=0. if kind!='API' else float(r['length']),strategy_compute=kind))
    pd.DataFrame(table).to_csv(OUT/'matched_summary.csv',index=False)
    # Reconstruct actual logical Sol requests per episode; cache hits still count.
    from study.luna_policy import RULES
    from pilot.env import NAMES
    import hashlib
    costs=[]
    for p in (ROOT/'sol_results/direct_medium_luna_cases').glob('*.json'):
        for r in json.loads(p.read_text(encoding='utf-8')):
            usd=0.;tokens=0.;last='ambiguity'
            for tr in r['trajectory']:
                public={'state':tr['before'],'config':r['config'],'last_interpreted_function':last}
                digest=hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()[:18]
                tag=f'sol_medium_policy_{r["context"]}_{r["condition"]}_{digest}'
                raw=json.loads((ROOT/'sol_results/api'/f'{tag}.raw.json').read_text(encoding='utf-8'))
                usd+=raw['usage']['cost'];tokens+=raw['usage']['prompt_tokens']+raw['usage']['completion_tokens'];last=tr['meaning']['kind']
            costs.append({'weight':r['weight'],'usd':usd,'tokens':tokens})
    findings={'matched_cases':64,'freshly_replayed_planning_paths':len(rows),'planning':planning,
        'learned_luna_utility':float(learned.net_utility.mean()),'sol_medium_utility':float(baseline.net_utility),
        'learned_retention_ratio':float(learned.net_utility.mean()/baseline.net_utility),
        'planning_minus_sol':planning['net_utility']-float(baseline.net_utility),
        'sol_strategic_calls_per_episode':float(baseline.length),
        'sol_strategic_api_usd_per_episode':sum(r['weight']*r['usd'] for r in costs)/64,
        'sol_strategic_tokens_per_episode':sum(r['weight']*r['tokens'] for r in costs)/64,
        'planning_local_seconds_total':sum(t['seconds'] for t in timings),
        'local_timing_scope':'64 complete enumerations of hidden/query branches, clearing planning cache before each cell; not a latency comparison with API.',
        'strategy_api_calls_for_controller':0,'new_api_calls_for_this_audit':0,
        'cost_scope':'Only strategy-selection API eliminated; cached language realization/interpretation and offline teacher/training costs are excluded. No end-to-end cost reduction percentage claim.'}
    write(OUT/'findings.json',findings)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Luna 전략 모듈 파일럿</title><style>body{font:16px/1.65 system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#173440}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ccd}pre{white-space:pre-wrap;background:#eef4f5;padding:15px}</style><h1>작은 Luna + 공개 상태 전략 모듈</h1>
    <p>주어진 외교 문장에 근거한 합성 협상에서, 발화 실현은 Luna가 맡고 질문·신호·양보·비준 선택은 상대 반응으로 갱신한 믿음과 비용을 사용하는 로컬 모듈이 맡는다.</p>'''
    page+=pd.DataFrame(table).round(5).to_html(index=False)
    page+=f'''<p>같은 기존 64개 조건에서 학습 정책은 Sol medium 순효용의 {findings['learned_retention_ratio']:.1%}에 해당한다. 알려진 공개 전이 규칙으로 계획하는 모듈은 Sol 대비 {findings['planning_minus_sol']:+.5f}였고, 이번에 {len(rows)}개 경로를 다시 실행해 기존 값과 일치함을 확인했다.</p>
    <p>Sol의 전략 선택은 협상당 평균 {findings['sol_strategic_calls_per_episode']:.3f}회, {findings['sol_strategic_tokens_per_episode']:.1f}토큰, ${findings['sol_strategic_api_usd_per_episode']:.5f}였다. 두 로컬 모듈의 전략 선택 API는 0회다. 공통 발화 bank·해석 비용과 학습·교사 비용을 제외한 수치이므로 전체 서비스 비용 절감률로 읽으면 안 된다.</p>
    <p>실제 자료는 표현과 맥락의 근거이며 효용·유형·반응은 합성이다. 계획기는 선언된 상대 반응 모델을 정확히 안다. 이는 알려진 규칙에서의 방법 상한을 구현한 것이며 범용 외교 성능이나 새 알고리즘의 증거가 아니다. Luna 단독은 none, Sol은 medium 설정이므로 크기만의 효과를 분리한 비교도 아니다. 새 온라인 발화를 생성한 결과가 아니라 고정된 모델 생성/해석 bank를 사용했다. 이 차이를 명시한 시스템 파일럿으로 사용한다.</p>
    <p>논문 스토리 후보: 언어 모델 크기를 키우는 대신, 공개 담화에서 갱신한 믿음과 행동 비용을 다루는 전략 제어를 분리하여 작은 모델 시스템의 협상 성능과 전략 추론 효율을 높인다. PPO는 과제의 필수 학습 비교이자 모듈 구현 중 하나이며 새 알고리즘으로 주장하지 않는다.</p></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(findings,indent=2),flush=True)


if __name__=='__main__':main()
