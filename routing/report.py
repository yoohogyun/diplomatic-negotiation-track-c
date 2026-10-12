"""Replay every pilot path, audit cached requests, and report paired resource outcomes."""
import hashlib, html, json, time
from dataclasses import asdict
import numpy as np
from routing import ROOT, OUT
from routing.run import load_models
from pilot.env import Config
from pilot.live import write, LEDGER
from study.env import LunaEnv


def main():
    selection=json.loads((OUT/'selection.json').read_text());summary=json.loads((OUT/'test_summary.json').read_text())
    cases=json.loads((OUT/'test_case_metrics.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    cheap,gates=load_models()
    import torch
    rows=[]
    for p in sorted((OUT/'cases').glob('*.json')):rows.extend(json.loads(p.read_text(encoding='utf-8')))
    contexts={c['id']:c for c in json.loads((ROOT/'study_results/contexts.json').read_text(encoding='utf-8'))}
    bank=json.loads((ROOT/'sol_results/language_bank.json').read_text(encoding='utf-8'));envs={s:LunaEnv(s,bank=bank) for s in ('dev','test')}
    public_requests={};turns=0;cheap_decisions=0
    for r in rows:
        env=envs[r['split']];o,_=env.reset(seed=1,options={'config':Config(**r['config']),'context':contexts[r['context']],
            'variant':0,**r['hidden_evaluation_only']})
        ret=0.;usage={'sol_calls':0.,'prompt_tokens':0.,'completion_tokens':0.,'reasoning_tokens':0.,'api_usd':0.}
        for i,trace in enumerate(r['trajectory']):
            assert asdict(env.s)==trace['before'];turns+=1
            if r['policy']!='planner':
                d=r['decisions'][i];assert d['public_state']==asdict(env.s)
                a=int(cheap.predict(o,deterministic=True)[0]);assert a==d['cheap_action'];cheap_decisions+=1
                with torch.no_grad():q=torch.stack([m(torch.as_tensor(o)) for m in gates]).mean(0)
                regret=float(q.max()-q[a]);assert abs(regret-d['predicted_regret'])<1e-6
                if r['policy']=='initial_sol':expected_route=env.s.t==0
                else:
                    threshold=-1. if r['policy']=='always_sol' else float('inf') if r['policy']=='cheap' else float(r['policy'].removeprefix('gate_'))
                    expected_route=regret>threshold
                assert d['route_to_sol']==expected_route
                if d['route_to_sol']:
                    tag=d['api_tag'];p=ROOT/'sol_results/api'/tag
                    request=json.loads(p.with_suffix('.request.json').read_text(encoding='utf-8'))
                    payload=json.loads(request['messages'][1]['content'])
                    assert set(payload)=={'source_context','our_measure','their_measure','public','actions'}
                    assert payload['public']=={'state':asdict(env.s),'config':r['config'],'last_interpreted_function':env.last_kind}
                    assert payload['source_context']==contexts[r['context']]['source']['sentence']
                    assert request['reasoning']['effort']=='medium'
                    parsed=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));assert parsed['parsed']['action']==d['selected_action']
                    raw=json.loads(p.with_suffix('.raw.json').read_text(encoding='utf-8'));tokens=raw['usage'];public_requests[tag]=tokens
                    usage['sol_calls']+=1
                    for k in ('prompt_tokens','completion_tokens'):usage[k]+=tokens.get(k,0)
                    usage['reasoning_tokens']+=tokens.get('completion_tokens_details',{}).get('reasoning_tokens',0)
                    usage['api_usd']+=tokens['cost']
                else:assert d['selected_action']==a
                assert trace['action']==d['selected_action']
            o,_,done,_,actual=env.step(trace['action']);assert actual==trace;ret+=actual['common_reward']
        assert done and abs(ret-r['net_utility'])<1e-9
        assert not r['acceptance_violations']
        for k,v in usage.items():assert abs(v-r[k])<1e-9
    fresh=json.loads((OUT/'fresh_numeric_test.json').read_text());old=json.loads((ROOT/'study_results/numeric_conditions.json').read_text())
    seen={json.dumps(c,sort_keys=True) for rs in old.values() for c in rs}
    seen.update(json.dumps(c,sort_keys=True) for c in json.loads((ROOT/'improvement_results/fresh_numeric_test.json').read_text()))
    assert len({json.dumps(c,sort_keys=True) for c in fresh})==16
    assert all(json.dumps(c,sort_keys=True) not in seen for c in fresh)
    start=json.loads((OUT/'test_started.json').read_text());assert selection['time']<start['time']
    assert hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()==start['selection_sha256']
    name=selection['policy'];hybrid=summary[name];baseline=summary['always_sol']
    ci_diffs=[]
    for ci in range(16):
        matched={p:np.mean([r['net_utility'] for r in cases if r['policy']==p and r['condition']==ci]) for p in (name,'always_sol')}
        ci_diffs.append(float(matched[name]-matched['always_sol']))
    rng=np.random.default_rng(17012026);boots=np.array(ci_diffs)[rng.integers(16,size=(10000,16))].mean(1)
    ledger=json.loads(LEDGER.read_text());records=[r for r in ledger['records'] if r['tag'].startswith('sol_medium_routing_')]
    spent=sum(r.get('cost',0) for r in ledger['records']);unresolved=[r['tag'] for r in ledger['records'] if r['status']!='validated']
    recovery_path=OUT/'transport_recovery.json'
    recoveries=json.loads(recovery_path.read_text()) if recovery_path.exists() else {'replacements':{}}
    allowed_pending={'sol_medium_'+t for t in recoveries['replacements']}
    assert set(unresolved)==allowed_pending
    reserved=sum(r['reserved'] for r in ledger['records'] if 'cost' not in r)
    assert spent+reserved<9.
    validated=[r for r in records if r['status']=='validated']
    assert len(validated)==len(public_requests)
    result={'selected':name,'threshold':selection['threshold'],'test_cells':32,'numeric_conditions':16,'source_documents':2,
        'utility_delta':hybrid['net_utility']-baseline['net_utility'],
        'utility_retention_ratio':hybrid['net_utility']/baseline['net_utility'],
        'calls_reduction_fraction':1-hybrid['sol_calls']/baseline['sol_calls'],
        'token_reduction_fraction':1-(hybrid['prompt_tokens']+hybrid['completion_tokens'])/(baseline['prompt_tokens']+baseline['completion_tokens']),
        'api_cost_reduction_fraction':1-hybrid['api_usd']/baseline['api_usd'],
        'paired_numeric_deltas':ci_diffs,'exploratory_numeric_cluster_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
        'uncertainty_scope':'16 numerical draws only, averaging two fixed exposed sources and one cached model draw per public prompt; not a historical or model-seed CI.',
        'actual_new_unique_calls':len(validated),'new_attempts_including_timeout':len(records),
        'actual_new_api_usd':sum(r.get('cost',0) for r in records),'cumulative_api_usd':spent,
        'unresolved_transport_reserve_usd':reserved,'conservative_cumulative_usd':spent+reserved,
        'gate_training':json.loads((OUT/'gate_training.json').read_text()),'test_summary':summary}
    result['secondary_frontier']={n:{'utility_delta':r['net_utility']-baseline['net_utility'],
        'calls_reduction_fraction':1-r['sol_calls']/baseline['sol_calls'],
        'api_cost_reduction_fraction':1-r['api_usd']/baseline['api_usd']} for n,r in summary.items() if n.startswith('gate_') or n=='initial_sol'}
    write(OUT/'findings.json',result)
    write(OUT/'audit.json',{'passed':True,'raw_paths_replayed':len(rows),'raw_turns_replayed':turns,
        'learned_decisions_reproduced':cheap_decisions,'unique_public_requests_verified':len(public_requests),
        'test_configuration_overlap':0,'selection_frozen_before_test':True,'acceptance_violations':0,
        'unresolved_transport_calls_retained':unresolved,'reserved_for_unresolved_usd':reserved})
    labels={'always_sol':'항상 Sol medium','cheap':'작은 학습 정책만','planner':'공개 규칙 계획기','initial_sol':'첫 판단만 Sol',name:'선택적 Sol (개발 선택)'}
    def table(data):
        s='<table><tr><th>방식</th><th>순효용</th><th>합의율</th><th>Sol 호출/회</th><th>총 토큰/회</th><th>API $/회</th><th>질문/회</th><th>신호/회</th></tr>'
        for n,r in data.items():
            s+=f'<tr><td>{html.escape(labels.get(n,n))}</td><td>{r["net_utility"]:.5f}</td><td>{r["agreement"]:.1%}</td><td>{r["sol_calls"]:.3f}</td><td>{r["prompt_tokens"]+r["completion_tokens"]:.1f}</td><td>{r["api_usd"]:.5f}</td><td>{r["queries"]:.3f}</td><td>{r["signals"]:.3f}</td></tr>'
        return s+'</table>'
    allocation={}
    for r in rows:
        if r['split']!='test' or not r['policy'].startswith('gate_'):continue
        for d in r['decisions']:
            key=r['policy']+':'+('accepted' if d['public_state']['accepted'] else 'not_accepted')
            z=allocation.setdefault(key,{'decisions':0.,'sol_calls':0.});z['decisions']+=r['weight']/32;z['sol_calls']+=r['weight']*d['route_to_sol']/32
    write(OUT/'routing_allocation.json',allocation)
    examples=[]
    for r in rows:
        if r['split']=='test' and r['policy'].startswith('gate_') and any(d['route_to_sol'] and d['cheap_action']!=d['selected_action'] for d in r['decisions']):
            examples.append(r)
    write(OUT/'routed_examples.json',examples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for ax,key,xlabel in zip(axes,['sol_calls','api_usd'],['Expected Sol calls / episode','Expected API USD / episode']):
        gate_names=[f'gate_{t}' for t in [.2,.1,.06,.03,.01]]
        ax.plot([summary[n][key] for n in gate_names],[summary[n]['net_utility'] for n in gate_names],'-o',color='#176b87',label='Learned gate thresholds')
        for n in gate_names:
            ax.annotate(n.removeprefix('gate_'),(summary[n][key],summary[n]['net_utility']),xytext=(4,5),textcoords='offset points',fontsize=8)
        for n,color,marker in [('always_sol','#a53c32','s'),('cheap','#555555','x'),('initial_sol','#d28b00','D')]:
            ax.scatter(summary[n][key],summary[n]['net_utility'],color=color,marker=marker,label=n.replace('_',' '),zorder=4)
        ax.axhline(summary['planner']['net_utility'],color='#777777',linestyle=':',label='Known-rule planner')
        ax.set(xlabel=xlabel,ylabel='Expected net utility');ax.grid(alpha=.18)
    axes[0].legend(fontsize=8);fig.suptitle('Exploratory selective-reasoning pilot: 16 numeric settings, 2 fixed sources',fontsize=11)
    fig.savefig(OUT/'utility_cost.png',dpi=180);fig.savefig(OUT/'utility_cost.pdf');plt.close(fig)
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Selective Sol pilot</title>
    <style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#18242c}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #ccd5db;text-align:right;padding:9px}}td:first-child,th:first-child{{text-align:left}}h1,h2{{line-height:1.25}}.note{{background:#eff5f7;padding:18px}}blockquote{{border-left:3px solid #54798a;padding-left:18px}}</style>
    <h1>필요한 판단만 Sol에 맡기는 외교 협상 파일럿</h1>
    <p class="note">항상 Sol 대비 순효용 차이 {result['utility_delta']:+.5f}; 호출 {result['calls_reduction_fraction']:.1%}, 토큰 {result['token_reduction_fraction']:.1%}, API 비용 {result['api_cost_reduction_fraction']:.1%} 감소. 합의율 {hybrid['agreement']:.1%}. 성공·실패를 포함한 고정된 파일럿 결과다.</p>
    <h2>새 수치 조건 평가</h2>{table(summary)}<img src="utility_cost.png" alt="Utility versus Sol calls and API cost" style="width:100%">
    <p>새 수치 설정 16개 × 기존 실제 문서 2개 = 32개 평가 셀. 숨은 유형과 질문 응답 분기를 확률 가중했다. 평가 스타일은 0이며 문서·언어 bank는 이전에 사용했다. 숫자·효용·상대 반응은 합성이다.</p>
    <h2>방법과 개발 선택</h2><p>기존 증류+PPO seed17을 고정했다. 공개 상태의 행동가치를 예측하는 작은 MLP 3개는 기존 학습 데이터 40,706개만 사용했다. 기존 정책 행동의 예측 손실이 임계값을 넘을 때 Sol medium을 호출한다. 배포 시 계획기는 호출하지 않는다. 공개 규칙 계획기 교사라는 추가 자원과 기존 PPO 학습을 사용하므로 같은 학습 비용 비교는 아니다.</p>
    <p>개발 설정 8개 × 문서 2개에서 실제 Sol을 사용했다. 항상 Sol 대비 효용 손실 0.01 이내인 후보 중 호출이 가장 적은 임계값 {selection['threshold']}을 선택하고 새 수치 평가 전에 고정했다. 평가 결과를 보고 바꾸지 않았다. 개발집합에서는 호출 0회 후보가 선택되어, 평가 시작 전 추가 기록으로 모든 기존 임계값의 보조 탐색 곡선과 첫 턴만 Sol을 쓰는 단순 비교도 정했다. 이 보조 결과에서 좋은 임계값이 보이더라도 그것을 개발 선택의 성공으로 바꾸어 보고하지 않는다.</p>{table(selection['summary'])}
    <h2>비용·검사·해석 범위</h2><p>표의 비용은 캐시 적중도 호출로 세는 가상 배포 1회당 기대값이다. 실제 파일럿은 중복 공개 프롬프트를 재사용했고 총 {len(validated)}개 성공 응답에 ${result['actual_new_api_usd']:.6f}, 전체 응답 기준 누적 ${spent:.6f}가 들었다. 전송 중단 요청 비용은 미확정이며 예약 ${reserved:.6f}를 추가로 유지한다. 이를 포함한 보수적 누적 상한은 ${spent+reserved:.6f}이다. 언어 bank 작성비·교사 계산·학습 비용은 표의 배포 비용에 포함하지 않는다. CPU 실행 시간이나 전체 시스템 지연의 감소로 해석하지 않는다.</p>
    <p>개발·평가 경로 {len(rows)}개, {turns}개 전이를 별도 프로세스에서 재생했다. 입력은 공개 상태만 포함하며 실제 숨은 유형은 API에 제공하지 않았다. 요청·토큰·비용·호출 기준·수락 조건을 감사했다.</p>
    <p>수치 설정 단위 paired bootstrap의 효용 차이 95% 구간: [{np.quantile(boots,.025):+.5f}, {np.quantile(boots,.975):+.5f}]. 이는 고정된 두 문서와 단일 캐시 응답에서의 탐색적 수치 변동만 나타낸다. 역사적 일반화·모델 샘플 변동·확정적 비열등성 검증은 아니다. 일반적인 routing 자체를 새 알고리즘이라고 주장하지 않는다.</p>
    <h2>사용한 실제 원문</h2>'''
    for ctx in protocol['test_contexts']:
        page+=f'<blockquote>{html.escape(ctx["source"]["sentence"])}</blockquote><p>{html.escape(ctx["source"]["doc_title"])} · {html.escape(str(ctx["source"]["date"]))}</p>'
    page+='</html>';(OUT/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('gate_training','test_summary')},indent=2),flush=True)


if __name__=='__main__':main()
