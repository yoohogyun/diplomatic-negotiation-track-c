"""Matched model comparison and explicit evidence for/against the earlier storyline."""
import json,gzip,collections,html,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solstudy import OUT,ROOT,IMPROVED
from study import OUT as LUNA
from pilot.live import write,LEDGER

def matched_selected(label,path):
    import torch
    from stable_baselines3 import PPO
    from study.env import LunaEnv
    from study.evaluate import run_episode
    from pilot.env import Config
    torch.set_num_threads(1)
    directory=ROOT/'improvement_results' if label=='Luna' else IMPROVED
    selected=json.loads((directory/'selection.json').read_text())['selected']
    bank=json.loads((path/'language_bank.json').read_text(encoding='utf-8'));env=LunaEnv('test',bank=bank)
    configs=[Config(**c) for c in json.loads((path/'numeric_conditions.json').read_text())['test'][:16]]
    metrics=[]
    for seed in [17,29,43]:
        model=PPO.load(directory/f'models/{selected}_{seed}',device='cpu');records=[]
        for ctx in env.contexts:
            for ci,cfg in enumerate(configs):
                for hard,pt in [(False,1-cfg.prior),(True,cfg.prior)]:
                    ph=cfg.reliability if hard else 1-cfg.reliability
                    for u,py in [(0.,ph),(.999999,1-ph)]:
                        if pt*py<=0:continue
                        z=run_episode(env,lambda o,e:int(model.predict(o,deterministic=True)[0]),cfg,ctx,0,hard,u,123)
                        records.append(dict(policy=selected,seed=seed,event_group=ctx['event_group'],context=ctx['id'],condition=ci,weight=pt*py,**z))
        file=OUT/f'matched_selected_cases/{label}_{seed}.json';write(file,records)
        row=aggregate([file],label+'_selected_'+str(seed));row.update(bank=label,policy='Selected_'+selected+'_'+str(seed));metrics.append(row)
    return metrics

def aggregate(paths,label):
    cases=collections.defaultdict(lambda:collections.defaultdict(float));counts=0
    for p in paths:
        for r in json.loads(p.read_text(encoding='utf-8')):
            tr=r['trajectory'];assert r['acceptance_violations']==0
            assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
            assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
            for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
            repeated=any(a['response']=='reject' and a['meaning']['kind']=='offer' and a['meaning']==b['meaning'] for a,b in zip(tr,tr[1:]))
            row=cases[(r['event_group'],r['context'],r['condition'])];w=r['weight'];row['weight']+=w
            for k in ['net_utility','agreement','queries','signals','cost','length']:row[k]+=w*r[k]
            row['repeated_rejected_offer']+=w*repeated;counts+=1
    assert cases and all(abs(r['weight']-1)<1e-8 for r in cases.values())
    df=pd.DataFrame([dict(group=k[0],context=k[1],condition=k[2],**v) for k,v in cases.items()])
    df.to_csv(OUT/(label+'_case_metrics.csv'),index=False)
    metrics=df.groupby('group')[['net_utility','agreement','queries','signals','cost','length','repeated_rejected_offer']].mean().mean().to_dict()
    return dict(policy=label,cases=len(cases),raw_paths=counts,**metrics)

def main():
    direct=[aggregate(sorted((LUNA/'luna_policy_cases').glob('*.json')),'Luna_none_Luna_bank')]
    for mode in ['none_luna','medium_luna','none_sol']:
        assert (OUT/f'direct_{mode}_complete.json').exists(),mode+' incomplete'
        direct.append(aggregate(sorted((OUT/f'direct_{mode}_cases').glob('*.json')),'Sol_'+mode))
    direct_df=pd.DataFrame(direct);direct_df.to_csv(OUT/'direct_comparison.csv',index=False)
    assert all(x['cases']==64 for x in direct)
    mainrows=[]
    for label,path in [('Luna',LUNA),('Sol',OUT)]:
        for r in json.loads((path/'seed_metrics.json').read_text()):mainrows.append(dict(model=label,**r))
    md=pd.DataFrame(mainrows);md.to_csv(OUT/'all_seed_metrics.csv',index=False)
    cols=['net_utility','agreement','queries','signals','cost','length']
    ms=md.groupby(['model','policy'])[cols].mean();ms.to_csv(OUT/'main_comparison.csv')
    imp=[]
    for label,path in [('Luna',ROOT/'improvement_results'),('Sol',IMPROVED)]:
        chosen=json.loads((path/'selection.json').read_text())['selected']
        for r in json.loads((path/'seed_metrics.json').read_text()):imp.append(dict(model=label,selected_method=chosen,**r))
    ip=pd.DataFrame(imp);ip.to_csv(OUT/'improvement_all_seeds.csv',index=False)
    ips=ip.groupby(['model','policy'])[cols].mean();ips.to_csv(OUT/'improvement_comparison.csv')
    languages=[]
    for label,path in [('Luna',LUNA),('Sol',OUT)]:languages.append(dict(model=label,**json.loads((path/'language_summary.json').read_text())))
    ld=pd.DataFrame(languages);ld.to_csv(OUT/'language_comparison.csv',index=False)
    # Match policy controls to the original first-16-config, style-0 direct comparison.
    matched=[]
    for label,path in [('Luna',LUNA),('Sol',OUT)]:
        for name in ['Heuristic_None','Planner_None','PPO_base_17','PPO_base_29','PPO_base_43']:
            cs=collections.defaultdict(lambda:collections.defaultdict(float))
            with gzip.open(path/'raw'/('test_'+name+'.jsonl.gz'),'rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line)
                    if r['condition']>=16 or r['variant']!=0:continue
                    z=cs[(r['context'],r['condition'])]
                    for k in cols:z[k]+=3*r['weight']*r[k]
            matched.append(dict(bank=label,policy=name,**pd.DataFrame(cs.values()).mean().to_dict()))
        matched+=matched_selected(label,path)
    matched_df=pd.DataFrame(matched);matched_df.to_csv(OUT/'matched_policy_controls.csv',index=False)
    # Audit all local learned policy and direct trajectories; retain scope boundaries.
    n=0;turns=0
    for folder in [OUT,IMPROVED]:
        for p in (folder/'raw').glob('*.gz'):
            weights=collections.defaultdict(float)
            with gzip.open(p,'rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);tr=r['trajectory'];n+=1;turns+=len(tr)
                    assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
                    assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
                    assert r['acceptance_violations']==0
                    for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
                    weights[(r['context'],r['condition'])]+=r['weight']
            assert all(abs(w-1)<1e-8 for w in weights.values())
    ledger=json.loads(LEDGER.read_text());records=ledger['records'];sol=[r for r in records if r['tag'].startswith('sol_')]
    assert all(r['status']=='validated' and 'cost' in r for r in sol)
    raw_cost=0;usage=collections.defaultdict(lambda:collections.defaultdict(float))
    for p in (OUT/'api').glob('*.raw.json'):
        raw=json.loads(p.read_text());assert raw['model'].startswith('openai/gpt-5.6-sol');raw_cost+=raw['usage']['cost']
        role='direct_medium' if p.name.startswith('sol_medium_policy') else ('direct_none' if p.name.startswith('sol_none_policy') else 'language')
        u=raw['usage'];usage[role]['calls']+=1;usage[role]['cost']+=u['cost']
        for field in ['prompt_tokens','completion_tokens']:usage[role][field]+=u.get(field,0)
        usage[role]['reasoning_tokens']+=u.get('completion_tokens_details',{}).get('reasoning_tokens',0)
    assert abs(raw_cost-sum(r['cost'] for r in sol))<1e-8
    audit={'learned_policy_raw_paths':n,'learned_policy_raw_turns':turns,'direct_raw_paths':sum(r['raw_paths'] for r in direct[1:]),
      'checks_passed':True,'sol_calls':len(sol),'sol_cost':sum(r['cost'] for r in sol),'all_authorized_cost':sum(r.get('cost',0) for r in records),
      'unresolved_calls':sum('cost' not in r for r in records)}
    audit['account_usage_increment']=ledger['final_key_usage']['usage']-ledger['baseline_key_usage']['usage']
    audit['response_cost_minus_account_increment']=audit['all_authorized_cost']-audit['account_usage_increment']
    audit['account_counter_matches_response_sum']=abs(audit['response_cost_minus_account_increment'])<1e-6
    audit['budget_accounting']='Use full response cost sum conservatively; retain any account-counter difference without assuming its cause.'
    write(OUT/'audit.json',audit)
    usage_df=pd.DataFrame([dict(role=k,**v) for k,v in usage.items()]);usage_df.to_csv(OUT/'inference_usage.csv',index=False)
    dd={r['policy']:r for r in direct}
    matched_selected_mean=float(matched_df[(matched_df.bank=='Luna')&matched_df.policy.str.startswith('Selected_')].net_utility.mean())
    matched_sol_selected_mean=float(matched_df[(matched_df.bank=='Sol')&matched_df.policy.str.startswith('Selected_')].net_utility.mean())
    luna_bank=json.loads((LUNA/'language_bank.json').read_text(encoding='utf-8'));sol_bank={r['id']:r for r in json.loads((OUT/'language_bank.json').read_text(encoding='utf-8'))}
    semantic_differences=[r['id'] for r in luna_bank if r['split']=='test' and r['variant']==0 and r['meaning']!=sol_bank[r['id']]['meaning']]
    matched_heuristic=float(matched_df[(matched_df.bank=='Luna')&(matched_df.policy=='Heuristic_None')].net_utility.iloc[0])
    conclusions={'model_only_delta_none':dd['Sol_none_luna']['net_utility']-dd['Luna_none_Luna_bank']['net_utility'],
      'sol_medium_minus_none':dd['Sol_medium_luna']['net_utility']-dd['Sol_none_luna']['net_utility'],
      'sol_medium_minus_matched_selected_luna_mean':dd['Sol_medium_luna']['net_utility']-matched_selected_mean,
      'sol_medium_minus_matched_selected_sol_mean':dd['Sol_medium_luna']['net_utility']-matched_sol_selected_mean,
      'matched_luna_selected_mean':matched_selected_mean,'matched_sol_selected_mean':matched_sol_selected_mean,
      'test_style_zero_bank_semantic_differences':semantic_differences,
      'sol_medium_minus_matched_heuristic':dd['Sol_medium_luna']['net_utility']-matched_heuristic,
      'comparison_limit':'Medium changes reasoning and output budget; same prompts and fixed execution bank. Constructed PPO failures are not general LLM failures.'}
    write(OUT/'conclusions.json',conclusions)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    display=['Luna / none','Sol / none','Sol / medium','Sol / own bank']
    for ax,col,title in zip(axes,['net_utility','agreement'],['Matched net utility','Agreement rate']):
        ax.bar(range(4),direct_df[col],color=['#888888','#357f87','#1b565c','#a5bfc2']);ax.set(xticks=range(4),xticklabels=display,title=title);ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(OUT/'direct_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,['Luna','Sol']):
        p=ips.loc[model];chosen=json.loads(((ROOT/'improvement_results') if model=='Luna' else IMPROVED).joinpath('selection.json').read_text())['selected']
        names=['base','Heuristic',chosen,'Planner'];ax.bar(range(4),p.loc[names,'net_utility'],color=['#888','#7aa0a5','#246f73','#b8c6c6']);ax.set(title=model+' execution bank',xticks=range(4),xticklabels=['PPO','Heuristic','Selected method','Planner'],ylabel='Follow-up net utility');ax.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(OUT/'improvement_comparison.png',dpi=180);plt.close(fig)
    tablecols=['policy','net_utility','agreement','queries','signals','repeated_rejected_offer']
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Sol 모델 확장 실험</title><style>body{max-width:1150px;margin:35px auto;padding:0 24px;font:16px/1.6 system-ui;color:#16393f}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;background:#f0f5f5;padding:16px}</style><h1>GPT-5.6 Sol 모델 확장 결과</h1><p>사용자의 요청으로 기존 Luna 조건을 Sol에서 반복했습니다. PPO 자체는 기존 알고리즘이며 이 협상 환경의 PPO 선택기와 증류 기반 시스템은 우리가 구성한 비교 방법입니다. 강한 LLM의 문제를 PPO의 약점으로 대신 주장하지 않도록 직접 모델 비교를 우선합니다.</p><h2>같은 조건에서 직접 행동 선택</h2><p>64개 조건, 같은 프롬프트, 스타일 0. 첫 세 행은 동일한 Luna 실행 bank입니다. Sol none은 Luna와 같은 추론 설정이며, Sol medium은 추론을 켜고 출력·추론 합계 상한을 4,096 tokens로 높인 별도 계산량 조건입니다. 마지막 행만 Sol 실행 bank를 사용합니다. 모델당 상태별 한 응답을 캐시했으므로 모델 표본 간 변동성은 측정하지 않았습니다.</p><img src="direct_comparison.png">'''+direct_df[tablecols].round(5).to_html(index=False)+'''<h3>직접 모델과 같은 하위 조건의 정책 비교</h3>'''+matched_df.round(5).to_html(index=False)+'''<h2>언어 검사</h2>'''+ld.drop(columns=['note']).to_html(index=False)+'''<p>생성 모델과 해석 모델이 같을 때의 일치율입니다. 사람 정답 정확도가 아니며 모든 불일치를 전이에 유지했습니다.</p><h2>전체 원래 평가: 모델별 학습 정책</h2>'''+ms.round(5).to_html()+'''<h2>개선 방법의 모델별 반복</h2><img src="improvement_comparison.png">'''+ips.round(5).to_html()+'''<p>동일한 학습량·3 seeds·개발 조건 선택·후속 수치 조건을 사용했습니다. 모델 변경으로 해석 bank의 상태 전이가 달라질 수 있어, 직접 모델 선택 효과와 위 전체 시스템 결과는 구분해야 합니다. 알려진 규칙을 이용한 교사 계산이 추가되었으므로 동일 자원 model-free 학습 비교가 아닙니다.</p><h2>검증과 비용</h2><pre>'''+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'''</pre><p>실제 외교 문서에 근거한 합성 환경입니다. 두 평가 사건군과 원문은 이미 노출된 자료이며, 모델 확장 결과를 새로운 역사적 검증으로 세지 않습니다. 논문은 작성하지 않았습니다. <a href="https://developers.openai.com/api/docs/models/gpt-5.6-sol">공식 Sol 모델 문서</a>와 실행 직전 OpenRouter endpoint metadata를 확인했고 실제 청구액은 원시 응답·공유 원장에 보존했습니다.</p></html>'''
    narrative=f'''<h2>이 비교가 바꾸는 결론</h2><p>같은 추론 none 조건에서 Sol-Luna 순효용 차이는 {conclusions['model_only_delta_none']:+.5f}입니다. Sol의 추론을 medium으로 켜고 추론/출력 예산을 늘리면 none 대비 {conclusions['sol_medium_minus_none']:+.5f}였습니다. 같은 Luna bank와 64개 조건에서 Sol medium은 기존 개선 정책 3-seed 평균 대비 {conclusions['sol_medium_minus_matched_selected_luna_mean']:+.5f}, 강한 규칙 정책 대비 {conclusions['sol_medium_minus_matched_heuristic']:+.5f}입니다. 따라서 모델 계열만 바꾼 결과와 추론 계산량을 바꾼 결과를 구분해야 합니다.</p><p>기존 PPO 비교는 우리가 구성한 학습 정책의 결과입니다. 직접 Sol의 성능이 높다면 앞선 실패를 강한 LLM 전반의 정보 활용 한계로 확장할 근거가 없습니다. 증류 정책의 개선은 별도로 관측할 수 있지만, 그것만으로 강한 추론 모델보다 우월하다고 주장할 수 없습니다.</p>'''
    page=page.replace('<h2>언어 검사</h2>',narrative+'<h2>언어 검사</h2>',1)
    page=page.replace('<h2>언어 검사</h2>',f'<p>Sol bank에서 재학습한 개선 정책도 같은 64개 조건에서 평균 {matched_sol_selected_mean:.5f}였고, Sol medium 직접 선택과의 차이는 {conclusions["sol_medium_minus_matched_selected_sol_mean"]:+.5f}입니다. 이 비교에 쓰인 모든 평가 맥락·스타일 0 행동의 bank 해석 차이는 {len(semantic_differences)}개입니다.</p><h2>언어 검사</h2>',1)
    page=page.replace('<h2>검증과 비용</h2>','<h2>실제 추론 사용량</h2>'+usage_df.round(6).to_html(index=False)+'<p>캐시된 공개 상태별 실제 API 청구액이며 온라인 배포 비용이나 교사 학습 계산량 전체를 뜻하지 않습니다.</p><h2>검증과 비용</h2>',1)
    (OUT/'report.html').write_text(page,encoding='utf-8')
    write(OUT/'completion.json',{'complete':True,'paper_written':False,'direct':direct,'audit':audit})
    print(json.dumps({'direct':direct,'audit':audit}),flush=True)

if __name__=='__main__':main()
