"""Report the follow-up honestly, with paired gains and all selected-policy ablations."""
import json,gzip,collections,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improvement import OUT
from pilot.live import write

def main():
    selection=json.loads((OUT/'selection.json').read_text());chosen=selection['selected']
    rows=json.loads((OUT/'seed_metrics.json').read_text());df=pd.DataFrame(rows);df.to_csv(OUT/'seed_metrics.csv',index=False)
    cols=['net_utility','agreement','queries','signals','cost','length','acceptance_violations']
    means=df.groupby('policy')[cols].mean();sd=df.groupby('policy')[cols].std();means.to_csv(OUT/'summary.csv');sd.to_csv(OUT/'seed_sd.csv')
    cases=[]
    for path in (OUT/'case_metrics').glob('test_*.json'):cases+=json.loads(path.read_text())
    c=pd.DataFrame(cases);c.to_csv(OUT/'case_metrics.csv',index=False)
    paired=c.groupby(['policy','context','condition'])[cols].mean().reset_index()
    wide=paired.pivot(index=['context','condition'],columns='policy',values='net_utility')
    info=wide.Planner-wide.Planner_no_query;valuable=info>1e-7
    metrics={'selected':chosen,'mean_gain_over_base':float((wide[chosen]-wide['base']).mean()),
      'mean_gain_over_heuristic':float((wide[chosen]-wide.Heuristic).mean()),
      'selected_minus_no_query':float((wide[chosen]-wide[chosen+'_no_query']).mean()),
      'selected_minus_no_signal':float((wide[chosen]-wide[chosen+'_no_signal']).mean()),
      'selected_minus_masked_belief':float((wide[chosen]-wide[chosen+'_masked_belief']).mean()),
      'positive_query_value_cases':int(valuable.sum()),'all_context_settings':len(wide),
      'query_removal_loss_on_positive_value_cases':float((wide[chosen]-wide[chosen+'_no_query'])[valuable].mean()),
      'query_removal_loss_on_other_cases':float((wide[chosen]-wide[chosen+'_no_query'])[~valuable].mean()),
      'gap_to_planner':float((wide.Planner-wide[chosen]).mean()),
      'no_new_api_calls':True,'main_results_unchanged':True}
    wide['planner_question_value']=info;wide['selected_query_removal_loss']=wide[chosen]-wide[chosen+'_no_query'];wide.to_csv(OUT/'paired_comparisons.csv')
    # Every path: cost consistency, branch weight, legal agreement and continuity.
    n=0;turns=0;weights=collections.defaultdict(float);examples=[]
    for p in sorted((OUT/'raw').glob('*.gz')):
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);tr=r['trajectory'];n+=1;turns+=len(tr)
                assert abs(sum(t['common_reward'] for t in tr)-r['net_utility'])<1e-8
                assert abs(sum(sum(t['costs'].values()) for t in tr)-r['cost'])<1e-8
                assert r['acceptance_violations']==0
                for a,b in zip(tr,tr[1:]):assert a['after']==b['before']
                weights[(p.name,r['context'],r['condition'])]+=r['weight']
                if p.name.startswith('test_') and r['policy']==chosen and r['agreement'] and r['queries']>0 and len(examples)<3 and not any((x['context'],x['condition'])==(r['context'],r['condition']) for x in examples):examples.append(r)
    assert all(abs(w-1)<1e-8 for w in weights.values())
    metrics.update(audited_episodes=n,audited_turns=turns,integrity_checks_passed=True)
    write(OUT/'findings.json',metrics);write(OUT/'successful_query_examples.json',examples)
    names=['base','Heuristic','distilled','distilled_ppo','Planner']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,col,title in zip(axes,['net_utility','queries','signals'],['Net utility','Private questions / episode','Signals / episode']):
        vals=means.loc[names,col];ax.bar(range(len(names)),vals,color=['#888888','#6a939b','#246f73','#66a7a7','#b8c6c6'])
        for i,name in enumerate(names):
            if pd.notna(sd.loc[name,col]):ax.errorbar(i,vals.iloc[i],yerr=sd.loc[name,col],color='black',capsize=3)
        ax.set(title=title,xticks=range(5),xticklabels=['Base PPO','Heuristic','Distilled','Distilled\n+ PPO','Planner']);ax.tick_params(axis='x',rotation=30)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ab=[chosen,chosen+'_no_query',chosen+'_no_signal',chosen+'_masked_belief']
    ax.bar(range(4),means.loc[ab,'net_utility'],color='#347e84');ax.set(xticks=range(4),xticklabels=['Selected','Query removed','Signal removed','Belief masked'],ylabel='Common net utility',title='Deployment interventions; no retraining');fig.tight_layout();fig.savefig(OUT/'interventions.png',dpi=180);plt.close(fig)
    status='개선 확인' if metrics['mean_gain_over_heuristic']>0 and metrics['mean_gain_over_base']>0 else '개선 범위 제한: 아래 비교 참조'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>정보를 활용하는 정책 실험</title><style>body{{max-width:1150px;margin:35px auto;font:16px/1.65 system-ui;padding:0 22px;color:#18383d}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f1f5f5;padding:15px}}</style><h1>정보를 활용하는 정책: {status}</h1><p>기존 결과를 확인한 뒤 사용자의 요청으로 시작한 후속 실험입니다. 개발 조건에서 선택한 방법: <b>{chosen}</b>. 기존 기록은 보존했습니다. 새로운 수치 조건 64개 × 기존 평가 원문 맥락 4개를 사용했습니다. 새로운 역사적 사건 검증이나 연구자 비노출 자료는 아닙니다.</p><p>공개정보 계획기의 학습 조건 행동가치를 작은 정책망에 증류했습니다. 학습·추론 입력에 실제 상대 유형은 없습니다. 새 정책은 평가 때 계획기를 호출하지 않지만, 학습에 알려진 전이 규칙과 추가 교사 계산을 이용했습니다. 동일 예산의 순수 모델 프리 RL 개선이나 새로운 알고리즘 발명으로 해석하지 않습니다. Luna 발화·해석 bank는 그대로이며 신규 API 호출은 없습니다.</p><img src="comparison.png"><p>오차막대는 학습 seed 3개의 표준편차입니다. 규칙 정책과 계획기는 고정 정책입니다.</p>{means.round(5).to_html()}<h2>개발 조건 선택 기록</h2>{pd.DataFrame(selection['dev'])[['method','seed','net_utility','queries','signals']].round(5).to_html(index=False)}<h2>정보 활용이 성과에 기여했는가</h2><img src="interventions.png"><pre>{html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))}</pre><p>질문·신호 제거는 해당 행동을 제외하고 다음으로 높은 학습된 행동 점수를 선택합니다. 믿음 가림은 사후확률을 초기 사전확률로 바꾸며 다른 공개 상태는 남깁니다. 별도 재학습한 제거 실험과 다릅니다. 질문 가치가 양수인 조건은 공개정보 계획기와 질문 제한 계획기의 차이로 정의했습니다.</p><h2>보존한 자료</h2><p>프로토콜, 교사 데이터와 학습 곡선, 6개 체크포인트, 모든 평가 경로, seed별 결과, 대응 조건 비교와 성공 질문 사례를 같은 폴더에 저장했습니다. 논문은 작성하지 않았습니다.</p></html>'''
    narrative=f'''<h2>결과 해석</h2><p>선택된 정책의 순효용은 {means.loc[chosen,'net_utility']:.5f} (seed SD {sd.loc[chosen,'net_utility']:.5f})입니다. 같은 새 평가에서 기존 PPO 대비 {metrics['mean_gain_over_base']:+.5f}, 강한 규칙 정책 대비 {metrics['mean_gain_over_heuristic']:+.5f}였습니다. 학습된 정책에서 신호를 막으면 평균 {metrics['selected_minus_no_signal']:.5f}, 갱신된 믿음을 가리면 {metrics['selected_minus_masked_belief']:.5f}만큼 순효용이 낮아졌습니다. 따라서 개선은 주로 선택적 신호와 상대 응답에서 얻은 정보의 활용으로 해석합니다.</p><p>유료 질문의 활용은 제한적입니다. 선택된 세 seed 중 {int(((df.policy==chosen)&(df.queries>0)).sum())}개만 질문을 사용했고, 질문 제거의 평균 순효용 손실은 {metrics['selected_minus_no_query']:.5f}였습니다. 이 실험으로 질문 활용 문제가 모든 seed에서 해결됐다고 주장하지 않습니다. 제거 실험은 정책 입력/출력의 교란이며 완전한 인과 분해나 재학습된 대체 정책 비교는 아닙니다.</p>'''
    page=page.replace('<h2>개발 조건 선택 기록</h2>',narrative+'<h2>개발 조건 선택 기록</h2>',1)
    if json.loads((OUT/'protocol.json').read_text()).get('model')=='openai/gpt-5.6-sol':
        page=page.replace('Luna 발화·해석 bank','Sol 발화·해석 bank')
    (OUT/'report.html').write_text(page,encoding='utf-8');print(json.dumps(metrics),flush=True)

if __name__=='__main__':main()
