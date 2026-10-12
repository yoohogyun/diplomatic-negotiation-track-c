# Track C 제출 기준

교수님 과제가 최우선이다. 아래는 제공된 가이드와 실제 산출물을 대조한 요약이며, 가이드 원문을 재배포하지 않는다.

| 항목 | 저장소 증거 / 상태 |
|---|---|
| 외교 협상 POMDP, 숨은 유형·공개 관측 분리, 양측 수락·비용 | `pilot/env.py`, `study/env.py`, 환경 테스트 |
| 6개 담화 기능과 의미에 따른 전이 | 11개 이산 행동, `study_results/language_bank.json`, 의미 probe |
| 실제 외교 자료와 10개 이상 환경 검증 사례 | 12개 문서 맥락, 유형별 24개 경로; `source_case_validation.json` |
| Random, 강한 Heuristic, Bandit, PPO | `study_results/summary.csv`, `models/`, `raw/` |
| Bandit/PPO 기본 3 seeds | 17, 29, 43; 각 102,400 상호작용 |
| 할인율·entropy·보상 ablation, 정보·규칙 민감도 | `protocol.json`, `sensitivity_metrics.json`, 관련 그림 |
| 효용·비용·합의·결렬·길이·전이·학습 곡선 | `study_results/` 표, raw, training_logs, 그림 |
| 최소 6개 에피소드의 담화 분석 | 본문 Figure 3 및 분석, Appendix C의 실제 합성 발화 인용을 포함한 6개 사례; Appendix D의 행동 전이 해석 |
| GitHub 코드·자료·실행 절차 | 이 저장소와 오프라인 검증 명령; 비공개이므로 평가자 접근 권한 필요 |
| 본문 4쪽 및 Limitations, 편집 가능한 논문 소스 | `paper/main.pdf`, `paper/*.tex`, Overleaf ZIP 완료. 저자 정보는 사용자가 기입 |
| 실제 다른 팀원의 재현과 기여도 | `submission/TEAM_RECORD.txt`에 실제 수행 후 사용자가 기입. 자동 재현과 구별 |

PPO가 1등일 필요는 없다. 고정 템플릿/캐시 언어 모듈과 Luna none도 허용된다. 계획기는 추가 비교이며 필수 4정책을 대체하지 않는다. 실제 학회 투고·게재가 과제 완료 조건은 아니다. 20턴 담화 검토는 권장이고, 우리가 만든 62개 라벨링 표는 추가 연구용이다.

## 팀원 재현 기록

새 Python 프로세스로 체크포인트를 재현한 자동 검증과 실제 다른 팀원의 실행을 구분한다. 수행 전 빈칸을 채우지 않는다.

```text
실행자:
날짜 / OS / Python / 패키지 버전:
Git commit:
실행 명령:
verification 결과 파일:
일치 여부 / 실패 원인:
실제 기여:
```

