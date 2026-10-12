# Diplomatic Negotiation — Track C

**우선순위: Track C 과제 제출과 재현 → COLING 2027 short paper 검토.**

실제 외교 문서를 맥락으로 삼고, 합성 효용·상대 유형·반응 규칙을 명시한 외교 협상 환경입니다. 가격 협상이나 실제 외교 성과 예측 연구가 아닙니다. 이 저장소에는 실행 코드, 제공된 원문과 출처, 고정 언어 bank, 학습 체크포인트, 원시 평가 경로, 표·그림 및 담화 사례가 포함됩니다. 논문 본문은 아직 작성하지 않았습니다.

현재 비공개 연구·과제 저장소입니다. 자료의 외부 재배포 조건을 확정한 공개 데이터 릴리스가 아니며, 교수님·팀원의 접근 권한은 저장소 소유자가 관리해야 합니다. [자료와 이용 범위](DATA_USAGE.md)를 확인하세요.

## 빠른 재현 — API 키와 유료 호출 불필요

실험 환경은 Python **3.13.9**와 [requirements.txt](requirements.txt)에 기록한 패키지입니다. CPU로 재현할 수 있습니다. 체크포인트와 언어 bank도 Git에 포함되어 별도 다운로드가 필요 없습니다.

```bash
git clone https://github.com/yoohogyun/diplomatic-negotiation-track-c.git
cd diplomatic-negotiation-track-c
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -X utf8 -m tools.verify_release
python -X utf8 -m tools.reproduce_release quick
python -X utf8 -m tools.reproduce_release all
```

`quick`은 단위 테스트와 Luna 기본 PPO 저장 경로 24개를 확인합니다. `all`은 여기에 전체 Luna 기록의 보상·가중치 감사, 개선 정책 seed 17의 전 경로, Sol 기본·개선 정책 재현을 더합니다. 각 단계는 별도 Python 프로세스에서 실행하며 HTTP/소켓 연결을 차단합니다. 결과는 `verification/`에 저장하고 보존된 원시 실험 파일을 덮어쓰지 않습니다. 설치는 인터넷이 필요할 수 있지만 재현 명령은 오프라인입니다.

실험에 사용한 패키지 버전의 설치 가능성은 플랫폼별로 다를 수 있습니다. 다른 버전으로 실행하면 해당 버전과 재현 결과를 기록하세요. 본 릴리스 검증은 기존 실험 호스트에서 수행했으며 깨끗한 새 컴퓨터에서의 설치 검증을 뜻하지 않습니다.

## 어디부터 볼 것인가

| 목적 | 파일 |
|---|---|
| 과제 요구사항과 남은 제출 작업 | [ASSIGNMENT.md](ASSIGNMENT.md) |
| 환경·공개 상태·수치 규칙 | [study/env.py](study/env.py), [pilot/env.py](pilot/env.py) |
| 실제 원문과 실험 맥락 연결 | [study_results/contexts.json](study_results/contexts.json), `data/source/` |
| 필수 Random / Heuristic / Bandit / PPO | [study_results/summary.csv](study_results/summary.csv), [Luna 보고서](study_results/report.html) |
| 정보 활용 개선 정책 | [improvement/README.md](improvement/README.md), [결과](improvement_results/findings.json) |
| 직접 Luna/Sol 동일 조건 비교 | [sol_results/direct_comparison.csv](sol_results/direct_comparison.csv), [matched controls](sol_results/matched_policy_controls.csv) |
| 작은 모델 + 전략 제어 비교 | [small_model_results/findings.json](small_model_results/findings.json) |
| 담화 본문에 쓸 자료 | [담화 경로](study_results/discourse_walkthroughs.txt), [같은 조건의 네 정책](small_model_results/outline_matched_discourse.json) |
| 원시 로그·체크포인트 | 각 `*_results/raw/`, `*_results/models/` |
| 실행별 API 요청·응답 | 각 `*_results/api/` — 인증 헤더/키 제외 |
| 연구 주장과 보완 항목 | [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md) |

HTML 보고서는 로컬 브라우저에서 여세요. 하위 README·보고서·프로토콜은 단계별 역사적 기록이라 당시의 비용/완료 상태를 담고 있습니다. 최신 저장소 사용법과 범위는 이 README 및 `provenance/RELEASE_NOTES.md`를 우선합니다.

## 주요 결과: 평가 집합을 구분해서 읽기

필수 본 실험은 4개 평가 문서 × 64개 수치 조건, 3개 발화 스타일입니다. 순효용은 Random **−0.1457**, Bandit **0.1183**, Heuristic **0.4190**, PPO **0.3581**(3-seed SD 0.0181), 공개 믿음 계획기 **0.4882**입니다. PPO가 Heuristic보다 우수하다는 결과는 아닙니다.

아래는 별도의 **동일 64개 문서–수치 조건, 스타일 0, 같은 Luna 실행 bank** 비교입니다. 위 전체 평가나 개선 실험의 새 수치 조건 집합과 혼합하면 안 됩니다.

| 전략 선택기 | 순효용 |
|---|---:|
| Luna 직접 선택, reasoning none | −0.367745 |
| Sol 직접 선택, reasoning none | −0.367279 |
| Sol 직접 선택, reasoning medium | 0.514012 |
| Heuristic + Luna 실행 bank | 0.469719 |
| 계획기 증류 후 PPO + Luna 실행 bank, 3 seeds | 0.475318 |
| 정확한 공개 믿음 계획기 + Luna 실행 bank | 0.533428 |

계획기는 알려진 합성 전이 규칙과 고정 발화의 해석을 이용합니다. 실제 숨은 유형은 입력받지 않습니다. 학습 정책은 Sol medium보다 낮고, 강한 규칙 정책과 차이도 작습니다. 로컬 제어기는 전략 선택 API 호출을 없애지만, 언어 bank 생성·해석과 교사/학습 비용까지 없는 것은 아닙니다. `none`과 `medium`은 추론 및 출력 토큰 예산도 다릅니다.

## 실험 상태

- Luna 본 실험, 필수 ablation, 규칙 민감도, 정보 활용 개선, Sol 반복 실험은 저장 결과가 있습니다.
- 문서는 12개/6개 사건군이고 평가 사건군은 시리아·이란 **2개**입니다. 수천 경로를 독립적인 역사적 협상으로 세지 않습니다.
- Luna 언어 bank의 생성–해석 일치 379/396과 합성 probe 180/180은 사람 정답 기준 정확도가 아닙니다. 오류는 전이에 그대로 반영했습니다.
- `routing/`과 `routing_results/`는 **중단된 부수 실험**입니다. 개발은 완료했지만 테스트는 32개 중 17개만 완료했습니다. 완결된 테스트 결과로 사용하지 않습니다.
- 논문은 `paper/main.pdf`와 편집 가능한 TeX 소스에 완성되어 있습니다. 본문 4쪽, 담화 분석 6개 에피소드, 필수 실험 진단 부록을 포함합니다. Overleaf용 최소 소스는 `submission/overleaf_source/`, 업로드용 ZIP은 `submission/overleaf_upload.zip`입니다.
- 사용자가 직접 마무리할 항목은 `paper/authors.tex`의 저자 정보, `submission/TEAM_RECORD.txt`의 실제 역할·다른 팀원 재현 기록, 비공개 저장소의 평가자 접근 권한입니다. 별도의 62개 사람 라벨링이나 학회 accept는 과제 필수 조건이 아닙니다.

## 추가 학습 또는 유료 재생성

기존 결과를 보존한 별도 복사본에서만 진행하세요. [REPRODUCING.md](REPRODUCING.md)는 오프라인 재현, 재학습과 역사적 유료 호출 코드를 구분합니다. 이 릴리스에는 `.env`, 개인 비용 원장, 교수님 배포 PDF, 임시 파일 및 설치된 의존성이 들어 있지 않습니다. 현재 릴리스 작업은 추가 모델 호출을 수행하지 않았습니다.

