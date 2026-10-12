Overleaf 공동작업용 최소 프로젝트

업로드
- 새 프로젝트: Overleaf의 New Project → Upload Project에서 overleaf_upload.zip을 선택합니다.
- 기존 공동 프로젝트: overleaf_source 폴더 안 파일과 figures 하위폴더를 프로젝트 최상위에 올립니다. 폴더 구조를 유지하고 같은 이름의 예전 파일은 교체합니다.
- Main document: main.tex / Compiler: pdfLaTeX / bibliography: BibTeX (자동 처리).
- main.pdf, 코드, raw 로그, 체크포인트, .env, PPTX는 업로드할 필요가 없습니다. figures/*.pdf는 본문에 들어가는 그림이므로 꼭 올립니다.

사용자가 작성할 곳
1. authors.tex: 실제 이름/소속/학번. pdfLaTeX에서는 영문 표기를 사용합니다.
2. main.tex 2행: 실명 과제 제출 시 [review]를 [final]로 변경합니다. 익명 검토 초안은 review를 유지합니다.
3. 역할·실제 팀원 재현 기록은 별도 제출 폴더의 TEAM_RECORD.txt에 작성합니다. 논문 본문에 억지로 넣을 필요는 없습니다.
4. 이름/내용 수정 후 결론까지4쪽인지 다시 확인합니다. 원본 ACL 여백/폰트를 줄이지 마세요.

편집 위치
main.tex: 제목, 초록, Introduction, Related Work
methods_results.tex: Method, Experimental Setup, Results, 본문 담화분석
discussion_conclusion.tex: Discussion, Conclusion
limitations.tex / ethics.tex: 별도 한계 / 윤리
appendix_experiments.tex: 부록 1쪽 — 핵심 환경·학습 설정과 담화6사례
appendix_assignment.tex: 부록 2쪽 — 전이행렬·학습곡선·성과/개입/민감도 표
assignment_*_table.tex: 저장결과에서 생성한 표
references.bib: 참고문헌
figures/: 컴파일용 벡터 PDF 그림5개

티저 수정: 별도 제공 teaser_editable_v2.pptx를 PowerPoint에서 편집한 후 PDF로 내보내 figures/teaser.pdf만 교체합니다.
논문용 수치 수정은 반드시 저장 실험 결과와 대조하세요.
현재 출력: 본문4쪽 + 한계·윤리·참고문헌2쪽 + 부록2쪽 = 전체8쪽.
긴 상세 설명은 GitHub paper/supplementary_appendix_full.tex에 별도 보존했습니다.

공식 사용법
https://www.overleaf.com/learn/latex/Kb/Uploading_a_project
https://docs.overleaf.com/managing-projects-and-files/adding-files-to-a-project/uploading-files-to-a-project
https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document
