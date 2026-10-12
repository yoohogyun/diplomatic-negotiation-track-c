# Diplomatic Negotiation — Track C

**Priority: Track C course submission and reproducibility, followed by consideration for a COLING 2027 short paper.**

This repository studies strategic control for **GPT-5.6 Luna** in diplomatic negotiation, with **GPT-5.6 Sol** as a stronger-model comparison. Real diplomatic documents ground the scenarios and utterances. Utilities, hidden opponent types, and response rules are explicitly synthetic. The repository includes source provenance, runnable code, frozen language banks, trained checkpoints, raw trajectories, tables, figures, discourse cases, and the manuscript.

Start with the [paper PDF](paper/main.pdf), [course requirement map](ASSIGNMENT.md), or [reproduction guide](REPRODUCING.md). The [Overleaf ZIP](submission/overleaf_upload.zip) contains the minimal editable project. The manuscript has four pages through Limitations, two reference pages, and two appendix pages, with 24 cited works. The full-width teaser appears above the abstract.

This course/research repository is currently public on GitHub. Redistribution rights for the supplied sources have not been established for a public dataset release; see [data usage](DATA_USAGE.md).

## Quick reproduction — no model key or paid calls

The experiment host used Python **3.13.9** and the versions in [requirements.txt](requirements.txt). Replay runs on CPU. Checkpoints and language banks are included in Git.

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

`quick` runs unit tests and replays 24 saved Luna base-PPO trajectories. `all` also audits rewards and branch weights across the Luna archive and replays improved Luna seed 17 and the base/improved Sol policies. Each stage runs in a separate Python process with HTTP/socket access blocked. Outputs go to `verification/`, preserving the original experiment files. Installing dependencies may require internet access; replay itself is offline.

Package availability may vary by platform. Record versions and outcomes if using a different environment. Existing verification was performed on the experiment host, not a fresh machine or by an independent teammate.

## Repository map

| Content | Location |
|---|---|
| Course requirements and evidence | [ASSIGNMENT.md](ASSIGNMENT.md) |
| Negotiation environment and public observations | [study/env.py](study/env.py), [pilot/env.py](pilot/env.py) |
| Source documents and scenario mapping | [contexts.json](study_results/contexts.json), [data/source/](data/source/) |
| Required Random / Heuristic / Bandit / PPO comparison | [summary.csv](study_results/summary.csv), [Luna report](study_results/report.html) |
| Information-use improvement | [method](improvement/README.md), [findings](improvement_results/findings.json) |
| Matched direct Luna/Sol comparison | [direct comparison](sol_results/direct_comparison.csv), [matched controls](sol_results/matched_policy_controls.csv) |
| Small model with strategic control | [findings](small_model_results/findings.json), [report](small_model_results/report.html) |
| Discourse evidence | [walkthroughs](study_results/discourse_walkthroughs.txt), [matched policy paths](small_model_results/outline_matched_discourse.json) |
| Raw trajectories and checkpoints | Each `*_results/raw/` and `*_results/models/` directory |
| Model requests and responses | Each `*_results/api/` directory; credentials excluded |
| Manuscript and editable figures | [paper/](paper/), [paper guide](paper/README.txt) |
| Overleaf project and submission records | [submission/](submission/) |
| Research claims and remaining scope | [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md) |

Open HTML reports in a local browser. Experiment-stage protocols and reports retain historical costs and observations; this README and [release notes](provenance/RELEASE_NOTES.md) describe the current release.

## Results — keep evaluation sets separate

The main required comparison uses four test documents, 64 numerical settings per document, and three utterance styles. Mean net utility is Random **−0.1457**, Bandit **0.1183**, Heuristic **0.4190**, PPO **0.3581** (three-seed SD 0.0181), and public-belief planner **0.4882**. Base PPO does not outperform the heuristic.

The following comparison uses a separate matched subset of **64 document–setting combinations, style 0, and the same Luna execution bank**. It must not be mixed with the full evaluation above or the new numerical conditions in the improvement experiment.

| Strategic controller | Net utility |
|---|---:|
| Direct Luna, reasoning none | −0.367745 |
| Direct Sol, reasoning none | −0.367279 |
| Direct Sol, reasoning medium | 0.514012 |
| Heuristic with Luna execution bank | 0.469719 |
| Planner distillation followed by PPO with Luna bank, three seeds | 0.475318 |
| Exact public-belief planner with Luna bank | 0.533428 |

The planner uses known synthetic transition rules and fixed utterance interpretations; it never receives the actual hidden type. The learned controller remains below Sol medium and only slightly above the strong heuristic. Local controllers eliminate model calls for strategy selection, while bank construction/interpretation and teacher/training costs remain. The none and medium conditions also differ in reasoning and output-token budgets.

## Completion status

- Saved results cover the main Luna experiment, required ablations, rule sensitivity, information-use improvements, and Sol replication.
- The source collection contains 12 documents from six event groups. Test documents cover only **two** groups, Syria and Iran. Enumerated paths are not independent historical negotiations.
- Luna generation–interpretation consistency of 379/396 and synthetic probe consistency of 180/180 are not accuracy against human labels. Mismatches remain in the transitions.
- `routing/` and `routing_results/` contain an **interrupted side experiment**. Development completed, but only 17 of 32 test cells completed. It is not a completed evaluation.
- The manuscript includes six discourse episodes and required experimental diagnostics. Current English TeX and the PDF are authoritative. Earlier bilingual drafts are retained locally outside this release.
- The owner will fill in [authors.tex](paper/authors.tex), actual roles and teammate execution in [TEAM_RECORD.txt](submission/TEAM_RECORD.txt). A separate 62-item human-labeling exercise and conference acceptance are not course completion requirements.

## Retraining and paid regeneration

Use a separate copy to preserve the saved release. [REPRODUCING.md](REPRODUCING.md) distinguishes offline replay, retraining, and historical paid-generation code. This release excludes `.env`, private cost ledgers, professor-distributed guide PDFs, temporary files, and installed dependencies. Packaging and English-language cleanup made no new model calls.
