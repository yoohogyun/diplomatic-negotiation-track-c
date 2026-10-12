# Track C submission requirements

The course assignment takes priority. This map summarizes the supplied guide against repository evidence without redistributing the guide itself.

| Requirement | Evidence / status |
|---|---|
| Diplomatic negotiation POMDP; hidden types separate from public observations; bilateral acceptance and costs | `pilot/env.py`, `study/env.py`, environment tests |
| Six discourse functions with meaning-dependent transitions | 11 discrete actions, `study_results/language_bank.json`, semantic probes |
| Real diplomatic sources and at least ten environment validation cases | 12 document contexts and 24 type-specific paths in `study_results/source_case_validation.json` |
| Random, strong Heuristic, Bandit, and PPO | `study_results/summary.csv`, saved `models/` and `raw/` |
| Three seeds for base Bandit/PPO | Seeds 17, 29, 43; 102,400 interactions per run |
| Discount, entropy, and reward ablations; information/rule sensitivity | `study_results/protocol.json`, `sensitivity_metrics.json`, figures |
| Both utilities, costs, agreement, breakdown, length, transitions, and learning curves | Tables, raw trajectories, training logs, and figures in `study_results/` |
| Discourse analysis of at least six episodes | Main Figure 3 and analysis, six quoted synthetic-dialogue cases in Appendix B, transition analysis in Appendix C |
| GitHub code, data, and execution instructions | Public repository and offline verification commands |
| Four pages before References, including Limitations, and editable sources | `paper/main.pdf`, TeX sources, and Overleaf ZIP; author information to be filled in by the owner |
| Actual teammate reproduction and contributions | Record actual execution in `submission/TEAM_RECORD.txt`; distinct from automated host verification |

PPO does not need to be the best policy. A fixed template/cached language module and Luna with reasoning none are allowed. The planner is an additional comparison, not a replacement for the four required policies. Conference submission or acceptance is not required for course completion. Reviewing 20 discourse turns is recommended; the separately prepared 62-item labeling sheet is optional research material.

## Teammate reproduction record

Distinguish automated replay in fresh Python processes from an actual teammate running the release. Leave fields blank until the work is performed.

```text
Executor:
Date / OS / Python / package versions:
Git commit:
Commands:
Verification output files:
Agreement with saved results / causes of failure:
Actual contribution:
```
