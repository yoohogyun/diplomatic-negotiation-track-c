# Track C experiment requirements

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
| Discourse analysis of at least six episodes | `study_results/discourse_walkthroughs.txt`, `small_model_results/outline_matched_discourse.json`, saved action-transition figures and logs |
| GitHub code, data, and execution instructions | Public repository and offline verification commands |

PPO does not need to be the best policy. A fixed template/cached language module and Luna with reasoning none are allowed. The planner is an additional comparison, not a replacement for the four required policies. Conference submission or acceptance is not required for course completion. Reviewing 20 discourse turns is recommended; the separately prepared 62-item labeling sheet is optional research material.
