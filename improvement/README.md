# Learning to use information: authorized follow-up

This follow-up implements the user's request to find a policy that benefits from information, rather than stop at describing the original failures. The original `study_results/` experiment is retained. This is method development after seeing the original study, not a retroactive claim of preregistration.

## Method

An exact public-belief planner provides action-value targets on the six training contexts and 96 original training configurations. It uses public transition rules and the interpreted Luna utterance bank, never the actual opponent type. Its conditional response branches average over the public posterior.

Collect expert paths and paths with 35% random exploration; enumerate nonzero latent branches to cover possible observations. Hidden type controls the simulated response but is not a teacher/student input. The 43,440 examples contain the same 31 public features used by the original PPO, eleven action-value targets and a value target. Source context identity is not a student feature.

Three fresh 64-by-64 policy networks use soft action-value distillation (temperature 0.03), plus value regression. Each gets 2,400 updates of 256 samples with Adam at 0.001. These form the `distilled` candidate. The `distilled_ppo` candidate further trains each checkpoint for 102,400 interactions using the original base PPO hyperparameters and original rewards. All six checkpoints and curves are retained.

The teacher has additional simulator knowledge and computation. This is an initialization/training intervention, not an equal-resource comparison between model-free learners and not a newly invented distillation algorithm. At deployment both candidates use only their network; neither calls the planner. Luna weights and its frozen language bank are unchanged. There are no additional API requests.

## Selection and evaluation

Choose the method by mean net utility across three seeds on the original development set. Only final checkpoints are eligible. Then evaluate both candidates, original PPO, the original dev-selected heuristic, and full/no-query public planners on 64 new numerical settings excluded from every original train/dev/test configuration. These settings use the same four test source contexts and three styles. The source material and language bank were already exposed; the new split adds a numerical holdout, not new historical evidence.

The selected method also receives three deployment interventions: block query and choose the highest remaining learned logit; block signal similarly; replace posterior belief by the prior while retaining other public state. They are not retrained ablations. Paired query-removal losses quantify whether using questions actually improves net utility. Conditions where the full planner outperforms the no-query planner define the positive-information-value subset. All three seed results and failed cases remain available.

## Outputs and reproduction

Open `improvement_results/report.html`. `selection.json` preserves development selection. `seed_metrics.csv`, `paired_comparisons.csv`, and `findings.json` give numerical results. `raw/` contains weighted trajectories, utterances, meanings, states and costs. `successful_query_examples.json` records successful query cases. `reproduction.json` verifies every selected seed-17 path after a fresh model load.

Offline commands, without retraining or API requests:

```powershell
python -X utf8 -m improvement.report
python -X utf8 -m improvement.reproduce
```

For a full rerun, use a separate fresh copy without `improvement_results/` so frozen files are not overwritten. Run `python -X utf8 -m improvement.run` with each stage in order: `prepare`, `collect`, `train`, `evaluate`; then the two commands above. Original `study_results/` banks, settings and checkpoints are required as read-only inputs. Dependencies match `study_results/runtime.json`.
