# Selective Sol reasoning pilot

**Stopped early on user steering and time constraint.** Development finished; test did not. See `routing_results/early_stop.json` and `status.json`. Do not report the partial test cohort as the planned 32-cell result or resume paid requests automatically. The current main direction and offline evidence are in `small_model_results/report.html`.

The user requested an actionable alternative and measured pilot after reasoning-enabled Sol outperformed the earlier learned policies. This experiment asks whether a learned public-state gate can preserve that strong baseline's negotiation utility while reducing its inference use. It does not introduce PPO or generic model routing as a new algorithm.

The cheap actor is the existing Sol-bank planning-distilled PPO policy, seed 17, fixed before this pilot. Three small Q predictors are trained on the existing **training-only** public-planner targets. If the predicted regret of the cheap actor's action exceeds a development-selected threshold, the system asks GPT-5.6 Sol with medium reasoning. Otherwise it executes the cheap action. Neither gate nor API prompt observes the opponent's actual hidden type. Deployment does not call the planner.

`routing_results/protocol.json` specifies all candidates and selection before development calls. Development uses the first eight original development settings and both development documents. The threshold minimizes expected calls among candidates within 0.01 utility of always-Sol. Test uses sixteen new numerical settings and one previously exposed source document per held-out event group, with style 0. Full latent/query branches are probability-weighted. No claim of a new historical holdout is made.

Run from this directory:

```powershell
python -m routing.run prepare
python -m routing.run train
python -m routing.run dev
python -m routing.run test
python -m routing.report
```

Preparation/training/finished evaluations refuse silent replacement. Interrupted evaluations reuse complete per-case results and exact API request caches; unresolved paid requests require inspection and are never retried automatically. The `dev` and `test` stages make paid requests, at four concurrent case workers, using the existing cumulative locked budget ledger. The report stage is fully offline and replays every saved transition and learned decision in a new process.

Each identical public prompt has one cached Sol sample shared between methods. Reported deployment calls, tokens, and API costs **count every logical invocation, including cache hits**. Actual unique billed requests are separately reported. Teacher planning, prior policy training, gate training and language-bank preparation are additional offline resources; this is not an equal-training-budget comparison. Wall-clock latency is not a measured outcome. The numeric-cluster bootstrap is exploratory and excludes model sampling and historical uncertainty.

Original Luna and Sol experiments are preserved. New requests live under `sol_results/api/sol_medium_routing_*`, with new case logs, frozen selection, checkpoints, findings, replay audit and HTML report in `routing_results/`. Paper writing is still deferred.
