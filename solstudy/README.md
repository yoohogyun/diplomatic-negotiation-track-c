# Sol replication

The user explicitly requested repeating the Luna experiments with GPT-5.6 Sol to check whether the observations persist for a stronger model. The original Luna files remain intact. The chosen PPO negotiation baseline is our implementation of an existing algorithm; its weakness alone is not a general finding about LLMs.

## Primary comparison

Direct Sol versus direct Luna uses the same first 16 numerical settings across four test contexts, the identical public-rule prompt, style 0, and the unchanged Luna execution bank. Both use reasoning `none`, model-default temperature and a 512-token output limit. This isolates selector model under the original inference settings. A separately frozen Sol `medium` condition uses the same cases and bank with a 4,096-token total output/reasoning limit. It changes inference computation as well as enabling reasoning, so keep it distinct from the model-only contrast. Each method caches one response per public state; there is no multi-seed LLM sampling claim.

A separate Sol-bank direct run changes the execution language module as well. It reuses a cached Sol response only when the full prompt is identical. Hidden type is used only to sample/enumerate environment responses; it is not included in the model prompt.

## Full execution-module replication

The 396-generation, 396-interpretation and 180-probe language protocol is repeated with Sol. The same 14 learned-policy runs, numerical conditions, source split, dev selection, ablations and rule sensitivities are repeated on the Sol bank. The improvement experiment repeats the six distillation/fine-tuning checkpoints and selects between the same two methods using development utility. It evaluates on the same follow-up numerical settings used for Luna, enabling paired model comparisons. These are previously exposed sources/settings, not new researcher-blind evidence.

`solstudy.run.configure()` redirects shared environment/training/evaluation code only inside the Sol process. Each Windows worker repeats that configuration. The numerical rules, public observations, reward, prompts and training budgets are unchanged. Model realization/parsing differences are retained in transitions.

## Cost, provenance and outputs

All paid requests share `budget/pilot_20261012.json` and a process/thread lock with the earlier Luna calls. The internal cumulative ceiling stays $9, below the user's exclusive $10 cap. The Sol endpoint was checked live and pinned to OpenAI standard routing with no fallback. No retries occur after ambiguous or billed outcomes. API records omit credentials. Two independent API batches can overlap, four workers each; CPU training uses two workers.

The imported Luna-only API contract was intentionally extended for the explicitly authorized Sol model and its separate reasoning-token limit. `IMPORT_MANIFEST.json` is unchanged; `INTENTIONAL_CHANGES.json` records original/current hashes and the reason. `verify(allow_intentional=True)` checks those exact documented changes; default `verify()` still detects deviation from the original handoff.

Results: `sol_results/report.html`, `direct_comparison.csv`, `matched_policy_controls.csv`, `main_comparison.csv`, `improvement_comparison.csv`, `language_comparison.csv`, `audit.json`. Full models, curves, per-case logs and API responses are retained in `sol_results/` and `sol_improvement_results/`.

To regenerate the final comparison offline:

```powershell
python -X utf8 -m solstudy.report
python -X utf8 -m solstudy.run reproduce
python -X utf8 -m solstudy.run improve_reproduce
```

A full fresh replication uses a separate copy, preserving original Luna inputs, and invokes `solstudy.run` stages: `prepare`, `language`, `none_luna`, `medium_luna`, `none_sol`, `train`, `evaluate`, `sensitivity`, `review`, `improve`, `improve_evaluate`, `improve_report`, `reproduce`, `improve_reproduce`. `prepare` requires a fresh live endpoint metadata snapshot at `sol_results/api/metadata_live.json`. The language/direct stages are paid; the others are local. Finish with `solstudy.report` and the ledger account reconciliation. Runtime dependencies match the original study.

Paper writing remains deferred. Neither generated agreement nor simulator payoff is evidence of real diplomatic outcomes.
