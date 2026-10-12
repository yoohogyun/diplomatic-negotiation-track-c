# Luna-only experiment package

This is the main study after the separate exploratory pilot. The user requested finishing experiments first and writing the paper together later; no paper is included in this stage. No additional language model was used for generation, interpretation or the prompted strategy baseline.

## Fixed scope

- 12 document contexts across six event groups, drawn from the provided diplomatic corpus. Group split: Korean Peninsula, South Sudan and Kashmir train (six documents); Gaza development (two); Syria and Iran evaluation (four).
- The sources motivate diplomatic objects and discourse. Reciprocal conditions, durations, opponent types, utilities and response probabilities are researcher-created. A call for dialogue is not evidence of an actual reciprocal bargain. The source-to-hypothetical mappings explicitly disclose this.
- 96 numerical settings per training context, 32 per development context, 64 per evaluation context: 576/64/256 parameterized context-settings. They are not that many historical events. Test has only TWO event groups.
- Existing source material had prior pilot/researcher exposure. The split is held out from policy training, not a claim of researcher blindness or an unseen historical corpus.
- 396 fresh Luna utterances: 12 contexts x 11 actions x three styles; independent Luna interpretation requests, plus 180 authored counterfactual probes (12 x 3 x 5).
- Three 102,400-step Bandit runs and eleven 102,400-step PPO runs. The latter include three base seeds, two discount settings, two entropy settings, two major-reward omissions, belief-masked retraining and query-disabled retraining. Additional settings use seed 17 and are exploratory.
- Separate direct-Luna strategy experiment: the first 16 numerical test settings across four source contexts (64 cases), style 0, one cached model response per distinct public observation. Compare it only to matched policy rows in `direct_luna_matched_summary.csv`.

The primary protocol, numeric conditions and contexts were saved before generation/training. The direct-Luna extension has its own protocol, fixed before those API outcomes were collected. Every condition and seed is retained; heuristic selection uses only development mean utility. The final fixed-budget PPO checkpoint is used for every run, with no test-based checkpoint selection.

## What the system learns

Luna is a fixed language execution module. A small PPO network learns action selection; Luna's model weights are not trained. A frozen bank of independently interpreted utterances allows on-policy environment interaction without repeated online API calls. Each episode selects a style; each step looks up the selected action's actual utterance and interpreted meaning. Misinterpretations or altered meanings affect state/reward; they are not silently replaced by intended actions.

The new environment in `env.py` uses the numerical transition rules defined in `pilot/env.py`. All six functions are represented: ambiguity, costly signal, conditional offer, private inquiry, ratification and withdrawal. The 31-dimensional public observation includes the public proposal, posterior belief, acceptance and commitment status, time, escalation, public rule parameters, style and last interpreted function. The actual type is never a policy input.

Opponent responses follow fixed rules and are displayed using explicitly synthetic response templates. Query likelihoods and offer acceptance/rejection both update the posterior. Acceptance expresses willingness; an accepted proposal can still be revised before ratification. This assumption allows diagnostic proposals and is not a claim about historical legal commitment.

The main planner knows the public rules AND the fixed utterance/interpretation mapping for the current context/style, but never the actual type. It is an exact model-aware benchmark, not a learning-efficiency peer. The no-query planner removes every bank action interpreted as a query, while keeping information from accept/reject responses.

The direct-Luna baseline receives explicit public rules in its prompt, unlike PPO receiving only interaction feedback plus public parameters. This knowledge/computation difference is disclosed. Its single-response cache is not evidence of stability across model seeds.

## Evaluation

For each deterministic policy, enumerate both types, both possible informative-query responses, and all three styles with their exact probabilities. All metrics use the same undiscounted evaluation reward, including policies trained with omitted costs. Aggregate repetitions within context/settings before averaging equally across the two test event groups. Three-seed SD measures training variability, not uncertainty across world events.

Random uses three evaluation RNG seeds and four action-sequence repetitions per nonzero latent cell. Its variability is Monte Carlo variation, not training variation. Raw gzip JSONL logs include all policies, weighted latent branches, actual utterances, interpretations, responses, states, reward components and termination reasons. Hidden metadata is attached only for evaluation.

Paired sensitivity sweeps change one public rule at a time on the same test contexts: question reliability, signal effect, acceptance feasibility, rejection costs and query costs. These are evaluation-only changes, distinct from retrained belief/query ablations. Belief masking leaves acceptance and proposal history available and is not an all-information removal intervention.

## Run or reproduce

Use the dependency versions saved in `study_results/runtime.json`. This host uses local `.pilot_deps` for Gymnasium and Stable-Baselines3. Importing the `pilot` package places it on the path and loads NumPy before PyTorch to avoid this host's OpenMP initialization issue.

```powershell
python -X utf8 -m unittest discover -s tests -p test_study.py -v
python -X utf8 -m study.reproduce
python -X utf8 -m study.audit
python -X utf8 -m study.finalize
```

The last three commands need the supplied saved results and models, make no API calls, and reproduce saved-model paths, audit all saved evaluation records, and rebuild result tables and plots. `reproduction_record.json` distinguishes this fresh-process verification from an actual teammate's independent reproduction.

For a full rerun, use a fresh output directory/copy so protocols are not overwritten. Run `study.prepare`, `study.language`, `study.train`, `study.luna_policy`, `study.evaluate`, `study.sensitivity`, `study.review`, `study.reproduce`, `study.audit`, then `study.finalize`. The language and direct-policy commands send paid API requests using OR_KEY loaded in memory; never include `.env` in an archive. Cached validated calls can be reused. If a request failed or has an uncertain outcome, inspect its saved record rather than automatically repeating it.

Four parallel API workers use durable reservations and a shared lock. Two independent local training processes avoid shared random-generator state. Raw request bodies never contain the authorization header. Costs remain on the same ledger as the initial pilot and are checked against account usage. The internal ceiling is $9 of additional spending, below the user's $10 limit.

## Review and remaining human tasks

`human_review_blank.csv` contains 62 blinded items: all 12 real source anchors, all 17 roundtrip mismatches, and 33 deterministic controls. It is deliberately error-enriched, not a prevalence sample. `human_review_key.json` should be hidden from reviewers until their labels are complete. Model-to-model agreement is NOT human annotation accuracy; generated errors and interpreter errors need separate assessment.

The programmatic case validation covers all 12 contexts and both types. Discourse walkthroughs, transition counts/probabilities/heatmaps, learning curves, all comparison metrics and paired rule sensitivities are included. A real teammate's reproduction and human discourse review remain unperformed. Paper writing, model expansion, external repository publishing and submission are deferred.
