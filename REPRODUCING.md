# Reproduction and preserved experimental history

## Supported offline entry point

Install `requirements.txt`, then run `python -X utf8 -m tools.verify_release` and `python -X utf8 -m tools.reproduce_release all` from the repository root. The wrapper requires no key, forbids network calls, runs independent child processes and redirects JSON verification outputs to `verification/`. Re-running it does not change the archived models, banks or raw trajectories.

Available stages are `quick` (all unit tests + main Luna saved-model replay), `main` (main replay + all main raw-log audit), `improved` (all selected Luna seed-17 paths), `sol` (Sol base + improved paths), and `all` (all of these). Results include paths and reward checks. `verification/summary.json` records process return codes; individual outputs/logs record the actual checks. This is a deterministic saved-policy/log reproduction, not retraining or a teammate sign-off.

The `provenance/source_export.json` manifest records original hashes before release edits. `RELEASE_MANIFEST.json` records the shipped hashes. `tools.verify_release` checks the latter, while `provenance/RELEASE_CHANGES.json` lists release edits. Verify before reproduction. Checkpoints use Python/PyTorch serialization and should only be loaded from a trusted source with matching hashes.

## Full local retraining

The historical per-module READMEs and protocols document training and evaluation. Use a separate copy and fresh result paths to preserve the published records. Core dependencies are included in the pinned requirements. `study/train.py` and `study/evaluate.py` implement the original Bandit/PPO and complete evaluation; `improvement/run.py` implements teacher collection, distillation and PPO refinement. Main learning requires the frozen contexts, numeric settings, protocol and language bank. Sol configuration redirects the shared implementation inside its process. Training was not repeated during repository packaging.

## Historical scripts that are not release entry points

`study.finalize` and `solstudy.report` are archived report builders tied to the original private `budget/pilot_20261012.json`. That account ledger is deliberately not shipped. In addition, the later interrupted routing run means `solstudy.report`'s original all-calls-settled assertion no longer applies to the final ledger. The frozen reports/tables are included; do not use those builders as the quick-start reproduction commands or infer that old cost snapshots cover later calls.

`tools/verify_package.py` was a pre-experiment handoff checker that forbade checkpoints/Git metadata. It is superseded in this release by `tools.verify_release`; the test of private imported hashes is replaced with the release-integrity check. All scientific environment/request/data tests are retained.

## Paid generation is separate

The historical `pilot.live`, `study.language`, direct-model selectors and `solstudy.api` contain the actual API request construction and reservation logic. They are preserved for inspection, not invoked by the release wrapper. Re-running paid generation requires an independently approved budget, `OR_KEY` set privately, fresh endpoint verification, and an explicit new output/ledger setup. Do not copy the experiment owner's key/account ledger or mistake a saved metadata snapshot for current API availability. The historical configs' `execution_enabled: false` is metadata; the transport modules, not that flag, govern live calls.

`routing/` is an interrupted exploratory extension. Its test is incomplete and has a stop guard; none of the quick-start commands resume it. The last response-cost total was $4.7896438 with an unresolved $0.048328 reservation ($4.8379718 conservatively). This is the project history, not authorization to spend that amount in a new run.
