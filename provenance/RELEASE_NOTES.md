# Release notes — 2026-10-12

This export preserves the original experiment code/data/results in their relative paths, with explicit packaging-only edits. Original artifact hashes are in `source_export.json`; the final package hashes are in `../RELEASE_MANIFEST.json`. `RELEASE_CHANGES.json` distinguishes modified, omitted and added files. Original workspace files were not edited.

Included: supplied diplomatic sources, frozen model requests/responses without keys, full language banks, protocols, numeric conditions, trained policies, learning logs, raw evaluations, tables, figures, discourse examples, and the interrupted routing extension clearly marked as incomplete.

Excluded: `.env`, credentials, the original private account/budget ledger, installed libraries/tools, temporary files, local handoff manifests, professor-distributed guides and unpublished paper planning. Three historical direct-model protocols have their machine-specific bank path replaced by the identical repository-relative path. No numerical outcome or trained checkpoint is changed.

Opaque encrypted reasoning payloads in API responses are replaced by an explicit omission marker. One ciphertext happened to match a credential-token pattern; no actual known key was present. These opaque fields are not used by the experimental code. Visible output, action choices, usage/cost and model metadata remain unchanged, with all original/release hashes recorded.

The private-handoff verifier is replaced by the release manifest verifier, and its one corresponding unit test is adapted. The remaining experiment/request/data tests are retained. A new offline wrapper prevents network use and writes fresh verification records outside the preserved archive. Pinned dependencies and course/research documentation are added.

Historical report costs/completion assertions apply to their original stage, not the later interrupted routing experiment. The current conservative cumulative cost is $4.8379718, consisting of $4.7896438 response cost and $0.048328 unresolved reservation. No new paid model calls were made for this release.

An automatic replay is not a real teammate's reproduction. The repository is a private course/research handoff, not a completed paper, anonymized review supplement or licensed public redistribution of third-party data.
