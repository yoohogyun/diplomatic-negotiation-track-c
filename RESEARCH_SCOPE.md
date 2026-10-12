# Short-paper direction, after the course requirements

Working question: **When can explicit belief-state control improve a small language execution module in a source-grounded diplomatic negotiation simulator?**

The potential contribution is a controlled empirical and discourse analysis. PPO, policy distillation, strategy/language separation and exact planning are existing methods; this work does not invent those algorithms. A strong inference-enabled model is an effective comparison, not an assumed failure case.

Evidence currently supports a limited result: an exact controller with declared model knowledge achieves good simulated utility using a frozen Luna realization/interpretation bank, with no online strategy-selection API calls. A distilled PPO improves the original learner but does not exceed Sol medium on the matched comparison. At evaluation style 0, all 44 Luna action meanings also match the canonical templates, so current utility scores alone do not demonstrate that Luna language generation is necessary.

For the four-page story, retain the mandatory policy table and the matched direct-model comparison as distinct panels. Reserve meaningful main-text space for discourse: repeated rejected offers, query-induced belief updates, costly signals changing acceptance, and the gap between the phrase “readiness to ratify” and actual ratification. In the matched example, the signal changes the opponent's utility; the query changes knowledge, not the opponent's preference. The corresponding saved paths are in `small_model_results/outline_matched_discourse.json`.

Before making a strong COLING contribution claim, the two highest-value checks are held-out wording/semantic fidelity versus templates, and controller-model mismatch with controller assumptions held fixed. Neither check is reported as completed here. Additional model families are not part of the current scope. The full paper will be written with the user later.

Limits: only two evaluation event groups; sources and banks had prior researcher exposure; simulated utilities and behavior; frozen bank instead of online generation; known-model planning advantage; single cached direct-model response per public state; single-seed auxiliary ablations; no end-to-end cost or latency superiority claim; no verified human semantic gold labels. A short-paper submission remains a goal, not a prediction of acceptance.

