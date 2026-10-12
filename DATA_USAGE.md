# Data provenance and use

This course/research repository includes supplied source material for reproducibility and is currently public on GitHub. Public visibility does not establish a blanket redistribution license for third-party diplomatic text or the supplied corpus. Source-specific terms and provenance remain applicable.

- `data/source/sentences/all_250.csv` and `condition_supplement.csv`: 281 primary source rows in total.
- `data/source/sentences/candidates.csv`: 7,429 candidate rows.
- `data/source/raw/`: 73 source text files. These are provenance/context, not 73 independent experimental negotiations.
- `study_results/contexts.json`: the 12 selected documents with source sentence, date, title, URL/DOI, and source-to-hypothetical mapping. Six event groups; two held-out event groups at evaluation.
- `infra/source_inventory.py`: explicitly retains source/provenance fields and discards pre-existing annotation/split fields when loading primary rows. Old annotations in supplied files are not treated as gold labels.

Each source's URL/DOI and document metadata are retained in the CSV/context records. Follow those references for the original documents and their own terms; a URL or DOI alone does not grant redistribution rights. The supplied materials include the CMFA PressCon reference DOI `10.7910/DVN/BAKGET` and diplomatic documents from their stated issuing institutions.

The generated utterances, opponent replies, utilities, hidden types, conditional timings and response probabilities are synthetic. They are not historical transcripts or observations of actual diplomatic agreements. Model interpretation agreement is not independently verified human annotation.

No professor-distributed guide PDFs, API keys, `.env` files or personal account billing records are included. API request/response artifacts preserve experiment prompts, outputs, usage and model metadata without authorization headers. No open-source license has been granted by this export for either the authors' code or third-party materials; repository access does not establish redistribution permission.

