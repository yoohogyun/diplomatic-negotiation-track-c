"""Read original diplomatic text without inheriting splits or annotations.

Inputs: two original source CSVs. Outputs: provenance-bearing unlabelled rows and
inventory counts. Side effects: read-only files, no network. Dependencies: stdlib.
Errors: missing fields, empty/duplicate IDs, absent source files. Does NOT infer
independent crises or create a final split; source/date/near-duplicate grouping
must be reviewed separately before transformation or generation.
"""
import csv
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('id','source','doc_title','date','url_or_doi','speaker_role','source_type','sentence')
PRIMARY=('all_250.csv','condition_supplement.csv')


def select_rows(rows):
    """Whitelist provenance/text only, explicitly dropping old split/label fields."""
    result=[];seen=set()
    for raw in rows:
        if any(not isinstance(raw.get(k),str) or not raw[k].strip() for k in FIELDS):
            raise ValueError('Missing source/provenance field')
        if raw['id'] in seen:raise ValueError('Duplicate source ID')
        seen.add(raw['id']);result.append({k:raw[k] for k in FIELDS})
    if not result:raise ValueError('Empty corpus')
    return result


def load_primary(directory=None):
    directory=Path(directory) if directory else ROOT/'data/source/sentences'
    rows=[]
    for name in PRIMARY:
        with (directory/name).open(encoding='utf-8-sig',newline='') as f:
            rows.extend(csv.DictReader(f))
    return select_rows(rows)


def inventory():
    rows=load_primary()
    with (ROOT/'data/source/sentences/candidates.csv').open(encoding='utf-8-sig',newline='') as f:
        candidates=sum(1 for _ in csv.DictReader(f))
    return dict(primary_rows=len(rows),candidate_rows=candidates,
                unique_url_title_pairs=len({(r['url_or_doi'],r['doc_title']) for r in rows}),
                context_text_files=len(list((ROOT/'data/source/raw').rglob('*.txt'))),
                exact_duplicate_text_rows=len(rows)-len({r['sentence'] for r in rows}),
                independently_human_labelled=False,final_split_created=False,
                historical_independence_verified=False,
                original_csv_sha256={name:hashlib.sha256((ROOT/'data/source/sentences'/name).read_bytes()).hexdigest() for name in PRIMARY})
