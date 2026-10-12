"""Verify shipped artifacts against the release manifest; no network or keys."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    checked=0
    for r in manifest['files']:
        p=(ROOT/r['path']).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Unsafe manifest path')
        if not p.is_file():raise AssertionError('Missing artifact: '+r['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            raise AssertionError('Changed artifact: '+r['path'])
        checked+=1
    return {'release_files_verified':checked,'all_hashes_match':True,'network_used':False}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
