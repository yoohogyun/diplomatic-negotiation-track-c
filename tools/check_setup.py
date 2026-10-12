"""Offline data and pinned-request check. Never reads credentials or sends HTTP.
Input local config/snapshot/CSV; output console JSON; no file writes.
The snapshot check is historical compatibility, NOT current API availability.
Run: python -B -m tools.check_setup
"""
import json
from infra import openrouter as api
from infra.source_inventory import inventory


def main():
    config=api.load_json(api.ROOT/'configs/luna_standard.json')
    messages=api.load_json(api.ROOT/'prompts/smoke_messages.json')
    request=api.build_request(config,messages,0)
    metadata=api.load_json(api.ROOT/config['metadata_snapshot'])
    api.check_metadata(metadata,request)
    print(json.dumps(dict(mode='offline only; no credentials/network',
        request_sha256=api.digest(request),model=request['model'],tier=request['service_tier'],
        live_execution_implemented=False,snapshot_is_not_live_verification=True,
        corpus=inventory()),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
