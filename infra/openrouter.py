"""Offline OpenRouter request/response contract for the approved LUNA model.
Inputs: config, role/content messages, model response or public endpoint metadata.
Outputs: validated request/digest/parsed response. No credential reads or network.
Side effects: save_new writes only a caller-selected exclusive JSON file.
Purpose: preserve tested routing, schema, price and replay-integrity primitives.
Live transport and budget reservation are deliberately NOT included: implement
only after numeric carryover reconciliation and explicit user approval.
Dependencies: Python standard library. Tests: unittest discover -s tests -v.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
META_URL = "https://openrouter.ai/api/v1/models/openai/gpt-5.6-luna/endpoints"
APPROVED_MODELS = {'openai/gpt-5.6-luna', 'openai/gpt-5.6-sol'}


def endpoint_tag(tier):
    if tier != 'default':
        raise ValueError('Service tier not explicitly authorized.')
    return 'openai'


def metadata_url(model):
    if model not in APPROVED_MODELS:
        raise ValueError('Model not explicitly authorized.')
    return 'https://openrouter.ai/api/v1/models/' + model + '/endpoints'


def canonical(value):
    """Stable UTF-8 serialization for content hashes; no side effects."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_new(path, value):
    """Write once; refuse overwrites, including interrupted experiments."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)




def build_request(config, messages, sample_index):
    """Create a fully specified request; sample index fixes the generation seed."""
    if config["endpoint_url"] != CHAT_URL or config["metadata_url"] != metadata_url(config['request']['model']):
        raise ValueError("Unexpected API URL; credentials must only go to OpenRouter.")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative.")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list.")
    for message in messages:
        if set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("Only role/content text messages are allowed.")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError("Empty or non-text message.")
    if len(canonical(messages)) > 24000:
        raise ValueError("Prompt exceeds this single-request pilot's 24KB limit.")
    request = json.loads(json.dumps(config["request"]))
    request["messages"] = messages
    request["seed"] = config["generation_seed_base"] + sample_index
    if config["automatic_retries"] != 0:
        raise ValueError("This pilot runner does not retry potentially billed requests.")
    provider = request["provider"]
    tag = endpoint_tag(request['service_tier'])
    if (request["model"] not in APPROVED_MODELS
            or provider["only"] != [tag] or provider["order"] != [tag]
            or provider["allow_fallbacks"] is not False or provider["require_parameters"] is not True
            or request["stream"] is not False or not 1 <= request["max_tokens"] <= (4096 if request['model']=='openai/gpt-5.6-sol' else 512)):
        raise ValueError("Strict model/tier/routing or pilot token limits were changed.")
    if not 0 <= request["seed"] < 2**31:
        raise ValueError("Seed exceeds the experiment's signed 32-bit range.")
    return request


def check_metadata(metadata, request):
    """Fail closed against the selected endpoint's advertised capabilities/prices."""
    entries = [e for e in metadata["data"]["endpoints"] if e.get("tag") == endpoint_tag(request['service_tier'])]
    if metadata["data"]["id"] != request["model"] or len(entries) != 1:
        raise ValueError("Expected model/pinned-tier endpoint missing.")
    entry = entries[0]
    controls = {"model", "messages", "provider", "stream", "service_tier", "prompt_cache_options"}
    required = set(request) - controls
    if request["response_format"]["type"] == "json_schema":
        required.add("structured_outputs")
    unsupported = required - set(entry["supported_parameters"])
    if unsupported:
        raise ValueError("Unsupported request parameters: " + ", ".join(sorted(unsupported)))
    for name in ("prompt", "completion"):
        if float(entry["pricing"][name]) * 1e6 > request["provider"]["max_price"][name] + 1e-12:
            raise ValueError("Endpoint price exceeds the configured per-million-token ceiling.")
    return entry


def validate_schema(value, schema):
    """Validate our small flat JSON-schema subset; fail on unsupported schemas.

    Supports strict objects with required fields and primitive/nullable fields,
    enums and numeric bounds. No external dependency; not a general JSON-schema
    implementation. Used for utterance generation and text-only interpretation.
    """
    allowed = {'type', 'properties', 'required', 'additionalProperties', 'enum', 'minimum', 'maximum'}
    if set(schema) - allowed:
        raise ValueError('Unsupported schema keyword.')
    kind = schema['type']
    if isinstance(kind, list):
        if set(kind) != {'number', 'null'}:
            raise ValueError('Unsupported union schema.')
        if value is None:
            return
        kind = 'number'
    if kind == 'object':
        props = schema['properties']
        if (not isinstance(value, dict) or schema.get('additionalProperties') is not False
                or set(schema['required']) != set(props) or set(value) != set(props)):
            raise ValueError('Object fields violate schema.')
        for name, field_schema in props.items():
            validate_schema(value[name], field_schema)
        return
    if kind == 'string':
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Expected nonempty string.')
    elif kind == 'number':
        import math
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < schema.get('minimum', -float('inf'))
                or value > schema.get('maximum', float('inf'))):
            raise ValueError('Invalid numeric value.')
    else:
        raise ValueError('Unsupported schema type.')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('Value not in enumeration.')


def validate_response(response, schema=None, expected_model='openai/gpt-5.6-luna', expected_tier='default'):
    """Reject wrong routing, incomplete output and schema violations, retaining raw evidence."""
    endpoint_tag(expected_tier)
    if response.get("error") or response.get("service_tier") != expected_tier:
        raise ValueError("Response failed or did not confirm requested tier; do not use this sample.")
    if response.get("provider") != "OpenAI":
        raise ValueError("Response did not confirm the intended OpenAI provider.")
    if expected_model not in APPROVED_MODELS or response.get("model") not in {expected_model, expected_model + '-20260709'}:
        raise ValueError("Response model mismatch; inspect before using this sample.")
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Missing, refused or truncated completion.")
    parsed = json.loads(choices[0]["message"]["content"])
    if schema is None:
        schema = {'type': 'object', 'properties': {'utterance': {'type': 'string'}},
                  'required': ['utterance'], 'additionalProperties': False}
    validate_schema(parsed, schema)
    return parsed


