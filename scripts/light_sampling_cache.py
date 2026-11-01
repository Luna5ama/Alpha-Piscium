"""Content-addressed offline derivatives; raw evidence is always verified by callers."""
import hashlib
import json
import os
from pathlib import Path


def algorithm_identity(*names):
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def cached_json(directory, identity, compute):
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = Path(directory) / (key + ".json")
    if path.exists():
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            encoded = json.dumps(entry["value"], sort_keys=True, separators=(",", ":")).encode()
            if entry["identity"] == identity and hashlib.sha256(encoded).hexdigest() == entry["value_sha256"]:
                return entry["value"], True
        except (ValueError, KeyError, TypeError):
            pass
    value = compute()
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    entry = dict(identity=identity, value=value, value_sha256=hashlib.sha256(encoded).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(entry, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
    return value, False
