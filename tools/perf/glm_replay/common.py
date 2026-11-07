"""Strict JSON, canonical hashing and schema primitives for offline replay tools."""
import hashlib
import json
import math
import re
from pathlib import Path


class ReplayError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def fields(value, names, label):
    require(isinstance(value, dict), f"{label}: expected object")
    require(set(value) == set(names.split()), f"{label}: unexpected or missing fields")


def integer(value, label, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum, f"{label}: invalid integer")
    require(maximum is None or value <= maximum, f"{label}: integer out of range")
    return value


def number(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: invalid number")
    require(value > 0 if positive else value >= 0, f"{label}: invalid duration")
    return value


def string(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: expected nonempty string")
    return value


def hex_id(value, label, length=64):
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
            f"{label}: invalid digest")
    return value


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _constant(value):
    raise ReplayError(f"nonfinite JSON constant: {value}")


def parse_json(text):
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReplayError(f"invalid JSON: {exc}") from exc


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_bytes(canonical_bytes(value) + b"\n")
