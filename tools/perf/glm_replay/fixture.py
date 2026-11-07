"""Immutable input validation; call before loading engines or launching rank collectives."""
from pathlib import PurePosixPath

from .capacity import validate_pool
from .common import digest, fields, hex_id, integer, read_json, require, string, text_digest

SCHEMA = "glm-replay-fixture-v1"
TARGET_REVISION = "078455ffe6472f9a52fbc1139f58b9db2881b25c"
DRAFTER_REVISION = "bf582e4eacc1810f76656d1811693ff6c6737d2a"
NATIVE_BASE = "e52eb42b52b1e50f3f23a5eae8115bbe80c5e7d9"
PYTHON_BASE = "c4646171139ee8a3c38103eaa1699dad226ec12b"


def tokens_payload(fixture):
    return [{key: s[key] for key in ("stream_id", "prompt_tokens", "tape_tokens")}
            for s in fixture["streams"]]


def seal_fixture(fixture):
    fixture["tokens_sha256"] = digest(tokens_payload(fixture))
    fixture["schedule_sha256"] = digest({key: fixture[key] for key in ("preparation", "forwards")})
    fixture["config_sha256"] = digest(fixture["config"])
    fixture["fixture_sha256"] = digest({k: v for k, v in fixture.items() if k != "fixture_sha256"})
    return fixture


def token_array(value, vocab, label):
    require(isinstance(value, list) and len(value) > 0, f"{label}: empty token array")
    for token in value:
        integer(token, label, 0, vocab - 1)


def _provenance(fixture):
    prov = fixture["provenance"]
    fields(prov, "target_revision drafter_revision tokenizer sources", "provenance")
    require(prov["target_revision"] == TARGET_REVISION, "target revision mismatch")
    require(prov["drafter_revision"] == DRAFTER_REVISION, "drafter revision mismatch")
    tok = prov["tokenizer"]
    fields(tok, "kind revision vocab_size files template template_sha256", "tokenizer")
    integer(tok["vocab_size"], "vocab_size", 1)
    string(tok["template"], "template")
    require(tok["template_sha256"] == text_digest(tok["template"]), "template hash mismatch")
    require(isinstance(tok["files"], list), "tokenizer files must be a list")
    paths = set()
    for asset in tok["files"]:
        fields(asset, "path sha256", "tokenizer asset")
        path = PurePosixPath(string(asset["path"], "tokenizer path"))
        require(not path.is_absolute() and ".." not in path.parts and "\\" not in str(path),
                "tokenizer asset path must be relative")
        require(str(path) not in paths, "duplicate tokenizer asset")
        paths.add(str(path))
        hex_id(asset["sha256"], "tokenizer asset")
    if fixture["kind"] == "campaign":
        require(tok["kind"] == "huggingface-local", "campaign requires real tokenizer")
        hex_id(tok["revision"], "tokenizer revision", 40)
        require(tok["revision"] == TARGET_REVISION, "tokenizer revision must match pinned checkpoint")
        require(tok["vocab_size"] == 154880, "campaign vocabulary must match pinned GLM checkpoint")
        require("tokenizer.json" in paths and "tokenizer_config.json" in paths,
                "campaign missing tokenizer assets")
    else:
        require(tok["kind"] == "synthetic-byte" and tok["revision"] == "synthetic-byte-v1"
                and tok["vocab_size"] == 256 and tok["files"] == [], "invalid synthetic identity")
    require(isinstance(prov["sources"], list) and len(prov["sources"]) == 4, "four sources required")
    ids = set()
    for source in prov["sources"]:
        fields(source, "source_id attribution license text sha256", "source")
        for key in ("source_id", "attribution", "license", "text"):
            string(source[key], f"source.{key}")
        require(source["source_id"] not in ids, "duplicate source ID")
        ids.add(source["source_id"])
        require(source["sha256"] == text_digest(source["text"]), "source hash mismatch")
    return tok["vocab_size"], ids


def _config(fixture):
    cfg = fixture["config"]
    fields(cfg, "speed tp slots dense_bits head_bits kv reset selection proposer active_streams placement pool", "config")
    for key, value in {"speed": "standard", "tp": 2, "slots": 4, "dense_bits": 8, "head_bits": 8,
                       "kv": "fp8", "reset": "fresh_prefill", "selection": "greedy", "proposer": False}.items():
        require(type(cfg[key]) is type(value) and cfg[key] == value, f"unsupported config: {key}")
    active = cfg["active_streams"]
    require(isinstance(active, list) and len(active) in (1, 4), "active streams must be solo or four")
    for sid in active:
        integer(sid, "active stream", 0, 3)
    require(len(set(active)) == len(active), "duplicate active stream")
    require(cfg["placement"] in ("balanced", "reversed"), "invalid placement control")


def _streams(fixture, vocab, sources):
    streams = fixture["streams"]
    require(isinstance(streams, list) and len(streams) == 4, "exactly four streams required")
    owners = [0, 0, 1, 1] if fixture["config"]["placement"] == "balanced" else [1, 1, 0, 0]
    for index, stream in enumerate(streams):
        fields(stream, "stream_id owner source_id rendered_prompt rendered_tape rendered_prompt_sha256 "
               "rendered_tape_sha256 prompt_encoding prompt_tokens tape_tokens", "stream")
        integer(stream["stream_id"], "stream_id", 0, 3)
        require(stream["stream_id"] == index, "streams must be in stream_id order")
        integer(stream["owner"], "owner", 0, 1)
        require(stream["owner"] == owners[index], "owner assignment mismatch")
        require(stream["source_id"] in sources, "unknown source ID")
        for which in ("prompt", "tape"):
            text = string(stream[f"rendered_{which}"], f"rendered {which}")
            require(stream[f"rendered_{which}_sha256"] == text_digest(text), f"rendered {which} hash mismatch")
        encoding = stream["prompt_encoding"]
        fields(encoding, "token_ids used_tokens", "prompt_encoding")
        token_array(encoding["token_ids"], vocab, "prompt encoding")
        used = integer(encoding["used_tokens"], "used prompt tokens", 1, len(encoding["token_ids"]))
        token_array(stream["prompt_tokens"], vocab, "prompt")
        require(stream["prompt_tokens"] == encoding["token_ids"][:used], "prompt prefix mismatch")
        token_array(stream["tape_tokens"], vocab, "tape")
        if fixture["kind"] == "synthetic":
            require(encoding["token_ids"] == list(stream["rendered_prompt"].encode("utf-8")),
                    "synthetic prompt encoding mismatch")
            require(stream["tape_tokens"] == list(stream["rendered_tape"].encode("utf-8")),
                    "synthetic tape encoding mismatch")
    for key in ("rendered_prompt", "rendered_tape", "prompt_tokens", "tape_tokens", "source_id"):
        require(len({digest(s[key]) for s in streams}) == 4, f"four distinct {key} values required")


def _schedule(fixture):
    offsets = [0] * 4
    timed_active = set(fixture["config"]["active_streams"])
    for phase in ("preparation", "forwards"):
        active = {0, 1, 2, 3} if phase == "preparation" else timed_active
        forwards = fixture[phase]
        require(isinstance(forwards, list), f"{phase} must be an array")
        require(phase == "preparation" or forwards, "empty timed schedule")
        for fid, forward in enumerate(forwards):
            fields(forward, "round_id forward_id parts", "forward")
            integer(forward["round_id"], "round_id")
            integer(forward["forward_id"], "forward_id")
            require(forward["round_id"] == fid and forward["forward_id"] == fid, "noncontiguous forward/round ID")
            require(isinstance(forward["parts"], list), "parts must be an array")
            seen, rows = set(), 0
            for part in forward["parts"]:
                fields(part, "stream_id position tape_offset offered kept", "part")
                sid = integer(part["stream_id"], "part.stream_id", 0, 3)
                require(sid in active and sid not in seen, "duplicate or inactive part stream")
                seen.add(sid)
                offered = integer(part["offered"], "offered", 1, 16)
                kept = integer(part["kept"], "kept", 1, offered)
                offset = integer(part["tape_offset"], "tape_offset")
                position = integer(part["position"], "position")
                stream = fixture["streams"][sid]
                require(offset == offsets[sid], "tape offset must advance by kept rows only")
                require(position == len(stream["prompt_tokens"]) + offset, "absolute position mismatch")
                require(offset + offered <= len(stream["tape_tokens"]), "insufficient lookahead tape")
                require(offset + kept < len(stream["tape_tokens"]), "missing next forced anchor")
                offsets[sid] += kept
                rows += offered
            require(seen == active, "forward must contain every active stream exactly once")
            require(rows <= 32, "shared forward exceeds 32 offered rows")
    return offsets


def validate_fixture(fixture, allow_synthetic=False, expected=None):
    fields(fixture, "schema fixture_id kind family provenance config streams preparation forwards tokens_sha256 "
           "schedule_sha256 config_sha256 fixture_sha256", "fixture")
    require(fixture["schema"] == SCHEMA, "unsupported fixture schema")
    string(fixture["fixture_id"], "fixture_id")
    require(fixture["kind"] in ("synthetic", "campaign"), "invalid fixture kind")
    require(fixture["kind"] == "campaign" or allow_synthetic, "synthetic fixture refused for device/campaign use")
    require(fixture["family"] in ("prose", "code", "pi"), "invalid fixture family")
    vocab, sources = _provenance(fixture)
    _config(fixture)
    _streams(fixture, vocab, sources)
    _schedule(fixture)
    validate_pool(fixture)
    copy = seal_fixture({k: v for k, v in fixture.items() if not k.endswith("_sha256")})
    for key in ("tokens_sha256", "schedule_sha256", "config_sha256", "fixture_sha256"):
        hex_id(fixture[key], key)
        require(fixture[key] == copy[key], f"{key} mismatch")
    bindings = {k: fixture[k] for k in ("fixture_sha256", "tokens_sha256", "schedule_sha256", "config_sha256")}
    bindings.update({"target_revision": fixture["provenance"]["target_revision"],
                     "drafter_revision": fixture["provenance"]["drafter_revision"],
                     "tokenizer_revision": fixture["provenance"]["tokenizer"]["revision"],
                     "template_sha256": fixture["provenance"]["tokenizer"]["template_sha256"],
                     "tokenizer_files_sha256": digest(fixture["provenance"]["tokenizer"]["files"]),
                     "vocab_size": vocab})
    for key, value in (expected or {}).items():
        require(key in bindings and type(bindings[key]) is type(value) and bindings[key] == value,
                f"expected {key} mismatch")
    return fixture


def load_fixture(path, allow_synthetic=False, expected=None):
    return validate_fixture(read_json(path), allow_synthetic, expected)


def offered_tokens(fixture, part):
    stream = fixture["streams"][part["stream_id"]]
    return stream["tape_tokens"][part["tape_offset"]:part["tape_offset"] + part["offered"]]


def next_anchor(fixture, part):
    return fixture["streams"][part["stream_id"]]["tape_tokens"][part["tape_offset"] + part["kept"]]
