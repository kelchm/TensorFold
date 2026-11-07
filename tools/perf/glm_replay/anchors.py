"""Authenticate retained phase anchors independently of headline wall receipts."""
import hashlib
import re
import struct
from pathlib import Path

from .common import digest, fields, hex_id, integer, number, parse_json, require

PHASES = "staging target tap_bookkeeping selection target_commit context_commit".split()
IDS = "run_id pair_id start_id start_order block_id cell_id".split()


class Anchors:
    def __init__(self, roots=None, blobs=None):
        self.blobs = {} if blobs is None else dict(blobs)
        self.checked = {}
        for key, raw in self.blobs.items():
            hex_id(key, "anchor digest")
            require(isinstance(raw, bytes) and hashlib.sha256(raw).hexdigest() == key,
                    "in-memory anchor hash mismatch")
        if roots is None:
            roots = []
        if isinstance(roots, (str, Path)):
            roots = [roots]
        for name in roots:
            root = Path(name).resolve(strict=True)
            require(root.is_dir(), "anchor root must be a directory")
            for path in root.rglob("*.json"):
                if not (path.name.endswith(".anchor.json") or re.fullmatch(r"[0-9a-f]{64}\.json", path.name)):
                    continue
                require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink in anchor store")
                require(path.resolve().is_relative_to(root), "anchor escaped root")
                raw = path.read_bytes()
                key = hashlib.sha256(raw).hexdigest()
                self.blobs.setdefault(key, raw)

    def check(self, record):
        key = record["timing_source"]["anchor_sha256"]
        require(key in self.blobs, "missing retained phase anchor")
        if key not in self.checked:
            self.checked[key] = parse_json(self.blobs[key])
        value = self.checked[key]
        if record["arm"] == "P":
            check_python(value, record)
        else:
            check_native(value, record)


def common(value, record):
    require(value["completed_ranks"] == [0, 1], "anchor missing both ranks")
    require(value["timing_source"] == "separate_observer_on_anchor", "wrong anchor timing source")
    require(value["ids"] == {key: record[key] for key in IDS}, "anchor block identity mismatch")
    hex_id(value["logical_evidence_sha256"], "anchor logical evidence")
    for key in ("session_nonce", "campaign_sha256", "checkpoints"):
        require(value[key] == record[key], f"anchor {key} mismatch")


def check_python(value, record):
    fields(value, "schema bindings ids logical_evidence_sha256 timing_source completed_ranks ranks "
           "session_nonce campaign_sha256 checkpoints", "Python anchor")
    require(value["schema"] == "glm-replay-python-phase-anchor-v1", "Python anchor schema mismatch")
    common(value, record)
    binding = value["bindings"]
    for key in ("fixture_sha256", "schedule_sha256", "tokens_sha256", "config_sha256"):
        require(binding.get(key) == record[key], f"anchor {key} mismatch")
    require(binding.get("source_sha256") == digest(record["source"])
            and binding.get("runtime_sha256") == record["effective_config"]["runtime_sha256"]
            and binding.get("arm") == "P", "anchor source/runtime mismatch")
    require(isinstance(value["ranks"], list) and [r.get("rank") for r in value["ranks"]] == [0, 1],
            "anchor rank order mismatch")
    for rank in value["ranks"]:
        fields(rank, "rank wall_ns phases_ns kept_outputs_sha256 final_sha256", "Python rank anchor")
        integer(rank["wall_ns"], "anchor wall", 1)
        for key in ("kept_outputs_sha256", "final_sha256"):
            hex_id(rank[key], key)
    phases = value["ranks"][record["rank"]]["phases_ns"]
    require(phases == [f["phases_ns"] for f in record["forwards"]], "anchor phases differ from result")


def check_native(value, record):
    fields(value, "schema fixture_sha256 schedule_sha256 runtime_sha256 logical_evidence_sha256 source ids "
           "kv_shard timing_source completed_ranks completion headline_completion session_nonce campaign_sha256 checkpoints", "native anchor")
    require(value["schema"] == "glm-replay-native-phase-anchor-v1", "native anchor schema mismatch")
    common(value, record)
    for key in ("fixture_sha256", "schedule_sha256", "source"):
        require(value[key] == record[key], f"anchor {key} mismatch")
    require(value["runtime_sha256"] == record["effective_config"]["runtime_sha256"]
            and value["kv_shard"] is (record["arm"] == "B"), "anchor runtime/arm mismatch")
    profile, headline = value["completion"], value["headline_completion"]
    for block in (profile, headline):
        fields(block, "ranks elapsed_ns records timings", "native anchor completion")
        integer(block["elapsed_ns"], "native anchor wall", 1)
        require(len(block["ranks"]) == 2 and len(block["records"]) == 2 * len(record["forwards"]),
                "incomplete native anchor completion")
    rank_receipt = headline["ranks"][record["rank"]]
    require(headline["timings"] == [] and rank_receipt["elapsed_ns"] == record["wall_ns"],
            "native headline observer or wall mismatch")
    require(rank_receipt["checks"] == record["layout_checks"], "headline layout-check mismatch")
    count = len(record["forwards"])
    require(len(profile["timings"]) == 2 * count, "missing native anchor timing")
    for actual, timed in zip(record["forwards"], profile["timings"][record["rank"] * count:][:count]):
        fields(timed, "id ms", "native anchor timing")
        require(timed["id"] == actual["forward_id"] and len(timed["ms"]) == 6, "native anchor timing order")
        for key, ms in zip(PHASES, timed["ms"]):
            number(ms, "native anchor interval")
            require(ms <= 3.4e38, "native anchor interval overflow")
            ns = struct.unpack("f", struct.pack("f", ms))[0] * 1e6
            require(actual["phases_ns"][key] == ns, "anchor phases differ from result")
