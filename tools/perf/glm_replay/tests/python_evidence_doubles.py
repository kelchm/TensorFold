"""Deterministic host captures exercising real receipt producers; never device qualification."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay.common import digest, read_json, text_digest
from tools.perf.glm_replay.fixture import load_fixture, seal_fixture, validate_fixture
from tools.perf.glm_replay.python_evidence import poisoned_rows
from tools.perf.glm_replay.python_logical import initial_positions
from tools.perf.glm_replay.python_provenance import HASHES, REQUIRED_ENV, gate_binding
from tools.perf.glm_replay.python_validation import oracle_schedule, validate_device

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_RANKS = ["a" * 64, "b" * 64]
RUNTIME = digest(RUNTIME_RANKS)


def fixture_case(solo=False, partial=False, reverse=False, edge=False):
    fixture = load_fixture(ROOT / "fixtures/host-prose.json", allow_synthetic=True)
    if edge:
        for stream in fixture["streams"]:
            stream["rendered_prompt"] += " " * 512
            tokens = list(stream["rendered_prompt"].encode())
            stream["rendered_prompt_sha256"] = text_digest(stream["rendered_prompt"])
            stream["prompt_encoding"] = {"token_ids": tokens, "used_tokens": 510}
            stream["prompt_tokens"] = tokens[:510]
    if solo:
        fixture["config"]["active_streams"] = [0]
    offsets = [0] * 4
    for phase in ("preparation", "forwards"):
        for forward in fixture[phase]:
            if solo and phase == "forwards":
                forward["parts"] = forward["parts"][:1]
            for part in forward["parts"]:
                sid = part["stream_id"]
                if partial and phase == "forwards":
                    part["kept"] = max(1, part["offered"] - 1)
                part["position"] = len(fixture["streams"][sid]["prompt_tokens"]) + offsets[sid]
                part["tape_offset"] = offsets[sid]
                offsets[sid] += part["kept"]
            if reverse and phase == "forwards":
                forward["parts"].reverse()
    fixture["fixture_id"] = f"HOST-DOUBLE-{solo}-{partial}-{reverse}-{edge}"
    return validate_fixture(seal_fixture(fixture), allow_synthetic=True)


def metadata(fixture):
    meta = read_json(ROOT / "python_metadata.example.json")
    meta["status"] = "ready"
    meta["bindings"] = {k: fixture[k] for k in HASHES.split()}
    for key in meta["source"]:
        if meta["source"][key] is None:
            meta["source"][key] = "sha256:" + "1" * 64 if key == "image_digest" else "1" * (
                40 if key == "harness_commit" else 64)
    meta["launch"]["environment_sha256"] = [digest(REQUIRED_ENV)] * 2
    return meta


def state(sid, pos):
    sha = digest([sid, pos])
    return {"position": pos, "device_position": pos, "ring_from": max(0, pos - 2048), "index_from": pos // 4 * 4,
            "conv": sha, "recurrent": sha, **{key: [sha] * 11 for key in
            ("latent_codes", "latent_scales", "index_keys", "index_gates", "pooled_keys")},
            "ring_k": [sha] * 2, "ring_v": [sha] * 2}


class CaptureRuntime:
    def __init__(self, fixture, meta=None, rank=0):
        self.fixture, self.meta, self.rank = fixture, meta or metadata(fixture), rank
        self.runtime_ranks, self.runtime_sha256 = RUNTIME_RANKS, RUNTIME
        self.block_number, self.nccl = 0, SimpleNamespace(store=None)

    def run(self, enabled=False, capture=False, oracle=False, poison=False, final_capture=False):
        positions = initial_positions(self.fixture)
        initial = {sid: state(sid, pos) for sid, pos in positions.items()}
        schedule = oracle_schedule(self.fixture["forwards"]) if oracle else self.fixture["forwards"]
        snapshots, frames = [], []
        for forward in schedule:
            kept = {}
            for part in forward["parts"]:
                sid, pos, count = str(part["stream_id"]), part["position"], part["kept"]
                kept[sid] = {"logits": [digest([sid, pos + r, "logits"]) for r in range(count)],
                             "taps": [digest([sid, pos + r, "taps"]) for r in range(count)],
                             "selected": [(pos + r) % 256 for r in range(count)]}
                positions[sid] = pos + count
            states = {sid: state(sid, pos) for sid, pos in positions.items()}
            if oracle:
                sid = str(forward["parts"][0]["stream_id"])
                states = {sid: states[sid]} if forward["oracle_part_end"] else {}
            snapshots.append({"states": states, "kept": kept})
            count = sum(p["kept"] for p in forward["parts"])
            frames.append(SimpleNamespace(graph_replays=int(not oracle), eager_fallbacks=0,
                target_commits=count, context_commits=count, ingest_calls=1))
        self.block_number += 1
        return {"initial_states": initial, "snapshots": snapshots, "frames": frames,
                "kept_outputs": [{sid: p["selected"] for sid, p in row["kept"].items()} for row in snapshots],
                "final": snapshots[-1], "poisoned_rows": poisoned_rows(self.fixture) if poison else 0}


def produced_validation(fixture, meta=None):
    runtime = CaptureRuntime(fixture, meta)
    def exchange(store, label, rank, local):
        ranks = [copy.deepcopy(local) for _ in range(2)]
        for r, item in enumerate(ranks):
            item["rank"], item["runtime_rank_sha256"] = r, RUNTIME_RANKS[r]
        return ranks
    with patch("tools.perf.glm_replay.python_validation.exchange", side_effect=exchange):
        return validate_device(runtime, gate_binding(fixture, runtime.meta, RUNTIME))


def rehash(validation):
    for rank in validation["ranks"]:
        rank["evidence_sha256"] = digest(rank["evidence"])
        for key, field in (("snapshot_sha256", "validated_forwards"), ("oracle_sha256", "oracle_rows"),
                           ("dispatch_sha256", "dispatch")):
            rank[key] = digest(rank["evidence"][field])
    validation["logical_evidence"]["ranks"] = validation["ranks"]
    validation["logical_evidence_sha256"] = digest(validation["logical_evidence"])
    return validation
