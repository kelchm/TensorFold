"""Planted retained-file and attribution faults; all receipts are invented host controls."""
import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from tools.perf.glm_replay.common import ReplayError, canonical_bytes, digest
from tools.perf.glm_replay.generate import build_fixture
from tools.perf.glm_replay.results import validate_results
from tools.perf.glm_replay.tests.support import campaign, receipt

IDS = "run_id pair_id start_id start_order block_id cell_id".split()


def anchor(records):
    first = records[0]
    shared = {"ids": {k: first[k] for k in IDS}, "logical_evidence_sha256": "a" * 64,
              "completed_ranks": [0, 1], "timing_source": "separate_observer_on_anchor",
              **{k: first[k] for k in ("session_nonce", "campaign_sha256", "checkpoints")}}
    if first["arm"] == "P":
        return {**shared, "schema": "glm-replay-python-phase-anchor-v1",
                "bindings": {**{k: first[k] for k in ("fixture_sha256", "schedule_sha256", "tokens_sha256", "config_sha256")},
                             "source_sha256": digest(first["source"]), "runtime_sha256": first["effective_config"]["runtime_sha256"],
                             "arm": "P"},
                "ranks": [{"rank": r["rank"], "wall_ns": 1200000,
                           "phases_ns": [f["phases_ns"] for f in r["forwards"]],
                           "kept_outputs_sha256": "d" * 64, "final_sha256": "e" * 64} for r in records]}
    ms = 0.0001
    ns = struct.unpack("f", struct.pack("f", ms))[0] * 1e6
    for record in records:
        for forward in record["forwards"]:
            forward["phases_ns"] = {k: ns for k in forward["phases_ns"]}
    count = len(first["forwards"])
    headline = {"ranks": [{"elapsed_ns": r["wall_ns"], "checks": r["layout_checks"]} for r in records],
                "elapsed_ns": first["wall_ns"], "records": [{}] * (2 * count), "timings": []}
    profile = {**headline, "elapsed_ns": 1200000,
               "timings": [{"id": i, "ms": [ms] * 6} for _ in (0, 1) for i in range(count)]}
    return {**shared, "schema": "glm-replay-native-phase-anchor-v1",
            **{k: first[k] for k in ("fixture_sha256", "schedule_sha256", "source")},
            "runtime_sha256": first["effective_config"]["runtime_sha256"], "kv_shard": first["arm"] == "B",
            "completion": profile, "headline_completion": headline}


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = build_fixture(repetitions=2)
        self.fixtures = {self.fixture["fixture_sha256"]: self.fixture}

    def validate(self, records, **kwargs):
        return validate_results(records, self.fixtures, allow_synthetic=True, **kwargs)

    def blobs(self, records):
        values, blobs = [], {}
        for i in range(0, len(records), 2):
            value = copy.deepcopy(anchor(records[i:i + 2]))
            raw = canonical_bytes(value) + b"\n"
            sha = hashlib.sha256(raw).hexdigest()
            for record in records[i:i + 2]:
                record["timing_source"]["anchor_sha256"] = sha
            values.append(value)
            blobs[sha] = raw
        return values, blobs

    def test_native_retained_files_match_actual_f32_phase_conversion_and_both_headline_walls(self):
        records = campaign(self.fixture, ratios=(1.1,), blocks=1)
        _, blobs = self.blobs(records)
        with tempfile.TemporaryDirectory() as temp:
            for index, raw in enumerate(blobs.values()):
                Path(temp, f"block-{index}.anchor.json").write_bytes(raw)
            self.validate(records, anchors_root=temp)
            records[1]["forwards"][0]["phases_ns"]["target"] += 1
            with self.assertRaisesRegex(ReplayError, "phases differ"):
                self.validate(records, anchors_root=temp)

    def test_python_in_memory_anchor_and_content_hash_are_checked_before_publication(self):
        records = [receipt(self.fixture, "P", rank=i) for i in (0, 1)]
        _, blobs = self.blobs(records)
        self.validate(records, anchors=blobs)
        key = next(iter(blobs))
        blobs[key] += b"\n"
        with self.assertRaisesRegex(ReplayError, "in-memory anchor hash"):
            self.validate(records, anchors=blobs)

    def test_rehashed_anchor_with_wrong_block_runtime_rank_or_phases_fails(self):
        original = [receipt(self.fixture, "P", rank=i) for i in (0, 1)]
        for fault in ("ids", "runtime", "rank", "phases", "nonce"):
            records = copy.deepcopy(original)
            values, _ = self.blobs(records)
            value = values[0]
            if fault == "ids":
                value["ids"]["block_id"] = "other-block"
            elif fault == "runtime":
                value["bindings"]["runtime_sha256"] = "0" * 64
            elif fault == "rank":
                value["ranks"].reverse()
            elif fault == "nonce":
                value["session_nonce"] = "0" * 64
            else:
                value["ranks"][1]["phases_ns"][0]["target"] += 1
            raw = canonical_bytes(value) + b"\n"
            sha = hashlib.sha256(raw).hexdigest()
            for record in records:
                record["timing_source"]["anchor_sha256"] = sha
            with self.subTest(fault=fault), self.assertRaises(ReplayError):
                self.validate(records, anchors={sha: raw})

    def test_missing_file_and_headline_with_observer_or_wrong_layout_delta_fail(self):
        records = campaign(self.fixture, ratios=(1.1,), blocks=1)
        values, blobs = self.blobs(records)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ReplayError, "missing retained"):
                self.validate(records, anchors_root=temp)
        values[0]["headline_completion"]["ranks"][1]["checks"] = 1
        raw = canonical_bytes(values[0]) + b"\n"
        sha = hashlib.sha256(raw).hexdigest()
        for record in records[:2]:
            record["timing_source"]["anchor_sha256"] = sha
        blobs[sha] = raw
        with self.assertRaisesRegex(ReplayError, "layout-check mismatch"):
            self.validate(records, anchors=blobs)

    def test_two_invocations_cannot_share_a_start_or_rename_one_session(self):
        records = campaign(self.fixture, ratios=(1.1,), blocks=2)
        for record in records[4:6]:
            record["session_nonce"] = "f" * 64
        with self.assertRaisesRegex(ReplayError, "inconsistent process-start"):
            self.validate(records)
        records = campaign(self.fixture, ratios=(1.1,), blocks=2)
        for record in records[4:6]:
            record["start_id"] = "renamed-start"
        with self.assertRaisesRegex(ReplayError, "one session nonce"):
            self.validate(records)
