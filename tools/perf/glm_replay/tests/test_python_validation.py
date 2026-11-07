"""Host negatives for poisoned lookahead, unsynchronized schedules and evidence-bound gates."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError, digest
from tools.perf.glm_replay.python_provenance import metadata_check
from tools.perf.glm_replay.python_evidence import validate_receipt
from tools.perf.glm_replay.tests.python_evidence_doubles import (
    RUNTIME, fixture_case, metadata, produced_validation, rehash,
)
from tools.perf.glm_replay.python_receipts import measurement_gates, measure_block
from tools.perf.glm_replay.python_validation import capture_forward, validate_device


class ValidationTests(unittest.TestCase):
    def test_light_capture_reads_logits_taps_without_full_cache(self):
        frame = SimpleNamespace(scheduled={"parts": [{"stream_id": 0, "kept": 1}]},
                                verified=SimpleNamespace(logits=[["logits"]], taps=[["taps"]]), sampled=[[9]])
        runtime = SimpleNamespace(torch=object())
        with patch("tools.perf.glm_replay.python_validation.tensor_hash", side_effect=lambda v, _: v):
            with patch("tools.perf.glm_replay.python_validation.logical_state") as full:
                result = capture_forward(runtime, frame, [])
        full.assert_not_called()
        self.assertEqual(result["states"], {})
        self.assertEqual(result["kept"]["0"]["selected"], [9])

    def test_natural_unsynchronized_race_prevents_correctness_gate(self):
        fixture = {"forwards": [{"forward_id": 0, "parts": [{"stream_id": 0, "offered": 1, "kept": 1}]}]}
        snap = {"states": {"0": {"ring": "correct"}}, "kept": {
            "0": {"logits": ["same"], "taps": ["same"], "selected": [9]}}}
        frame = SimpleNamespace(graph_replays=1, eager_fallbacks=0, target_commits=1,
                                context_commits=1, ingest_calls=1)
        base = {"initial_states": {}, "snapshots": [snap], "frames": [frame],
                "kept_outputs": [{"0": [9]}], "final": snap}
        calls = []
        def run(**kwargs):
            calls.append(kwargs)
            result = copy.deepcopy(base)
            if kwargs.get("final_capture") and not kwargs.get("poison"):
                result["kept_outputs"] = [{"0": [8]}]
            return result
        runtime = SimpleNamespace(run=run, fixture=fixture)
        with self.assertRaisesRegex(ReplayError, "unsynchronized schedule changed kept outputs"):
            validate_device(runtime, {})
        self.assertTrue(any(c.get("poison") and c.get("final_capture") for c in calls))
        self.assertTrue(any(not c.get("poison") and c.get("final_capture") for c in calls))

    def gate(self):
        fixture = fixture_case(partial=True)
        meta = metadata(fixture)
        return fixture, meta, RUNTIME, produced_validation(fixture, meta)

    def test_correctness_booleans_without_logical_evidence_are_refused(self):
        fixture, meta, runtime, gate = self.gate()
        del gate["logical_evidence"]
        with patch("tools.perf.glm_replay.python_receipts.artifact_receipt", return_value=gate):
            with self.assertRaisesRegex(ReplayError, "validation receipt"):
                measurement_gates(fixture, meta, runtime, Path("."))

    def test_changed_evidence_cannot_keep_prior_gate_sha(self):
        fixture, meta, runtime, gate = self.gate()
        gate["ranks"][0]["evidence"]["expected_kept_outputs"] = [{"0": [8]}]
        with patch("tools.perf.glm_replay.python_receipts.artifact_receipt", return_value=gate):
            with self.assertRaisesRegex(ReplayError, "logical evidence hash"):
                measurement_gates(fixture, meta, runtime, Path("."))

    def test_rehashed_outer_gate_does_not_hide_changed_rank_evidence(self):
        fixture, meta, runtime, gate = self.gate()
        gate["ranks"][0]["evidence"]["expected_kept_outputs"] = [{"0": [8]}]
        gate["logical_evidence_sha256"] = digest(gate["logical_evidence"])
        with patch("tools.perf.glm_replay.python_receipts.artifact_receipt", return_value=gate):
            with self.assertRaisesRegex(ReplayError, "rank logical evidence hash"):
                measurement_gates(fixture, meta, runtime, Path("."))

    def test_measured_kept_outputs_are_checked_after_observer_off_completion(self):
        expected = {"expected_kept_outputs": [{"0": [9]}], "expected_final": {"state": "correct"}}
        gate = {"ranks": [{"evidence": expected}]}
        runs = []
        def run(**kwargs):
            runs.append(kwargs)
            return {"kept_outputs": [{"0": [9 if kwargs["enabled"] else 8]}], "final": expected["expected_final"]}
        runtime = SimpleNamespace(meta={"validation_gate": {}}, rank=0, run=run)
        with patch("tools.perf.glm_replay.python_receipts.artifact_receipt", return_value=gate):
            with self.assertRaisesRegex(ReplayError, "measured kept outputs"):
                measure_block(runtime, {"correctness": True, "timing_neutral": True}, ".")
        self.assertEqual([r["enabled"] for r in runs], [True, False])
        self.assertTrue(all(r["final_capture"] for r in runs))

    def test_python_server_gate_is_explicitly_not_applicable(self):
        fixture = fixture_case()
        meta = metadata(fixture)
        metadata_check(meta, fixture)
        meta["server_gate"] = {"path": "old-server.json", "sha256": "a" * 64}
        with self.assertRaisesRegex(ReplayError, "server gate must be null"):
            metadata_check(meta, fixture)

    def test_poison_count_is_actual_and_zero_for_full_keep(self):
        for partial in (False, True):
            fixture = fixture_case(partial=partial)
            meta = metadata(fixture)
            validation = produced_validation(fixture, meta)
            self.assertEqual(validation["coverage"]["partial"], partial)
            for rank in validation["ranks"]:
                self.assertEqual(rank["evidence"]["poisoned_rows"] > 0, partial)
            validate_receipt(validation, fixture, meta, RUNTIME)

    def test_rehashed_poison_counts_cannot_invent_partial_coverage(self):
        fixture, meta, runtime, validation = self.gate()
        validation["ranks"][0]["evidence"]["poisoned_rows"] = 0
        with self.assertRaisesRegex(ReplayError, "poisoned discarded-row"):
            validate_receipt(rehash(validation), fixture, meta, runtime)

    def test_rehashed_opaque_or_incorrect_logical_payload_is_refused(self):
        for mutate in (
            lambda e: e["validated_forwards"][0]["states"]["0"].update(position=0),
            lambda e: e["dispatch"][0].__setitem__(0, 0),
            lambda e: e["runs"][1].update(snapshots_sha256="0" * 64),
            lambda e: e["oracle_rows"][0]["kept"]["0"]["selected"].__setitem__(0, 999999),
        ):
            fixture, meta, runtime, validation = self.gate()
            mutate(validation["ranks"][0]["evidence"])
            with self.assertRaises(ReplayError):
                validate_receipt(rehash(validation), fixture, meta, runtime)


if __name__ == "__main__":
    unittest.main()
