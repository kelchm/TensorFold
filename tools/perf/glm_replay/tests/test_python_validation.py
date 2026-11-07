"""Host negatives for poisoned lookahead, unsynchronized schedules and evidence-bound gates."""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.perf.glm_replay.common import ReplayError, digest
from tools.perf.glm_replay.python_provenance import HASHES, gate_binding
from tools.perf.glm_replay.python_receipts import measurement_gates, measure_block, server_gate
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
        fixture = {**{k: "1" * 64 for k in HASHES.split()}, "forwards": [{}]}
        meta = {"source": {}, "source_tree_sha256": "2" * 64, "checkpoints": {},
                "validation_gate": {"path": "validation.json", "sha256": "3" * 64}}
        runtime = digest(["a" * 64, "b" * 64])
        binding = gate_binding(fixture, meta, runtime)
        evidence = {"poisoned_lookahead": True, "unsynchronized_poisoned_and_natural": True,
                    "expected_kept_outputs": [{"0": [9]}]}
        ranks = [{"rank": r, "status": "pass", "runtime_rank_sha256": char * 64,
                  "evidence": copy.deepcopy(evidence), "evidence_sha256": digest(evidence)}
                 for r, char in enumerate(("a", "b"))]
        logical = {"bindings": binding, "ranks": ranks}
        gate = {"schema": "glm-replay-python-validation-v1", "status": "pass", "execution": "device",
                "bindings": binding, "completed_ranks": [0, 1], "ranks": ranks,
                "checks": {"correctness": True, "timing_neutral": True, "fresh_prefill": True},
                "logical_evidence": logical, "logical_evidence_sha256": digest(logical)}
        return fixture, meta, runtime, gate

    def test_correctness_booleans_without_logical_evidence_are_refused(self):
        fixture, meta, runtime, gate = self.gate()
        del gate["logical_evidence"]
        with patch("tools.perf.glm_replay.python_receipts.artifact_receipt", return_value=gate):
            with self.assertRaisesRegex(ReplayError, "logical evidence"):
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

    def test_server_gate_uses_shared_deferred_validator_without_replay_runtime(self):
        validate = Mock(return_value={"status": "pass", "execution": "host-double"})
        module = SimpleNamespace(validate_server_gate=validate)
        meta = {"server_gate": {"path": "server.json", "sha256": "1" * 64},
                "source": {"harness_commit": "2" * 40, "fatbin_sha256": "3" * 64}}
        with patch.dict(sys.modules, {"tools.perf.glm_replay.server_neutrality": module}):
            server_gate(meta, Path("receipts"))
        validate.assert_called_once_with(meta["server_gate"], meta["source"], "python", roots=(Path("receipts"),))


if __name__ == "__main__":
    unittest.main()
