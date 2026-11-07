"""Invented retained evidence exercises offline gates, never real device execution."""
import copy
import tempfile
import unittest
from pathlib import Path

from tools.perf.glm_replay import ReplayError
from tools.perf.glm_replay.common import digest, file_digest, write_json
from tools.perf.glm_replay.server_neutrality import produce, validate_server_gate


class ServerGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"harness_commit": "1" * 40, "fatbin_sha256": "2" * 64,
                       "binary_sha256": "3" * 64}
        self.original, self.candidate = self.root / "original", self.root / "instrumented"
        self.original.write_bytes(b"invented original host-test server")
        self.candidate.write_bytes(b"invented instrumented host-test server")
        identity = {"execution": "device", "engine": "native", "harness_commit": "1" * 40,
                    "fatbin_sha256": "2" * 64, "original_server_sha256": file_digest(self.original),
                    "instrumented_server_sha256": file_digest(self.candidate), "completed_ranks": [0, 1]}
        case = {"case_id": "host-control", "request_sha256": "4" * 64, "config_sha256": "5" * 64,
                "original_tokens": [11, 22, 33], "instrumented_tokens": [11, 22, 33]}
        self.tokens = {"schema": "glm-replay-server-tokens-v1", **identity, "cases": [case]}
        policy = {"schema": "glm-replay-server-policy-v1", "minimum_pairs": 3, "max_slowdown_ppm": 50_000}
        policy["predeclared_policy_sha256"] = digest(policy)
        self.performance = {"schema": "glm-replay-server-performance-v1", **identity, "policy": policy,
                            "samples": [{"case_id": "host-control", "pair_id": str(i), "start_order": i % 2,
                                         "request_sha256": "4" * 64, "config_sha256": "5" * 64,
                                         "original_elapsed_ns": 1000, "instrumented_elapsed_ns": 1020}
                                        for i in range(3)]}
        self.output = self.root / "gate.json"

    def ref(self, path):
        return {"path": str(path), "sha256": file_digest(path)}

    def run_producer(self):
        token_path, perf_path = self.root / "tokens.json", self.root / "performance.json"
        write_json(token_path, self.tokens)
        write_json(perf_path, self.performance)
        return produce(self.output, self.ref(token_path), self.ref(perf_path), self.original,
                       self.candidate, self.source, "native")

    def test_offline_producer_binds_real_server_artifacts_not_replay_binary(self):
        gate = self.run_producer()
        self.assertNotEqual(gate["instrumented_server_sha256"], self.source["binary_sha256"])
        self.assertEqual(validate_server_gate(self.ref(self.output), self.source, "native", [self.root]), gate)
        with self.assertRaisesRegex(ReplayError, "overwrite"):
            self.run_producer()

    def test_token_mismatch_cannot_generate_a_pass(self):
        self.tokens["cases"][0]["instrumented_tokens"][1] = 23
        with self.assertRaisesRegex(ReplayError, "tokens differ"):
            self.run_producer()
        self.assertFalse(self.output.exists())

    def test_unrun_or_missing_rank_evidence_is_refused(self):
        self.tokens["execution"] = "unrun"
        with self.assertRaisesRegex(ReplayError, "unrun"):
            self.run_producer()
        self.tokens["execution"] = "device"
        self.tokens["completed_ranks"] = [0]
        with self.assertRaisesRegex(ReplayError, "both ranks"):
            self.run_producer()

    def test_wrong_server_source_or_work_identity_is_refused(self):
        self.tokens["instrumented_server_sha256"] = "6" * 64
        with self.assertRaisesRegex(ReplayError, "identity mismatch"):
            self.run_producer()
        self.tokens["instrumented_server_sha256"] = file_digest(self.candidate)
        self.performance["samples"][0]["request_sha256"] = "6" * 64
        with self.assertRaisesRegex(ReplayError, "work differs"):
            self.run_producer()

    def test_policy_threshold_pairs_and_order_are_enforced(self):
        saved = copy.deepcopy(self.performance)
        for sample in self.performance["samples"]:
            sample["instrumented_elapsed_ns"] = 1060
        with self.assertRaisesRegex(ReplayError, "slowdown limit"):
            self.run_producer()
        self.performance = copy.deepcopy(saved)
        self.performance["samples"].pop()
        with self.assertRaisesRegex(ReplayError, "independent server pairs"):
            self.run_producer()
        self.performance = copy.deepcopy(saved)
        for sample in self.performance["samples"]:
            sample["start_order"] = 0
        with self.assertRaisesRegex(ReplayError, "counterbalanced"):
            self.run_producer()
        self.performance = copy.deepcopy(saved)
        self.performance["policy"]["max_slowdown_ppm"] = 60_000
        with self.assertRaisesRegex(ReplayError, "policy digest"):
            self.run_producer()

    def test_mutated_evidence_and_paths_are_refused_even_with_gate_rehashed(self):
        gate = self.run_producer()
        gate["token_evidence"]["sha256"] = "9" * 64
        write_json(self.output, gate)
        with self.assertRaisesRegex(ReplayError, "artifact hash"):
            validate_server_gate(self.ref(self.output), self.source, "native", [self.root])
        with self.assertRaisesRegex(ReplayError, "permitted roots"):
            validate_server_gate(self.ref(self.output), self.source, "native", [])


if __name__ == "__main__":
    unittest.main()
