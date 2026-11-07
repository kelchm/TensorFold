"""Driver-free checks of real adapter startup helpers and full production graph capture."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.perf.glm_replay.common import ReplayError
from tools.perf.glm_replay.common import digest
from tools.perf.glm_replay.python_runtime import session_provenance
from tools.perf.glm_replay.python_startup import capture_bank, launch_settings, open_transport
from tools.perf.glm_replay.tests.python_evidence_doubles import RUNTIME, fixture_case, metadata
from tools.perf.glm_replay.python_provenance import gate_binding


class StartupTests(unittest.TestCase):
    def launch(self, rows=2048, grid=0, credit=0):
        production = SimpleNamespace(prompt_grid=Mock(return_value=grid))
        result = launch_settings(production, SimpleNamespace(credit=lambda: credit),
                                 {"prefill_rows": rows, "grid": 0}, SimpleNamespace(PREFILL_ROWS=2048))
        production.prompt_grid.assert_called_once_with(rows)
        return result

    def test_actual_helpers_derive_launch_and_record_calibration_omission(self):
        with patch.dict(os.environ, {}, clear=True):
            startup = self.launch()
        self.assertEqual(startup["prefill_rows"], 2048)
        self.assertIn("Drafter.propose", startup["deviations"]["calibration"])

    def test_environment_and_grid_must_equal_metadata(self):
        with patch.dict(os.environ, {"TF_GLM_PREFILL_ROWS": "4096"}, clear=True):
            with self.assertRaisesRegex(ReplayError, "metadata differs"):
                self.launch()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ReplayError, "metadata differs"):
                self.launch(grid=256)

    def test_nonzero_display_credit_is_not_silently_dropped(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ReplayError, "display KV credit"):
                self.launch(credit=1024)

    def test_capture_uses_no_subset_argument_and_requires_all_32_graphs(self):
        verifier = SimpleNamespace(graphs={}, capture=Mock())
        verifier.capture.side_effect = lambda: verifier.graphs.update({n: object() for n in range(1, 33)})
        self.assertEqual(capture_bank(verifier, [1, 4, 8, 16, 32]), list(range(1, 33)))
        verifier.capture.assert_called_once_with()

    def test_subset_or_unexpected_capture_bank_is_refused(self):
        for bank in ([1, 4, 8, 16, 32], list(range(1, 34))):
            verifier = SimpleNamespace(graphs={n: object() for n in bank}, capture=Mock())
            with self.assertRaisesRegex(ReplayError, "graph bank mismatch"):
                capture_bank(verifier, [1, 4, 8])

    def test_nccl_barrier_precedes_roce_constructor(self):
        calls = []
        nccl = SimpleNamespace(barrier=lambda: calls.append("barrier"))
        def roce(comm, rank, world):
            self.assertIs(comm, nccl)
            calls.append(("roce", rank, world))
            return "observed_roce"
        with patch("tools.perf.glm_replay.python_startup.importlib.import_module",
                   return_value=SimpleNamespace(RoceComm=roce)):
            self.assertEqual(open_transport(nccl, 1, "roce"), "observed_roce")
        self.assertEqual(calls, ["barrier", ("roce", 1, 2)])

    def test_real_session_nonce_changes_without_changing_gate_runtime_identity(self):
        fixture = fixture_case()
        meta = metadata(fixture)
        values = []
        for pair in (["a" * 64, "b" * 64], ["c" * 64, "d" * 64]):
            with patch("tools.perf.glm_replay.python_runtime.exchange", return_value=pair):
                values.append(session_provenance(None, 0, meta))
        self.assertNotEqual(values[0]["session_nonce"], values[1]["session_nonce"])
        self.assertEqual(values[0]["campaign_sha256"], digest(meta))
        self.assertEqual(values[0]["checkpoints"]["target"], [digest(files) for files in meta["checkpoints"]["target"]["ranks"]])
        self.assertNotIn("session_nonce", gate_binding(fixture, meta, RUNTIME))


if __name__ == "__main__":
    unittest.main()
