"""Host negatives for both-rank completion and truthful measurement output gating."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError
from tools.perf.glm_replay.python_receipts import build_record, publish_local, write_exclusive
from tools.perf.glm_replay.python_runtime import completed


class ReceiptTests(unittest.TestCase):
    def test_rank_one_incomplete_is_refused(self):
        expected = [3, 12, 0, 0]
        with self.assertRaisesRegex(ReplayError, "disagreement"):
            completed([[0, *expected], [1, 2, 8, 0, 0]], expected)

    def test_rank_one_missing_is_refused(self):
        with self.assertRaisesRegex(ReplayError, "missing rank"):
            completed([[0, 3, 12, 0, 0]], [3, 12, 0, 0])

    def test_duplicate_rank_zero_is_refused(self):
        with self.assertRaisesRegex(ReplayError, "disagreement"):
            completed([[0, 3, 12, 0, 0]] * 2, [3, 12, 0, 0])

    def test_lost_or_proposer_rank_delta_is_refused(self):
        for tail in ((1, 0), (0, 1)):
            with self.assertRaisesRegex(ReplayError, "disagreement"):
                completed([[0, 3, 12, 0, 0], [1, 3, 12, *tail]], [3, 12, 0, 0])

    def test_two_completed_ranks_accept_real_count_vector(self):
        self.assertEqual(completed([[0, 3, 12, 0, 0], [1, 3, 12, 0, 0]], [3, 12, 0, 0]), [0, 1])

    def test_both_ranks_cannot_agree_to_lost_or_proposer_work(self):
        for expected in ([3, 12, 1, 0], [3, 12, 0, 1], [0, 0, 0, 0]):
            with self.assertRaisesRegex(ReplayError, "completion audit"):
                completed([[0, *expected], [1, *expected]], expected)

    def test_no_result_construction_after_partial_rank_completion(self):
        runtime = SimpleNamespace(fixture={}, meta={})
        with self.assertRaisesRegex(ReplayError, "rank completion"):
            build_record(runtime, {"completed_ranks": [0]}, {}, "0" * 64)

    def test_no_result_with_unmeasured_phase_intervals(self):
        runtime = SimpleNamespace(fixture={}, meta={})
        with self.assertRaisesRegex(ReplayError, "missing separate anchor"):
            build_record(runtime, {"completed_ranks": [0, 1], "phases": None}, {}, "0" * 64)

    def test_unknown_correctness_cannot_be_set_true_by_output_writer(self):
        runtime = SimpleNamespace(fixture={"forwards": [1]}, meta={}, rank=0)
        block = {"completed_ranks": [0, 1], "phases": None, "frames": [1]}
        anchor = {"ranks": [{"phases_ns": [{}]}]}
        with self.assertRaisesRegex(ReplayError, "gate not passed"):
            build_record(runtime, block, {"correctness": "unrun"}, "0" * 64, anchor)

    def test_no_file_when_peer_output_missing(self):
        runtime = SimpleNamespace(nccl=SimpleNamespace(store=None), rank=0, meta={"ids": {"block_id": "test"}})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "receipt.jsonl"
            with patch("tools.perf.glm_replay.python_receipts.exchange", return_value=[{"rank": 0}]):
                with self.assertRaisesRegex(ReplayError, "missing/duplicate"):
                    publish_local(runtime, output, {"rank": 0})
            self.assertFalse(output.exists())

    def test_output_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "sealed.jsonl"
            output.write_bytes(b"sealed\n")
            with self.assertRaises(FileExistsError):
                write_exclusive(output, [{"different": True}])
            self.assertEqual(output.read_bytes(), b"sealed\n")


if __name__ == "__main__":
    unittest.main()
