"""Offline rejection controls for falsely complete, duplicated or unmatched measurements."""
import copy
import tempfile
import unittest
from pathlib import Path

from tools.perf.glm_replay import ReplayError, digest, validate_results
from tools.perf.glm_replay.common import canonical_bytes
from tools.perf.glm_replay.generate import build_fixture
from tools.perf.glm_replay.results import load_results
from tools.perf.glm_replay.tests.support import campaign, receipt


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = build_fixture(repetitions=2)
        cls.fixtures = {cls.fixture["fixture_sha256"]: cls.fixture}

    def setUp(self):
        self.records = campaign(self.fixture, ratios=(1.1,), blocks=1)

    def validate(self, records=None):
        return validate_results(self.records if records is None else records, self.fixtures, allow_synthetic=True)

    def change_config(self, key, value):
        self.records[0]["effective_config"][key] = value
        self.records[0]["effective_config_sha256"] = digest(self.records[0]["effective_config"])

    def test_complete_two_rank_paired_work(self):
        self.assertEqual(len(self.validate()), 4)
        with self.assertRaisesRegex(ReplayError, "synthetic fixture refused"):
            validate_results(self.records, self.fixtures)

    def test_shared_forward_counters_not_request_participation(self):
        self.records[2]["forwards"][0]["owner"]["layer_visits"] *= 4
        with self.assertRaisesRegex(ReplayError, "owner layer_visits mismatch"):
            self.validate()

    def test_rank_local_send_bytes_follow_solo_owner(self):
        fixture = build_fixture(active_streams=(2,), repetitions=2)
        records = campaign(fixture, ratios=(1.1,), blocks=1)
        validate_results(records, {fixture["fixture_sha256"]: fixture}, allow_synthetic=True)
        rank0, rank1 = records[2], records[3]
        self.assertEqual(rank0["forwards"][0]["owner"]["output_send_bytes"], 0)
        self.assertEqual(rank1["forwards"][0]["owner"]["query_send_bytes"], 0)
        rank0["forwards"][0]["owner"]["query_send_bytes"] *= 2
        with self.assertRaisesRegex(ReplayError, "owner query_send_bytes mismatch"):
            validate_results(records, {fixture["fixture_sha256"]: fixture}, allow_synthetic=True)

    def test_incomplete_extra_duplicate_and_permuted_forward_receipts(self):
        for mode in ("missing", "extra", "duplicate", "parts"):
            with self.subTest(mode=mode):
                records = copy.deepcopy(self.records)
                forwards = records[0]["forwards"]
                if mode == "missing":
                    forwards.pop()
                elif mode == "extra":
                    forwards.append(copy.deepcopy(forwards[-1]))
                elif mode == "duplicate":
                    forwards[1] = copy.deepcopy(forwards[0])
                else:
                    forwards[0]["parts"].reverse()
                with self.assertRaises(ReplayError):
                    self.validate(records)

    def test_missing_rank_and_duplicate_block_are_rejected(self):
        with self.assertRaisesRegex(ReplayError, "incomplete or duplicate rank"):
            self.validate(self.records[:-1])
        with self.assertRaisesRegex(ReplayError, "duplicate rank/block"):
            self.validate(self.records + [copy.deepcopy(self.records[0])])
        with self.assertRaisesRegex(ReplayError, "unmatched native A/B"):
            self.validate(self.records[:2])

    def test_both_rank_completion_and_state_checks_are_mandatory(self):
        self.records[0]["completed_ranks"] = [0]
        with self.assertRaisesRegex(ReplayError, "both-rank completion"):
            self.validate()
        self.records[0]["completed_ranks"] = [0, 1]
        for check in ("correctness", "timing_neutral", "default_server_neutral", "fresh_prefill"):
            self.records[0]["checks"][check] = False
            with self.assertRaisesRegex(ReplayError, "failed or unrun"):
                self.validate()
            self.records[0]["checks"][check] = True
        self.records[0]["checks"]["twins"] = 1
        with self.assertRaisesRegex(ReplayError, "nonzero twins"):
            self.validate()

    def test_incorrect_target_context_ingest_or_proposer_accounting(self):
        for key, value in (("target_commits", 0), ("context_commits", 0), ("ingest_calls", 4),
                           ("proposer_calls", 1), ("lost", 1)):
            records = copy.deepcopy(self.records)
            records[0]["forwards"][0][key] = value
            with self.assertRaisesRegex(ReplayError, key):
                self.validate(records)

    def test_phase_intervals_not_restricted_to_nonoverlapping_sum(self):
        self.records[0]["forwards"][0]["phases_ns"]["target"] = 2_000_000
        self.validate()
        self.records[0]["forwards"][0]["phases_ns"]["target"] = float("inf")
        with self.assertRaisesRegex(ReplayError, "invalid number"):
            self.validate()

    def test_hashes_and_effective_configuration_are_bound(self):
        self.records[0]["schedule_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayError, "schedule_sha256 mismatch"):
            self.validate()
        self.records[0]["schedule_sha256"] = self.fixture["schedule_sha256"]
        self.change_config("speed", "fast")
        with self.assertRaisesRegex(ReplayError, "speed mismatch"):
            self.validate()
        self.change_config("speed", "standard")
        self.change_config("page_argument", 512 - 4)
        with self.assertRaisesRegex(ReplayError, "unequal logical"):
            self.validate()

    def test_geometry_and_physical_memory_controls(self):
        self.change_config("local_heads", 16)
        with self.assertRaisesRegex(ReplayError, "geometry mismatch"):
            self.validate()
        self.change_config("local_heads", 32)
        self.change_config("query_bytes_per_row", 32769)
        with self.assertRaisesRegex(ReplayError, "row geometry"):
            self.validate()
        self.change_config("query_bytes_per_row", 32768)
        self.records[2]["memory"]["row_page_bytes"] = self.records[0]["memory"]["row_page_bytes"]
        with self.assertRaisesRegex(ReplayError, "physical row-page byte"):
            self.validate()

    def test_unmeasured_physical_peak_is_null_not_a_zero_claim(self):
        for record in self.records:
            for key in ("tables_bytes", "scratch_bytes", "other_bytes", "peak_physical_bytes"):
                record["memory"][key] = None
        self.validate()
        self.records[0]["memory"]["peak_physical_bytes"] = 0
        with self.assertRaisesRegex(ReplayError, "footprint below"):
            self.validate()

    def test_native_build_must_be_identical_even_with_rehashed_metadata(self):
        self.records[2]["source"]["binary_sha256"] = "8" * 64
        self.records[3]["source"]["binary_sha256"] = "8" * 64
        with self.assertRaisesRegex(ReplayError, "native build changed"):
            self.validate()

    def test_start_identity_and_counterbalanced_orders(self):
        self.records[2]["start_id"] = self.records[0]["start_id"]
        self.records[3]["start_id"] = self.records[0]["start_id"]
        with self.assertRaisesRegex(ReplayError, "duplicate rank/block|inconsistent process-start"):
            self.validate()
        records = campaign(self.fixture, ratios=(1.1, 1.2), blocks=1)
        for record in records:
            record["start_order"] = int(record["arm"] == "B")
        with self.assertRaisesRegex(ReplayError, "not counterbalanced"):
            self.validate(records)

    def test_excluded_attempts_require_both_ranks_and_reason(self):
        for record in self.records[2:]:
            record.update(status="excluded", exclusion_reason="prescribed hardware alert")
            record["forwards"].pop()
            record["checks"]["correctness"] = False
        self.validate()
        self.records[3]["exclusion_reason"] = None
        with self.assertRaisesRegex(ReplayError, "exclusion_reason"):
            self.validate()

    def test_python_graph_keys_and_unexpected_eager_fallback(self):
        records = [receipt(self.fixture, "P", rank=rank) for rank in (0, 1)]
        self.validate(records)
        records[0]["forwards"][0]["eager_fallbacks"] = 1
        with self.assertRaisesRegex(ReplayError, "eager_fallbacks"):
            self.validate(records)
        records[0]["forwards"][0]["eager_fallbacks"] = 0
        records[0]["effective_config"]["graph_rows"] = [1, 4, 8, 32]
        records[0]["effective_config_sha256"] = digest(records[0]["effective_config"])
        with self.assertRaisesRegex(ReplayError, "missing expected Python graph"):
            self.validate(records)

    def test_strict_jsonl_rejects_duplicate_keys_blank_lines_and_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "results.jsonl"
            path.write_bytes(b"\n".join(canonical_bytes(r) for r in self.records) + b"\n")
            self.assertEqual(load_results([path]), self.records)
            for text in ('{"rank":0,"rank":1}\n', '\n', '{"wall_ns":NaN}\n'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_results([path])


if __name__ == "__main__":
    unittest.main()
