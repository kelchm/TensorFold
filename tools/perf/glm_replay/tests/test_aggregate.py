"""Known paired effects and exclusion controls prevent block/stream pseudo replication."""
import math
import unittest

from tools.perf.glm_replay import ReplayError, summarize_paired
from tools.perf.glm_replay.generate import build_fixture
from tools.perf.glm_replay.tests.support import campaign, receipt


class AggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = build_fixture(repetitions=2)
        cls.fixtures = {cls.fixture["fixture_sha256"]: cls.fixture}

    def summarize(self, records):
        return summarize_paired(records, self.fixtures, allow_synthetic=True)

    def test_known_effects_have_three_units_not_fifteen_blocks_or_sixty_streams(self):
        report = self.summarize(campaign(self.fixture))
        cell = report["cells"][0]
        self.assertEqual(cell["B_over_A"]["independent_pairs"], 3)
        self.assertAlmostEqual(cell["B_over_A"]["estimate"], (1.1 * 1.2 * 1.3) ** (1 / 3), places=6)
        self.assertEqual(len(cell["paired_starts"]), 3)
        self.assertEqual(cell["paired_starts"][0]["A"]["block_count"], 5)
        self.assertEqual(cell["unique_forwards_per_block"], 3)
        self.assertEqual(cell["committed_tokens_per_block"], 48)
        self.assertIn("replay throughput", report["rate_label"])
        self.assertTrue(report["synthetic_host_only"])

    def test_blocks_are_summarized_within_starts_before_pair_ratio(self):
        records = campaign(self.fixture, ratios=(1.0, 1.0, 1.0))
        for record in records:
            i = int(record["block_id"].split("-")[-1])
            record["wall_ns"] = ([1, 10, 100, 1000, 10000] if record["arm"] == "A"
                                 else [3, 20, 100, 500, 1000])[i] * 1000
        cell = self.summarize(records)["cells"][0]
        ratio = sum([3, 20, 100, 500, 1000]) / sum([1, 10, 100, 1000, 10000])
        self.assertAlmostEqual(cell["B_over_A"]["estimate"], ratio)
        self.assertNotAlmostEqual(cell["B_over_A"]["estimate"], (3 + 2 + 1 + .5 + .1) / 5)

    def test_follower_wall_time_is_included_without_adding_overlapping_ranks(self):
        records = campaign(self.fixture, ratios=(1.0, 1.0, 1.0))
        for record in records:
            record["wall_ns"] = 1_000_000 if record["rank"] == 0 else 2_000_000
        start = self.summarize(records)["cells"][0]["paired_starts"][0]["A"]
        self.assertEqual(start["mean_wall_ns"], 2_000_000)
        self.assertEqual(start["forced_committed_tokens_per_second"], 24_000)

    def test_many_blocks_in_one_start_cannot_replace_independent_pairs(self):
        with self.assertRaisesRegex(ReplayError, "independent native paired starts"):
            self.summarize(campaign(self.fixture, ratios=(1.2,), blocks=30))
        with self.assertRaisesRegex(ReplayError, "matched blocks"):
            self.summarize(campaign(self.fixture, blocks=4))

    def test_retains_exclusions_and_removes_matched_block_in_both_arms(self):
        records = campaign(self.fixture, blocks=6)
        for record in records:
            if record["pair_id"] == "pair-1" and record["block_id"] == "block-2" and record["arm"] == "B":
                record.update(status="excluded", exclusion_reason="predeclared fabric health alert")
                record["forwards"].pop()
        report = self.summarize(records)
        self.assertEqual(len(report["excluded_attempts"]), 1)
        starts = report["cells"][0]["paired_starts"]
        self.assertEqual(starts[0]["A"]["block_count"], 6)
        self.assertEqual(starts[1]["A"]["block_count"], 5)
        self.assertEqual(starts[1]["B"]["block_count"], 5)
        self.assertEqual(len(report["excluded_attempts"][0]["attempts"]), 4)

    def test_uncertain_interval_is_explicitly_inconclusive(self):
        cell = self.summarize(campaign(self.fixture, ratios=(0.9, 1.0, 1.1)))["cells"][0]
        self.assertLess(cell["B_over_A"]["ci95"][0], 1)
        self.assertGreater(cell["B_over_A"]["ci95"][1], 1)
        self.assertEqual(cell["interpretation"], "inconclusive")
        self.assertTrue(math.isfinite(cell["B_minus_A_ns"]["estimate"]))

    def test_python_context_is_separate_and_requires_two_starts(self):
        records = campaign(self.fixture)
        records += [receipt(self.fixture, "P", pair, block, rank) for pair in range(2)
                    for block in range(5) for rank in (0, 1)]
        report = self.summarize(records)
        self.assertEqual(len(report["python_context"]), 2)
        self.assertEqual(report["cells"][0]["B_over_A"]["independent_pairs"], 3)
        records = [r for r in records if r["arm"] != "P" or r["start_id"] == "P-start-0"]
        with self.assertRaisesRegex(ReplayError, "at least two independent starts"):
            self.summarize(records)


if __name__ == "__main__":
    unittest.main()
