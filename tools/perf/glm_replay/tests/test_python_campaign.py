"""Host campaign negatives and one-session case-loop doubles; no GPU qualification."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError, file_digest, read_json, write_json
from tools.perf.glm_replay.fixture import load_fixture, seal_fixture
from tools.perf.glm_replay.python_campaign import preflight_campaign, run_campaign
from tools.perf.glm_replay.python_provenance import gate_binding
from tools.perf.glm_replay.tests.python_evidence_doubles import CaptureRuntime, produced_validation

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = {"schema": "glm-replay-python-campaign-v2", "status": "ready", "arm": "P",
                         "run_id": "host-double", "pair_id": "host-pair", "start_id": "one-real-session",
                         "start_order": 0, "cases": []}
        self.fixtures, self.metadata = [], []
        for i in range(2):
            fixture = load_fixture(ROOT / "fixtures" / "host-prose.json", allow_synthetic=True)
            fixture["fixture_id"] = f"host-case-{i}"
            fixture = seal_fixture(fixture)
            meta = read_json(ROOT / "python_metadata.example.json")
            meta["status"] = "ready"
            meta["bindings"] = {k: fixture[k] for k in meta["bindings"]}
            for key in meta["source"]:
                if meta["source"][key] is None:
                    meta["source"][key] = "sha256:" + "1" * 64 if key == "image_digest" else "1" * (
                        40 if key == "harness_commit" else 64)
            meta["launch"]["environment_sha256"] = ["2" * 64] * 2
            meta["ids"] = {k: self.campaign[k] for k in ("run_id", "pair_id", "start_id", "start_order")}
            meta["ids"].update({"cell_id": f"cell-{i}", "block_id": "pilot-unused"})
            self.fixtures.append(fixture)
            self.metadata.append(meta)
            self.campaign["cases"].append({"fixture": {}, "metadata": {},
                "blocks": [{"block_id": f"case-{i}-block-{j}",
                            "telemetry": {"path": f"telemetry-{i}-{j}.json"}}
                           for j in range(5)]})
        self.path = self.root / "campaign.json"
        self.save()

    def save(self):
        for i, (fixture, meta) in enumerate(zip(self.fixtures, self.metadata)):
            for role, value in (("fixture", fixture), ("metadata", meta)):
                path = self.root / f"{role}-{i}.json"
                write_json(path, value)
                self.campaign["cases"][i][role] = {"path": path.name, "sha256": file_digest(path)}
        write_json(self.path, self.campaign)

    def preflight(self, mode="validate"):
        report = {"bindings": self.metadata[0]["bindings"], "source_tree_sha256": "4" * 64,
                  "status": "host-checked", "hardware_checks": "unrun"}
        with patch("tools.perf.glm_replay.python_campaign.load_fixture",
                   side_effect=lambda p: load_fixture(p, allow_synthetic=True)):
            with patch("tools.perf.glm_replay.python_campaign.preflight", return_value=(None, None, report)):
                return preflight_campaign(self.path, self.root, self.root, {}, 0, self.root, mode)

    def test_all_cases_and_blocks_prevalidated_with_one_graph_bank(self):
        _, cases, report = self.preflight()
        self.assertEqual((report["cases"], report["blocks"]), (2, 10))
        self.assertEqual(report["graph_rows"], list(range(1, 33)))
        self.assertEqual(len(cases), 2)
        self.assertEqual(report["hardware_checks"], "unrun")

    def test_bad_later_fixture_hash_refused_before_session(self):
        self.campaign["cases"][1]["fixture"]["sha256"] = "9" * 64
        write_json(self.path, self.campaign)
        with self.assertRaisesRegex(ReplayError, "input identity"):
            self.preflight()

    def test_changed_capacity_in_later_case_is_refused(self):
        self.fixtures[1]["config"]["pool"]["h"] += 4
        self.fixtures[1] = seal_fixture(self.fixtures[1])
        self.metadata[1]["bindings"] = {k: self.fixtures[1][k] for k in self.metadata[1]["bindings"]}
        self.save()
        with self.assertRaisesRegex(ReplayError, "changed pool/capacity"):
            self.preflight()

    def test_changed_source_or_runtime_launch_is_refused(self):
        self.metadata[1]["launch"]["prefill_rows"] = 4096
        self.save()
        with self.assertRaisesRegex(ReplayError, "runtime launch"):
            self.preflight()

    def test_fabricated_second_start_id_is_refused(self):
        self.metadata[1]["ids"]["start_id"] = "pretend-new-process"
        self.save()
        with self.assertRaisesRegex(ReplayError, "genuine process-start"):
            self.preflight()

    def test_duplicate_block_ids_within_a_cell_are_refused(self):
        self.campaign["cases"][0]["blocks"][1]["block_id"] = "case-0-block-0"
        self.save()
        with self.assertRaisesRegex(ReplayError, "duplicate/invalid cell/block"):
            self.preflight()

    def test_integer_block_id_is_not_silently_coerced(self):
        self.campaign["cases"][0]["blocks"][0]["block_id"] = 123
        self.save()
        with self.assertRaisesRegex(ReplayError, "block ID"):
            self.preflight()

    def test_unsafe_future_telemetry_path_is_refused_before_session(self):
        self.campaign["cases"][1]["blocks"][4]["telemetry"]["path"] = "../sealed.json"
        self.save()
        with self.assertRaisesRegex(ReplayError, "unsafe telemetry path"):
            self.preflight()

    def test_future_telemetry_reference_refuses_prehashed_stubs(self):
        self.campaign["cases"][0]["blocks"][0]["telemetry"]["sha256"] = "3" * 64
        self.save()
        with self.assertRaisesRegex(ReplayError, "future telemetry reference"):
            self.preflight()

    def test_measurement_needs_five_blocks_in_each_cell(self):
        self.campaign["cases"][0]["blocks"].pop()
        self.save()
        with patch("tools.perf.glm_replay.python_campaign.artifact_receipt",
                   return_value={"bindings": {"runtime_sha256": "5" * 64}}), \
             patch("tools.perf.glm_replay.python_campaign.measurement_gates", return_value={}):
            with self.assertRaisesRegex(ReplayError, "at least five"):
                self.preflight(mode="measure")

    def test_unrun_later_gate_cannot_start_measurement(self):
        with patch("tools.perf.glm_replay.python_campaign.artifact_receipt",
                   return_value={"bindings": {"runtime_sha256": "5" * 64}}):
            with patch("tools.perf.glm_replay.python_campaign.measurement_gates",
                       side_effect=ReplayError("unrun hardware gate")):
                with self.assertRaisesRegex(ReplayError, "unrun hardware"):
                    self.preflight(mode="measure")

    def interleave(self):
        first, second = self.campaign["cases"]
        self.campaign["cases"] = [{**first, "blocks": first["blocks"][:2]}, second,
                                  {**first, "blocks": first["blocks"][2:]}]
        write_json(self.path, self.campaign)

    def test_cell_may_repeat_in_order_with_total_replication_across_entries(self):
        self.interleave()
        with patch("tools.perf.glm_replay.python_campaign.artifact_receipt",
                   return_value={"bindings": {"runtime_sha256": "5" * 64}}), \
             patch("tools.perf.glm_replay.python_campaign.measurement_gates", return_value={}):
            _, cases, report = self.preflight("measure")
        self.assertEqual([c["metadata"]["ids"]["cell_id"] for c in cases], ["cell-0", "cell-1", "cell-0"])
        self.assertEqual(report["blocks"], 10)

    def test_cell_fixture_mapping_cannot_change_in_later_entry(self):
        self.metadata[1]["ids"]["cell_id"] = "cell-0"
        self.save()
        with self.assertRaisesRegex(ReplayError, "cell changed fixture"):
            self.preflight()

    def test_block_id_can_repeat_in_distinct_cells(self):
        self.campaign["cases"][1]["blocks"][0]["block_id"] = "case-0-block-0"
        self.save()
        _, _, report = self.preflight()
        self.assertEqual(report["blocks"], 10)

    def test_reused_telemetry_paths_are_refused(self):
        self.campaign["cases"][1]["blocks"][0]["telemetry"] = self.campaign["cases"][0]["blocks"][0]["telemetry"]
        self.save()
        with self.assertRaisesRegex(ReplayError, "reused telemetry"):
            self.preflight()

    def validation_loop(self, fail_second=False):
        campaign, cases, report = self.preflight()
        created, calls = [], []
        class FakeRuntime(CaptureRuntime):
            def __init__(self, fixture, meta, roots, rank, master, port, report, **kwargs):
                super().__init__(fixture, meta, rank)
                created.append(kwargs)
            def select_case(self, fixture, meta):
                self.fixture, self.meta = fixture, meta
        def validate(runtime, binding):
            calls.append(runtime.fixture["fixture_sha256"])
            if fail_second and len(calls) == 2:
                raise ReplayError("planted later device failure")
            return produced_validation(runtime.fixture, runtime.meta)
        args = SimpleNamespace(output=str(self.root / "validations"), source_root="unused", rank=0,
            master="reserved.invalid", port=12345, mode="validate", receipts_root=self.root)
        with patch("tools.perf.glm_replay.python_runtime.Runtime", FakeRuntime), \
             patch("tools.perf.glm_replay.python_runtime.exchange"), \
             patch("tools.perf.glm_replay.python_provenance.install_import_boundary"), \
             patch("tools.perf.glm_replay.python_validation.validate_device", side_effect=validate):
            if fail_second:
                with self.assertRaisesRegex(ReplayError, "planted later"):
                    run_campaign(args, campaign, cases, report, {})
            else:
                run_campaign(args, campaign, cases, report, {})
        return created, calls

    def test_later_failure_preserves_prior_exclusively_published_validation(self):
        _, calls = self.validation_loop(fail_second=True)
        paths = list((self.root / "validations").glob("*.json"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(paths), 1)
        self.assertEqual(read_json(paths[0])["bindings"]["fixture_sha256"], calls[0])

    def test_interleaved_duplicate_validation_reuses_authenticated_successful_receipt(self):
        self.interleave()
        created, calls = self.validation_loop()
        self.assertEqual(len(created), 1)
        self.assertEqual(len(created[0]["fixtures"]), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(list((self.root / "validations").glob("*.json"))), 2)

    def test_case_loop_uses_one_engine_and_exact_block_strings_in_one_start(self):
        campaign, cases, report = self.preflight()
        report["expected_runtime_sha256"] = "5" * 64
        created, runs = [], []
        class FakeRuntime:
            def __init__(self, fixture, meta, roots, rank, master, port, preflight, **kwargs):
                self.fixture, self.meta, self.rank = fixture, meta, rank
                self.runtime_sha256 = "5" * 64
                self.nccl = SimpleNamespace(store=None)
                created.append(kwargs)
            def select_case(self, fixture, meta):
                self.fixture, self.meta = fixture, meta
            def run(self, enabled):
                runs.append(copy.deepcopy(self.meta["ids"]))
                return {}
        args = SimpleNamespace(output=str(self.root / "results.jsonl"), source_root="unused",
                               rank=0, master="reserved.invalid", port=12345, mode="measure", receipts_root=self.root)
        def records(runtime, record, anchor):
            return [{**runtime.meta["ids"], "rank": r} for r in (0, 1)]
        def measured(runtime, checks, root):
            runtime.run(enabled=True)
            return runtime.run(enabled=False), {}
        with patch("tools.perf.glm_replay.python_runtime.Runtime", FakeRuntime), \
             patch("tools.perf.glm_replay.python_runtime.exchange"), \
             patch("tools.perf.glm_replay.python_provenance.install_import_boundary"), \
             patch("tools.perf.glm_replay.results.CONFIG_FIELDS", "runtime_sha256"), \
             patch("tools.perf.glm_replay.results.RECORD_FIELDS", "timing_source"), \
             patch("tools.perf.glm_replay.python_campaign.measurement_gates", return_value={}), \
             patch("tools.perf.glm_replay.python_receipts.block_telemetry", return_value="3" * 64), \
             patch("tools.perf.glm_replay.python_receipts.build_record", return_value={}), \
             patch("tools.perf.glm_replay.python_receipts.measure_block", side_effect=measured), \
             patch("tools.perf.glm_replay.python_receipts.retain_anchor"), \
             patch("tools.perf.glm_replay.python_receipts.collect_records", side_effect=records):
            run_campaign(args, campaign, cases, report, {})
        self.assertEqual(len(created), 1)
        self.assertEqual(len(created[0]["fixtures"]), 2)
        self.assertEqual(len(runs), 20)
        self.assertEqual({r["start_id"] for r in runs}, {"one-real-session"})
        self.assertEqual([r["block_id"] for r in runs[1::2]],
                         [f"case-{i}-block-{j}" for i in range(2) for j in range(5)])
        self.assertEqual(len(Path(args.output).read_text().splitlines()), 20)


if __name__ == "__main__":
    unittest.main()
