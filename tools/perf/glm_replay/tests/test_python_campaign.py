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

ROOT = Path(__file__).resolve().parents[1]


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.campaign = {"schema": "glm-replay-python-campaign-v1", "status": "ready", "arm": "P",
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
                            "telemetry": {"path": f"telemetry-{i}-{j}.json", "sha256": "3" * 64}}
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
        self.assertEqual(report["graph_rows"], [1, 4, 8, 16, 32])
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

    def test_duplicate_block_ids_across_cells_are_refused(self):
        self.campaign["cases"][1]["blocks"][0]["block_id"] = "case-0-block-0"
        self.save()
        with self.assertRaisesRegex(ReplayError, "duplicate/invalid block"):
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

    def test_validation_keeps_unknown_future_telemetry_hash_unrun(self):
        self.campaign["cases"][0]["blocks"][0]["telemetry"]["sha256"] = None
        self.save()
        _, _, report = self.preflight()
        self.assertEqual(report["hardware_checks"], "unrun")

    def test_measurement_needs_five_blocks_in_each_cell(self):
        self.campaign["cases"][0]["blocks"].pop()
        self.save()
        with self.assertRaisesRegex(ReplayError, "at least five"):
            self.preflight(mode="measure")

    def test_unrun_later_gate_cannot_start_measurement(self):
        with patch("tools.perf.glm_replay.python_campaign.artifact_receipt",
                   return_value={"bindings": {"runtime_sha256": "5" * 64}}):
            with patch("tools.perf.glm_replay.python_campaign.measurement_gates",
                       side_effect=ReplayError("unrun hardware gate")):
                with self.assertRaisesRegex(ReplayError, "unrun hardware"):
                    self.preflight(mode="measure")

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
        def telemetry(spec, root):
            i, j = [int(v) for v in Path(spec["path"]).stem.split("-")[1:]]
            return {"bindings": gate_binding(cases[i]["fixture"], cases[i]["metadata"], "5" * 64),
                    "completed_ranks": [0, 1], "block_id": f"case-{i}-block-{j}"}
        args = SimpleNamespace(output=str(self.root / "results.jsonl"), source_root="unused",
                               rank=0, master="reserved.invalid", port=12345, mode="measure", receipts_root=self.root)
        def records(runtime, record):
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
             patch("tools.perf.glm_replay.python_campaign.artifact_receipt", side_effect=telemetry), \
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
