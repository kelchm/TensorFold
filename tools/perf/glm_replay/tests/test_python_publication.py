"""Exercise actual Python anchor/result producers against the integrated common authenticator."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError, canonical_bytes, digest, file_digest, write_json
from tools.perf.glm_replay.python_control import Frame
from tools.perf.glm_replay.python_receipts import build_record, measure_block, retain_anchor
from tools.perf.glm_replay.results import validate_results
from tools.perf.glm_replay.tests.python_evidence_doubles import (
    RUNTIME, fixture_case, metadata, produced_validation,
)
from tools.perf.glm_replay.tests.support import receipt


class PublicationTests(unittest.TestCase):
    def produced_records(self, root):
        fixture = fixture_case(partial=True)
        meta = metadata(fixture)
        validation = produced_validation(fixture, meta)
        path = root / "validation.json"
        write_json(path, validation)
        meta["validation_gate"] = {"path": path.name, "sha256": file_digest(path)}
        base = receipt(fixture, arm="P")
        base["effective_config"]["graph_rows"] = list(range(1, 33))
        base["effective_config"]["runtime_sha256"] = RUNTIME
        records, anchors = [], []
        for rank in (0, 1):
            evidence = validation["ranks"][rank]["evidence"]
            frames = []
            for forward in fixture["forwards"]:
                count = sum(p["kept"] for p in forward["parts"])
                frames.append(Frame(forward, [], [], [], graph_replays=1, target_commits=count,
                                    context_commits=count, ingest_calls=1))
            phases = [{k: 100 for k in base["forwards"][0]["phases_ns"]} for _ in frames]
            def run(enabled, final_capture):
                self.assertTrue(final_capture)
                return {"kept_outputs": evidence["expected_kept_outputs"], "final": evidence["expected_final"],
                    "frames": frames, "completed_ranks": [0, 1], "phases": phases if enabled else None,
                    "preparation_ns": 500, "wall_ns": 1000}
            runtime = SimpleNamespace(fixture=fixture, meta=meta, rank=rank, runtime_sha256=RUNTIME,
                session_nonce="c" * 64, campaign_sha256=digest(meta), checkpoints={"scheme": "file-manifest-v1",
                **{role: [digest(files) for files in meta["checkpoints"][role]["ranks"]]
                   for role in ("target", "drafter")}}, run=run, nccl=SimpleNamespace(store=None))
            def exchange(store, label, local_rank, local):
                return [{**local, "rank": r} for r in (0, 1)]
            with patch("tools.perf.glm_replay.python_receipts.exchange", side_effect=exchange):
                headline, anchor = measure_block(runtime, base["checks"], root)
            with patch("tools.perf.glm_replay.python_receipts.effective_config", return_value=base["effective_config"]), \
                 patch("tools.perf.glm_replay.python_receipts.memory_receipt", return_value=base["memory"]):
                record = build_record(runtime, headline, base["checks"], "e" * 64, anchor)
            records.append(record)
            anchors.append(anchor)
        self.assertEqual(anchors[0], anchors[1])
        return fixture, records, anchors[0]

    def test_produced_anchor_authenticates_before_and_after_exclusive_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, records, anchor = self.produced_records(root)
            sha = hashlib.sha256(canonical_bytes(anchor) + b"\n").hexdigest()
            fixtures = {fixture["fixture_sha256"]: fixture}
            validate_results(records, fixtures, allow_synthetic=True, anchors={sha: canonical_bytes(anchor) + b"\n"})
            output = root / "HOST-DOUBLE.jsonl"
            self.assertEqual(retain_anchor(output, anchor), sha)
            validate_results(records, fixtures, allow_synthetic=True, anchors_root=str(output) + ".anchors")
            self.assertEqual(records[0]["session_nonce"], anchor["session_nonce"])
            self.assertEqual(records[0]["checkpoints"], anchor["checkpoints"])

    def test_changed_phase_or_process_identity_cannot_reuse_prior_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, records, anchor = self.produced_records(root)
            sha = records[0]["timing_source"]["anchor_sha256"]
            for mutate in (
                lambda r: r[0]["forwards"][0]["phases_ns"].update(target=101),
                lambda r: r[0].update(session_nonce="0" * 64),
                lambda r: r[0]["checkpoints"]["target"].__setitem__(0, "0" * 64),
            ):
                changed = copy.deepcopy(records)
                mutate(changed)
                with self.assertRaises(ReplayError):
                    validate_results(changed, {fixture["fixture_sha256"]: fixture}, allow_synthetic=True,
                                     anchors={sha: canonical_bytes(anchor) + b"\n"})


if __name__ == "__main__":
    unittest.main()
