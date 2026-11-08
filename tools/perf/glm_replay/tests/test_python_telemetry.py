"""Actual offline telemetry parser/request tests with planted window, hash and sample failures."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError, canonical_bytes, digest, file_digest, write_json
from tools.perf.glm_replay.python_provenance import gate_binding, metadata_check
from tools.perf.glm_replay.telemetry import (
    TELEMETRY_SCHEMA, block_window, reference, validate_telemetry, wait_telemetry, write_request,
)
from tools.perf.glm_replay.tests.python_evidence_doubles import RUNTIME, fixture_case, metadata


def window_and_samples():
    fixture = fixture_case()
    meta = metadata(fixture)
    window = block_window(gate_binding(fixture, meta, RUNTIME), meta["ids"],
        [{"rank": r, "realtime_start_ns": 1000 + r, "realtime_end_ns": 2000 + r} for r in (0, 1)],
        "c" * 64, "d" * 64, {"scheme": "file-manifest-v1", "target": ["e" * 64] * 2, "drafter": ["f" * 64] * 2})
    telemetry = {k: copy.deepcopy(window[k]) for k in
                 ("bindings", "ids", "completed_ranks", "session_nonce", "campaign_sha256", "checkpoints")}
    telemetry.update(schema=TELEMETRY_SCHEMA, status="measured", window_sha256=digest(window), ranks=[])
    for rank in window["ranks"]:
        telemetry["ranks"].append({**rank, "device_uuid": f"HOST-DOUBLE-{rank['rank']}", "max_sample_gap_ns": 1000,
            "samples": [{"realtime_ns": t, "sm_clock_mhz": 1200, "memory_clock_mhz": 1500,
                         "temperature_millicelsius": 60000, "power_milliwatts": 20000, "used_memory_bytes": 8192}
                        for t in (500, 1500, 2500)]})
    return window, telemetry


class TelemetryTests(unittest.TestCase):
    def test_window_is_complete_before_atomic_publication_and_temporary_is_removed(self):
        window, _ = window_and_samples()
        with tempfile.TemporaryDirectory() as directory:
            spec = {"path": "blocks/future.json"}
            expected = canonical_bytes(window) + b"\n"
            link = os.link
            def publish(source, destination, **kwargs):
                self.assertFalse(destination.exists())
                self.assertEqual(source.read_bytes(), expected)
                return link(source, destination, **kwargs)
            with patch("tools.perf.glm_replay.telemetry.os.link", side_effect=publish):
                request = write_request(spec, directory, window)
            self.assertEqual(request.read_bytes(), expected)
            self.assertEqual(list(request.parent.iterdir()), [request])

    def test_atomic_window_publication_preserves_a_racing_writer_and_cleans_temporary(self):
        window, _ = window_and_samples()
        with tempfile.TemporaryDirectory() as directory:
            spec = {"path": "blocks/future.json"}
            link = os.link
            def publish(source, destination, **kwargs):
                destination.write_bytes(b"preserve concurrent writer")
                return link(source, destination, **kwargs)
            with patch("tools.perf.glm_replay.telemetry.os.link", side_effect=publish):
                with self.assertRaises(FileExistsError):
                    write_request(spec, directory, window)
            request = Path(directory) / "blocks/future.json.window.json"
            self.assertEqual(request.read_bytes(), b"preserve concurrent writer")
            self.assertEqual(list(request.parent.iterdir()), [request])

    def test_post_block_request_is_exclusive_and_actual_receipt_hash_is_computed_later(self):
        window, telemetry = window_and_samples()
        with tempfile.TemporaryDirectory() as directory:
            spec = {"path": "blocks/future.json"}
            request = write_request(spec, directory, window)
            self.assertEqual(request.name, "future.json.window.json")
            self.assertFalse((Path(directory) / spec["path"]).exists())
            with self.assertRaises(FileExistsError):
                write_request(spec, directory, window)
            write_json(Path(directory) / spec["path"], telemetry)
            value, sha = wait_telemetry(spec, directory, window, timeout_s=0)
            self.assertEqual(value, telemetry)
            self.assertEqual(sha, file_digest(Path(directory) / spec["path"]))
            with self.assertRaisesRegex(ReplayError, "hash mismatch"):
                wait_telemetry(spec, directory, window, timeout_s=0, expected_sha256="0" * 64)

    def test_missing_telemetry_stays_unrun_at_bounded_deadline(self):
        window, _ = window_and_samples()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ReplayError, "bounded deadline"):
                wait_telemetry({"path": "missing.json"}, directory, window, timeout_s=0)

    def test_stub_and_old_prehashed_path_are_refused(self):
        window, _ = window_and_samples()
        with self.assertRaises(ReplayError):
            validate_telemetry({"bindings": window["bindings"], "completed_ranks": [0, 1], "block_id": "old"}, window)
        with self.assertRaisesRegex(ReplayError, "future telemetry reference"):
            reference({"path": "old.json", "sha256": "a" * 64}, Path.cwd())

    def test_both_rank_windows_and_samples_are_required(self):
        mutations = [
            lambda t: t["ranks"].pop(),
            lambda t: t["ranks"][1].update(realtime_start_ns=1000),
            lambda t: t["ranks"][1].update(samples=[]),
            lambda t: t["ranks"][0]["samples"].pop(0),
            lambda t: t["ranks"][1].update(max_sample_gap_ns=100),
            lambda t: t["ranks"][1]["samples"].reverse(),
            lambda t: t["ranks"][1]["samples"].pop(1),
            lambda t: t["ranks"][1]["samples"][0].update(sm_clock_mhz=True),
            lambda t: t["ranks"][0]["samples"][0].update(power_milliwatts=0),
        ]
        for mutate in mutations:
            window, telemetry = window_and_samples()
            mutate(telemetry)
            with self.subTest(mutate=mutate), self.assertRaises(ReplayError):
                validate_telemetry(telemetry, window)

    def test_window_binding_ids_nonce_campaign_and_checkpoints_are_authenticated(self):
        for key in ("bindings", "ids", "session_nonce", "campaign_sha256", "checkpoints", "window_sha256"):
            window, telemetry = window_and_samples()
            telemetry[key] = "0" * 64
            with self.subTest(key=key), self.assertRaises(ReplayError):
                validate_telemetry(telemetry, window)

    def test_shared_parser_uses_common_native_checkpoint_scheme_for_a_and_b(self):
        for arm in ("A", "B"):
            window, telemetry = window_and_samples()
            window["bindings"]["arm"] = arm
            window["checkpoints"]["scheme"] = "tensorfold-full-identity-v1"
            telemetry["bindings"] = copy.deepcopy(window["bindings"])
            telemetry["checkpoints"] = copy.deepcopy(window["checkpoints"])
            telemetry["window_sha256"] = digest(window)
            validate_telemetry(telemetry, window)

    def test_unsafe_and_symlink_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "alias").symlink_to(root, target_is_directory=True)
            for path in ("../escape", "/absolute", "a\\b", ".", "a//b", "alias/data.json"):
                with self.subTest(path=path), self.assertRaises(ReplayError):
                    reference({"path": path}, root)

    def test_metadata_path_validation_does_not_resolve_receipts_against_cwd(self):
        fixture = fixture_case()
        meta = metadata(fixture)
        meta["telemetry"] = {"path": "future/receipt.json"}
        metadata_check(meta, fixture)
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(reference(meta["telemetry"], directory), Path(directory).resolve() / "future/receipt.json")


if __name__ == "__main__":
    unittest.main()
