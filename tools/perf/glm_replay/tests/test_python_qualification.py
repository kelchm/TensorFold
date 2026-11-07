"""Offline producers and real validators over host doubles with planted evidence failures."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError, digest, file_digest, read_json, write_json
from tools.perf.glm_replay.python_evidence import artifact_receipt
from tools.perf.glm_replay.python_external import (
    ACTIVATION_SCHEMA, LISTING_SCHEMA, common_binding, installed_listing,
)
from tools.perf.glm_replay.python_provenance import REQUIRED_ENV, SOURCE_MANIFEST, gate_binding
from tools.perf.glm_replay.python_qualification import build_qualification, main, validate_qualification
from tools.perf.glm_replay.tests.python_evidence_doubles import (
    RUNTIME, RUNTIME_RANKS, fixture_case, metadata, produced_validation, rehash,
)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures = [fixture_case(solo=True), fixture_case(partial=True, reverse=True, edge=True), fixture_case()]
        self.meta = metadata(self.fixtures[0])
        compiled = self.root / "HOST-DOUBLE.cubin"
        compiled.write_bytes(b"HOST TEST DATA, NOT A DEVICE KERNEL")
        member = {"path": compiled.name, "bytes": compiled.stat().st_size, "sha256": file_digest(compiled)}
        bundle = self.save("kernels.json", {"schema": "glm-replay-python-kernels-v1", "files": [member]})
        self.meta["artifacts"]["fatbin"] = {**bundle, "bytes": (self.root / "kernels.json").stat().st_size}
        self.meta["source"]["fatbin_sha256"] = bundle["sha256"]
        pins = read_json(SOURCE_MANIFEST)
        files = [{k: a[k] for k in ("path", "bytes", "sha256")} for a in pins["files"]]
        self.installed = [self.save(f"installed-{rank}.json", {"schema": LISTING_SCHEMA, "status": "listed",
            "rank": rank, "source": self.meta["source"], "source_tree_sha256": self.meta["source_tree_sha256"],
            "files": files}) for rank in (0, 1)]
        activation = {"schema": ACTIVATION_SCHEMA, "status": "observed", "execution": "device",
            "bindings": common_binding(gate_binding(self.fixtures[0], self.meta, RUNTIME)), "completed_ranks": [0, 1],
            "ranks": [{"rank": rank, "runtime_rank_sha256": RUNTIME_RANKS[rank], "source": self.meta["source"],
                "environment": REQUIRED_ENV,
                "device": {"uuid": f"HOST-DOUBLE-{rank}", "pci_bus_id": "HOST-DOUBLE", "compute_capability": [12, 1],
                           "driver_version": "HOST-DOUBLE"},
                "transport": {"kind": "nccl", "implementation": "tensorfold.cuda.comm.NCCL",
                    "bootstrap": "tensorfold.cuda.comm.NCCL", "world": 2, "peer_rank": 1 - rank,
                    "collective_counts": {"all_gather": 2, "barrier": 2}, "sent_bytes": 8, "received_bytes": 8,
                    "route_devices": ["HOST-DOUBLE"]},
                "kernels": {"bundle": self.meta["artifacts"]["fatbin"],
                            "loaded": [{"artifact": member, "symbols": ["HOST-DOUBLE"], "launch_count": 1}]}}
                for rank in (0, 1)]}
        self.activation = self.save("activation.json", activation)
        self.entries = [{"fixture": self.save(f"fixture-{i}.json", fixture),
                         "validation": self.save(f"validation-{i}.json", produced_validation(fixture, self.meta))}
                        for i, fixture in enumerate(self.fixtures)]

    def save(self, name, value):
        write_json(self.root / name, value)
        return {"path": name, "sha256": file_digest(self.root / name)}

    def build(self, fixture=None):
        return build_qualification(self.entries, self.activation, self.installed, fixture or self.fixtures[0],
                                   self.meta, RUNTIME, self.root, allow_synthetic=True)

    def test_suite_producer_derives_coverage_and_validator_reopens_actual_evidence(self):
        gate = self.build()
        self.assertTrue(all(gate["coverage"].values()))
        self.assertNotIn("owner_swap", gate["coverage"])
        validate_qualification(gate, self.fixtures[0], self.meta, RUNTIME, self.root, allow_synthetic=True)
        self.assertEqual(gate["bindings"], common_binding(gate_binding(self.fixtures[0], self.meta, RUNTIME)))
        self.assertFalse(produced_validation(self.fixtures[0], self.meta)["coverage"]["partial"])

    def test_missing_measured_fixture_or_suite_coverage_is_refused(self):
        with self.assertRaisesRegex(ReplayError, "measured fixture absent"):
            self.build(fixture_case(solo=True, partial=True))
        self.entries.pop(1)
        with self.assertRaisesRegex(ReplayError, "derived coverage"):
            self.build()

    def test_old_blanket_booleans_and_opaque_evidence_are_refused(self):
        old = {"schema": "glm-replay-python-external-gate-v1", "status": "pass", "execution": "device",
               "bindings": {}, "completed_ranks": [0, 1], "checks": {"hardware_qualification": True},
               "evidence": [self.save("opaque.json", {"true": True})]}
        with self.assertRaises(ReplayError):
            validate_qualification(old, self.fixtures[0], self.meta, RUNTIME, self.root)
        self.activation = old["evidence"][0]
        with self.assertRaisesRegex(ReplayError, "external activation"):
            self.build()

    def test_changed_binding_on_either_rank_is_refused_even_with_rehashed_artifacts(self):
        original = artifact_receipt(self.entries[1]["validation"], self.root)
        for rank in (0, 1):
            changed = copy.deepcopy(original)
            changed["ranks"][rank]["runtime_rank_sha256"] = "0" * 64
            self.entries[1]["validation"] = self.save("changed.json", rehash(changed))
            with self.assertRaisesRegex(ReplayError, "runtime rank identities"):
                self.build()

    def test_source_mismatch_on_either_installed_rank_and_seven_file_listing_are_refused(self):
        originals = copy.deepcopy(self.installed)
        for rank in (0, 1):
            for mutate in (
                lambda v: v["source"].update(image_digest="sha256:" + "0" * 64),
                lambda v: v["files"][438].update(sha256="0" * 64),
                lambda v: v.update(files=v["files"][:7]),
            ):
                self.installed = copy.deepcopy(originals)
                value = artifact_receipt(self.installed[rank], self.root)
                mutate(value)
                self.installed[rank] = self.save("bad-installed.json", value)
                with self.subTest(rank=rank), self.assertRaises(ReplayError):
                    self.build()

    def test_rehashed_activation_source_runtime_transport_and_kernel_failures_are_refused(self):
        original = artifact_receipt(self.activation, self.root)
        mutations = [
            lambda v: v["ranks"][1]["source"].update(binary_sha256="0" * 64),
            lambda v: v["ranks"][1].update(runtime_rank_sha256="0" * 64),
            lambda v: v["ranks"][1]["environment"].update(TF_GLM_MULTI_LONE="1"),
            lambda v: v["ranks"][1]["transport"].update(implementation="fake.NCCL"),
            lambda v: v["ranks"][1]["transport"].update(peer_rank=1),
            lambda v: v["ranks"][1]["transport"]["collective_counts"].update(all_gather=0),
            lambda v: v["ranks"][1]["kernels"]["loaded"][0].update(launch_count=0),
            lambda v: v["ranks"][1]["kernels"].update(loaded=[]),
            lambda v: v["ranks"][1]["kernels"]["loaded"][0]["artifact"].update(sha256="0" * 64),
        ]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            self.activation = self.save("changed-activation.json", value)
            with self.subTest(mutate=mutate), self.assertRaises(ReplayError):
                self.build()

    def test_artifact_tampering_cannot_keep_old_sha(self):
        (self.root / self.entries[0]["validation"]["path"]).write_text("{}\n")
        with self.assertRaisesRegex(ReplayError, "artifact identity"):
            self.build()

    def test_synthetic_host_suite_is_refused_by_default_device_validator(self):
        gate = self.build()
        with self.assertRaisesRegex(ReplayError, "synthetic fixture refused"):
            validate_qualification(gate, self.fixtures[0], self.meta, RUNTIME, self.root)


class InstalledProducerTests(unittest.TestCase):
    def test_driver_free_listing_command_hashes_every_extracted_file_and_refuses_source_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            extracted = root / "installed"
            extracted.mkdir()
            pins = read_json(SOURCE_MANIFEST)
            pins["files"] = []
            for i in range(439):
                name, raw = f"module{i}.py", f"HOST_DOUBLE_{i} = {i}\n".encode()
                (extracted / name).write_bytes(raw)
                pins["files"].append({"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                      "qualification": "local-source-only"})
            pin_path = root / "pins.json"
            write_json(pin_path, pins)
            meta = metadata(fixture_case())
            meta["source_tree_sha256"] = digest(pins)
            meta_path, output = root / "metadata.json", root / "listing.json"
            write_json(meta_path, meta)
            with patch("tools.perf.glm_replay.python_external.SOURCE_MANIFEST", pin_path), \
                 patch("tools.perf.glm_replay.python_provenance.SOURCE_MANIFEST", pin_path):
                rc = main(["installed-list", "--installed-root", str(extracted), "--metadata", str(meta_path),
                    "--rank", "1", "--image-digest", meta["source"]["image_digest"], "--device-image-digest",
                    meta["source"]["device_image_digest"], "--output", str(output)])
                self.assertEqual(rc, 0)
                self.assertEqual(len(read_json(output)["files"]), 439)
                (extracted / "module438.py").write_text("changed\n")
                with self.assertRaisesRegex(ReplayError, "identity mismatch"):
                    installed_listing(extracted, meta, 1, meta["source"]["image_digest"],
                                      meta["source"]["device_image_digest"])


if __name__ == "__main__":
    unittest.main()
