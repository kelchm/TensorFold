"""Host identity refusals must happen without importing torch or any frozen engine module."""
import copy
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.perf.glm_replay.common import ReplayError, digest, write_json
from tools.perf.glm_replay.fixture import PYTHON_BASE, load_fixture, seal_fixture
from tools.perf.glm_replay.python_provenance import (
    DEVICE, KIT, REQUIRED_ENV, FrozenImports, FrozenLoader, environment, graph_rows,
    install_import_boundary, preflight, require_gate, verify_files,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "host-prose.json"


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = self.root / "tensorfold"
        self.engine.mkdir()
        self.code = b"raise AssertionError('GPU source must not execute during host preflight')\n"
        (self.engine / "__init__.py").write_bytes(self.code)
        self.manifest = {"schema": "glm-replay-python-sources-v1", "engine_commit": PYTHON_BASE,
                         "kit_commit": KIT, "device_image": DEVICE,
                         "files": [{"path": "__init__.py", "bytes": len(self.code),
                                    "sha256": hashlib.sha256(self.code).hexdigest(),
                                    "qualification": "local-source-only"}]}
        self.pin = self.root / "pins.json"
        write_json(self.pin, self.manifest)
        self.patch = patch("tools.perf.glm_replay.python_provenance.SOURCE_MANIFEST", self.pin)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_host_preflight_never_executes_frozen_source(self):
        before = set(sys.modules)
        _, _, report = preflight(FIXTURE, self.engine, allow_synthetic=True)
        self.assertEqual(report["hardware_checks"], "unrun")
        self.assertEqual(report["source_tree_sha256"], digest(self.manifest))
        self.assertFalse(any(k == "torch" or k.startswith("torch.") for k in set(sys.modules) - before))

    def test_source_mismatch_before_any_gpu_import(self):
        (self.engine / "__init__.py").write_bytes(b"import torch\n" + self.code)
        class RefuseGpu:
            def find_spec(self, fullname, *args):
                if fullname == "torch" or fullname.startswith(("torch.", "tensorfold.")):
                    raise AssertionError("GPU import attempted before pin refusal")
        with patch.object(sys, "meta_path", [RefuseGpu(), *sys.meta_path]):
            with self.assertRaisesRegex(ReplayError, "identity mismatch"):
                preflight(FIXTURE, self.engine, allow_synthetic=True)

    def test_bad_fixture_fails_before_source_read(self):
        fixture = load_fixture(FIXTURE, allow_synthetic=True)
        fixture["forwards"][0]["parts"][0]["kept"] = 0
        invalid = self.root / "invalid.json"
        write_json(invalid, seal_fixture(fixture))
        with patch("tools.perf.glm_replay.python_provenance.verify_sources") as verify:
            with self.assertRaises(ReplayError):
                preflight(invalid, self.engine, allow_synthetic=True)
            verify.assert_not_called()

    def test_extra_unpinned_module_is_refused(self):
        (self.engine / "poison.py").write_text("import torch\n")
        with self.assertRaisesRegex(ReplayError, "unpinned"):
            preflight(FIXTURE, self.engine, allow_synthetic=True)

    def test_missing_source_is_refused(self):
        (self.engine / "__init__.py").unlink()
        with self.assertRaisesRegex(ReplayError, "missing file"):
            preflight(FIXTURE, self.engine, allow_synthetic=True)

    def test_symlink_artifact_is_refused(self):
        (self.engine / "link.py").symlink_to(self.engine / "__init__.py")
        item = {**self.manifest["files"][0], "path": "link.py"}
        with self.assertRaisesRegex(ReplayError, "symlink"):
            verify_files(self.engine, [item])

    def test_path_escape_is_refused(self):
        item = {**self.manifest["files"][0], "path": "../pins.json"}
        with self.assertRaisesRegex(ReplayError, "unsafe artifact"):
            verify_files(self.engine, [item])

    def test_symlink_directory_cannot_add_unpinned_import_namespace(self):
        (self.engine / "alias").symlink_to(self.engine, target_is_directory=True)
        with self.assertRaisesRegex(ReplayError, "symlink"):
            preflight(FIXTURE, self.engine, allow_synthetic=True)

    def test_existing_gpu_import_is_refused(self):
        with patch.dict(sys.modules, {"torch": object()}):
            with self.assertRaisesRegex(ReplayError, "before validated preflight"):
                install_import_boundary(self.engine)

    def test_import_resolution_rejects_other_installed_tensorfold(self):
        other = self.root / "other.py"
        other.write_text("pass\n")
        spec = importlib.util.spec_from_file_location("tensorfold", other)
        with patch("importlib.machinery.PathFinder.find_spec", return_value=spec):
            with self.assertRaisesRegex(ReplayError, "outside frozen"):
                FrozenImports(self.engine).find_spec("tensorfold")

    def test_loader_ignores_unverified_bytecode(self):
        loader = FrozenLoader("frozen_test", str(self.engine / "__init__.py"))
        with patch.object(loader, "get_data", return_value=b"sentinel = 73\n"):
            namespace = {}
            exec(loader.get_code("frozen_test"), namespace)
            self.assertEqual(namespace["sentinel"], 73)

    def test_graph_keys_include_core_and_four_row_preparation_in_solo(self):
        fixture = load_fixture(FIXTURE, allow_synthetic=True)
        fixture["config"]["active_streams"] = [0]
        for forward in fixture["forwards"]:
            forward["parts"] = forward["parts"][:1]
        self.assertTrue({1, 4, 8, 16, 32} <= set(graph_rows(fixture)))

    def test_lone_default_is_refused_instead_of_silently_overridden(self):
        env = dict(REQUIRED_ENV)
        del env["TF_GLM_MULTI_LONE"]
        with self.assertRaisesRegex(ReplayError, "MULTI_LONE"):
            environment(env)

    def test_environment_hash_catches_unlisted_execution_knob(self):
        a = environment(REQUIRED_ENV)
        b = environment({**REQUIRED_ENV, "TF_GLM_NEW_OVERLAP": "1"})
        self.assertNotEqual(digest(a), digest(b))

    def test_host_unrun_gate_cannot_claim_device_success(self):
        gate = {"schema": "gate-v1", "status": "unrun", "bindings": {}, "completed_ranks": [0, 1],
                "execution": "host", "checks": {"correctness": True}}
        with self.assertRaisesRegex(ReplayError, "unrun"):
            require_gate(gate, {}, ("correctness",), "gate-v1")
        gate["status"] = "pass"
        with self.assertRaisesRegex(ReplayError, "host doubles"):
            require_gate(gate, {}, ("correctness",), "gate-v1")

    def test_gate_binding_changed_arm_runtime_or_fixture_is_refused(self):
        binding = {"arm": "P", "runtime_sha256": "1" * 64, "fixture_sha256": "2" * 64}
        gate = {"schema": "gate-v1", "status": "pass", "bindings": binding, "completed_ranks": [0, 1],
                "execution": "device", "checks": {"correctness": True}}
        for key in binding:
            changed = copy.deepcopy(binding)
            changed[key] = "changed"
            with self.assertRaisesRegex(ReplayError, "binding mismatch"):
                require_gate(gate, changed, ("correctness",), "gate-v1")


if __name__ == "__main__":
    unittest.main()
