"""A matrix reuses history IDs; shape changes never invoke another tokenizer pass."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.perf.glm_replay import ReplayError, load_fixture
from tools.perf.glm_replay.bundle import generate_bundle
from tools.perf.glm_replay.generate import SyntheticTokenizer


class BundleTests(unittest.TestCase):
    def test_matrix_tokenizes_once_and_preserves_exact_arrays_across_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            codec = SyntheticTokenizer()
            with patch.object(codec, "encode", wraps=codec.encode) as encode:
                manifest = generate_bundle(output, codec, 1280, 262144, 2, contexts=(128,), repetitions=2)
            self.assertEqual(encode.call_count, 24)
            self.assertEqual(manifest["fixture_count"], 27)
            families = {}
            for entry in manifest["fixtures"]:
                fixture = load_fixture(output / entry["path"], allow_synthetic=True)
                ids = [(s["prompt_tokens"], s["tape_tokens"]) for s in fixture["streams"]]
                self.assertEqual(families.setdefault(fixture["family"], ids), ids)
                self.assertEqual(fixture["config"]["pool"]["h"], 1280)
            with self.assertRaisesRegex(ReplayError, "overwrite immutable"):
                generate_bundle(output, codec, 1280, 262144, 2, contexts=(128,), repetitions=2)

    def test_invalid_bundle_capacity_is_refused_and_never_gets_completion_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bundle"
            with self.assertRaisesRegex(ReplayError, "production-minimum"):
                generate_bundle(output, SyntheticTokenizer(), 1, 262144, 2, contexts=(128,), repetitions=2)
            self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
