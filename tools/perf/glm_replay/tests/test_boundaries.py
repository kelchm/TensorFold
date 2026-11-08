"""Verify genuine page crossing and preserved token provenance in derived host controls."""
import tempfile
import unittest
from pathlib import Path

from tools.perf.glm_replay.boundaries import derive_bundle
from tools.perf.glm_replay.bundle import generate_bundle
from tools.perf.glm_replay.common import ReplayError, read_json
from tools.perf.glm_replay.generate import SyntheticTokenizer
from tools.perf.glm_replay.python_qualification import suite_coverage


class BoundaryTests(unittest.TestCase):
    def test_derived_controls_cross_page_and_reuse_exact_encoding_and_parent_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root) / "source", Path(root) / "derived"
            parent = generate_bundle(source, SyntheticTokenizer(), 256, 1088, 3, contexts=(512,), repetitions=160)
            result = derive_bundle(source, output)
            self.assertEqual(result["additional_encode_calls"], 0)
            fixtures = [read_json(output / e["path"]) for e in result["fixtures"]]
            self.assertTrue(all(suite_coverage(fixtures).values()))
            for entry in parent["fixtures"]:
                self.assertEqual((source / entry["path"]).read_bytes(), (output / entry["path"]).read_bytes())
            for fixture in fixtures[-3:]:
                original = next(f for f in fixtures[:-3] if f["family"] == fixture["family"])
                for before, after in zip(original["streams"], fixture["streams"]):
                    self.assertEqual(before["prompt_encoding"]["token_ids"], after["prompt_encoding"]["token_ids"])
                    self.assertEqual(before["tape_tokens"], after["tape_tokens"])
                    self.assertEqual(before["prompt_tokens"][:-2], after["prompt_tokens"])
            with self.assertRaisesRegex(ReplayError, "overwrite"):
                derive_bundle(source, output)
            path = source / parent["fixtures"][0]["path"]
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ReplayError, "changed"):
                derive_bundle(source, Path(root) / "bad")
