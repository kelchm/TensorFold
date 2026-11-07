"""Malformed input controls that remain invalid even after all hashes are recomputed."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay import ReplayError, load_fixture, next_anchor, offered_tokens, validate_fixture
from tools.perf.glm_replay.capacity import minimum_h, reservations, validate_pool
from tools.perf.glm_replay.common import canonical_bytes, file_digest, parse_json, write_json
from tools.perf.glm_replay.fixture import TARGET_REVISION, seal_fixture
from tools.perf.glm_replay.generate import PinnedTokenizer, SyntheticTokenizer, build_fixture, main, template_messages
from tools.perf.glm_replay.histories import source_material


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = build_fixture(rounds=3, preparation_rounds=1, repetitions=2)

    def assert_rejected(self, mutate, message):
        fixture = copy.deepcopy(self.base)
        mutate(fixture)
        seal_fixture(fixture)
        with self.assertRaisesRegex(ReplayError, message):
            validate_fixture(fixture, allow_synthetic=True)

    def test_synthetic_requires_explicit_host_opt_in(self):
        with self.assertRaisesRegex(ReplayError, "synthetic fixture refused"):
            validate_fixture(self.base)
        self.assertIs(validate_fixture(self.base, allow_synthetic=True), self.base)

    def test_generator_is_reproducible_and_four_streams_are_distinct(self):
        for family in ("prose", "code", "pi"):
            one = build_fixture(family, repetitions=2)
            two = build_fixture(family, repetitions=2)
            self.assertEqual(canonical_bytes(one), canonical_bytes(two))
            self.assertEqual(len({tuple(s["prompt_tokens"]) for s in one["streams"]}), 4)
            self.assertEqual(len({tuple(s["tape_tokens"]) for s in one["streams"]}), 4)
            self.assertTrue(all(s["license"] == "Apache-2.0" for s in one["provenance"]["sources"]))

    def test_source_bytes_hash_and_template_controls(self):
        self.assert_rejected(lambda f: f["provenance"]["sources"][0].update(text="changed"), "source hash")
        self.assert_rejected(lambda f: f["provenance"]["tokenizer"].update(template="changed"), "template hash")
        self.assert_rejected(lambda f: f["streams"][0].update(rendered_prompt="changed"), "rendered prompt hash")

    def test_hash_mismatch_and_expected_rank_bindings(self):
        fixture = copy.deepcopy(self.base)
        fixture["fixture_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayError, "fixture_sha256 mismatch"):
            validate_fixture(fixture, allow_synthetic=True)
        with self.assertRaisesRegex(ReplayError, "expected schedule_sha256 mismatch"):
            validate_fixture(self.base, True, {"schedule_sha256": "0" * 64})
        with self.assertRaisesRegex(ReplayError, "expected vocab_size mismatch"):
            validate_fixture(self.base, True, {"vocab_size": 155000})

    def test_duplicate_json_keys_nonfinite_and_boolean_ids(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ReplayError):
                parse_json(raw)
        self.assert_rejected(lambda f: f["forwards"][0]["parts"][0].update(stream_id=False), "invalid integer")
        self.assert_rejected(lambda f: f["config"].update(tp=True), "unsupported config")

    def test_unknown_fields_and_wrong_model(self):
        self.assert_rejected(lambda f: f.update(extra=True), "unexpected or missing")
        self.assert_rejected(lambda f: f["provenance"].update(target_revision="0" * 40), "target revision")
        self.assert_rejected(lambda f: f["config"].update(speed="fast"), "unsupported config")
        self.assert_rejected(lambda f: f["config"].update(proposer=True), "unsupported config")

    def test_duplicate_missing_streams_and_forward_ids(self):
        self.assert_rejected(lambda f: f["forwards"][0]["parts"][1].update(stream_id=0), "duplicate or inactive")
        self.assert_rejected(lambda f: f["forwards"][0]["parts"].pop(), "every active stream")
        self.assert_rejected(lambda f: f["forwards"][1].update(forward_id=0), "noncontiguous")
        self.assert_rejected(lambda f: f["forwards"].clear(), "empty timed")

    def test_partial_commit_reoffers_discarded_lookahead(self):
        fixture = build_fixture(widths=(8, 4, 2, 1), keeps=(2, 1, 1, 1), permute=True,
                                preparation_rounds=1, repetitions=2)
        first = fixture["forwards"][0]["parts"][0]
        second = next(p for p in fixture["forwards"][1]["parts"] if p["stream_id"] == 0)
        self.assertEqual(second["tape_offset"], 3)
        self.assertEqual(offered_tokens(fixture, second)[:6], offered_tokens(fixture, first)[2:])
        self.assertEqual(next_anchor(fixture, first), offered_tokens(fixture, second)[0])
        self.assertEqual([p["stream_id"] for p in fixture["forwards"][1]["parts"]], [3, 2, 1, 0])

    def test_preparation_advances_tape_outside_timed_id_space(self):
        part = self.base["forwards"][0]["parts"][0]
        self.assertEqual(part["tape_offset"], 1)
        self.assertEqual(part["position"], 129)
        self.assertEqual(self.base["forwards"][0]["forward_id"], 0)
        self.assert_rejected(lambda f: f["forwards"][0]["parts"][0].update(tape_offset=0), "advance by kept")

    def test_solo_preparation_covers_all_four_contexts(self):
        fixture = build_fixture(active_streams=(2,), repetitions=2)
        self.assertEqual([p["stream_id"] for p in fixture["preparation"][0]["parts"]], [0, 1, 2, 3])
        self.assertEqual(fixture["forwards"][0]["parts"][0]["tape_offset"], 1)
        fixture["preparation"][0]["parts"].pop()
        with self.assertRaisesRegex(ReplayError, "every active stream"):
            validate_fixture(seal_fixture(fixture), True)

    def test_count_position_width_and_vocab_controls(self):
        for key, value, error in (("offered", 17, "out of range"), ("kept", 0, "invalid integer"),
                                  ("kept", 5, "out of range"), ("position", 999, "position mismatch")):
            self.assert_rejected(lambda f: f["forwards"][0]["parts"][0].update({key: value}), error)
        self.assert_rejected(lambda f: [p.update(offered=16) for p in f["forwards"][0]["parts"]], "32 offered rows")
        self.assert_rejected(lambda f: f["streams"][0]["tape_tokens"].__setitem__(0, 256), "out of range")
        self.assert_rejected(lambda f: f["streams"][0]["prompt_tokens"].__setitem__(0, 0), "prompt prefix")

    def test_final_full_keep_requires_next_forced_anchor(self):
        fixture = build_fixture(active_streams=(0,), rounds=1, preparation_rounds=1, widths=(4, 4, 4, 4), repetitions=2)
        stream = fixture["streams"][0]
        stream["rendered_tape"] = "abcde"
        stream["tape_tokens"] = list(b"abcde")
        from tools.perf.glm_replay.common import text_digest
        stream["rendered_tape_sha256"] = text_digest("abcde")
        with self.assertRaisesRegex(ReplayError, "missing next forced anchor"):
            validate_fixture(seal_fixture(fixture), True)

    def test_owner_and_capacity_refusal_gates(self):
        self.assert_rejected(lambda f: f["streams"][0].update(owner=1), "owner assignment")
        self.assert_rejected(lambda f: f["config"]["pool"].update(h=255), "production-minimum")
        self.assert_rejected(lambda f: f["config"]["pool"].update(prompt_twins=0), "invalid integer")
        self.assert_rejected(lambda f: f["config"]["pool"].update(page_tokens=256), "page size")
        reversed_fixture = build_fixture(placement="reversed", repetitions=2)
        self.assertEqual([s["owner"] for s in reversed_fixture["streams"]], [1, 1, 0, 0])
        capacity = validate_pool(self.base)
        self.assertEqual(capacity["on_page_argument"] - capacity["off_page_argument"], 4)
        probe = copy.deepcopy(self.base)
        for stream in probe["streams"]:
            stream["prompt_tokens"] *= 1024
        self.assertGreater(minimum_h(probe, 512, 1 << 20), 512)

    def test_page_boundary_reservation_includes_final_anchor(self):
        before = build_fixture(context_tokens=494, active_streams=(0,), widths=(1, 1, 1, 1),
                               preparation_rounds=1, rounds=1, repetitions=2)
        after = build_fixture(context_tokens=495, active_streams=(0,), widths=(1, 1, 1, 1),
                              preparation_rounds=1, rounds=1, repetitions=2)
        self.assertEqual(reservations(before, 512, 1)["owner_pages"], [3, 3])
        self.assertEqual(reservations(after, 512, 1)["owner_pages"], [4, 3])

    def test_empty_preparation_and_unaligned_capacity_fail_before_engine_load(self):
        self.assert_rejected(lambda f: f["preparation"].clear(), "empty preparation")
        self.assert_rejected(lambda f: f["config"]["pool"].update(configured_capacity=131073), "multiple of 64")
        self.assert_rejected(lambda f: f["config"]["pool"].update(prompt_twins=2), "one transient twin")

    def test_encode_each_rendered_stream_once(self):
        codec = SyntheticTokenizer()
        with patch.object(codec, "encode", wraps=codec.encode) as encode:
            fixture = build_fixture(tokenizer=codec, repetitions=2)
        self.assertEqual(encode.call_count, 8)
        self.assertEqual([call.args[0] for call in encode.call_args_list],
                         [s[k] for s in fixture["streams"] for k in ("rendered_prompt", "rendered_tape")])

    def test_pi_tool_history_renders_object_arguments_without_mutating_source(self):
        source = source_material("pi", 0, 2)
        original = copy.deepcopy(source["messages"])
        normalized = template_messages(source["messages"])
        function = normalized[2]["tool_calls"][0]["function"]
        self.assertEqual(function["arguments"], {"path": "examples/harbor/ledger.txt"})
        self.assertEqual(source["messages"], original)
        function["arguments"] = "[]"
        with self.assertRaisesRegex(ReplayError, "must be JSON objects"):
            template_messages(normalized)

    def test_true_generator_requires_local_hashed_assets_before_import(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            write_json(manifest, {"schema": "glm-replay-tokenizer-assets-v1", "revision": "0" * 40, "files": []})
            with self.assertRaisesRegex(ReplayError, "pinned to target"):
                PinnedTokenizer(root, manifest)
            with self.assertRaises(FileNotFoundError):
                PinnedTokenizer(root / "missing", manifest)

    def test_asset_hash_path_and_unlisted_file_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "assets"
            directory.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (directory / name).write_text("{}", encoding="utf-8")
            manifest = Path(temp) / "manifest.json"
            assets = [{"path": p.name, "sha256": file_digest(p)} for p in sorted(directory.iterdir())]
            value = {"schema": "glm-replay-tokenizer-assets-v1", "revision": TARGET_REVISION, "files": assets}
            assets[0]["sha256"] = "0" * 64
            write_json(manifest, value)
            with self.assertRaisesRegex(ReplayError, "asset hash mismatch"):
                PinnedTokenizer(directory, manifest)
            assets[0]["sha256"] = file_digest(directory / assets[0]["path"])
            (directory / "unlisted.json").write_text("{}", encoding="utf-8")
            write_json(manifest, value)
            with self.assertRaisesRegex(ReplayError, "all local tokenizer directory files"):
                PinnedTokenizer(directory, manifest)
            assets[0]["path"] = "../escape.json"
            write_json(manifest, value)
            with self.assertRaisesRegex(ReplayError, "unsafe tokenizer asset path"):
                PinnedTokenizer(directory, manifest)

    def test_real_tokenizer_entries_can_be_fewer_than_padded_model_vocabulary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "assets"
            directory.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (directory / name).write_text("{}", encoding="utf-8")
            manifest = Path(temp) / "manifest.json"
            assets = [{"path": p.name, "sha256": file_digest(p)} for p in sorted(directory.iterdir())]
            write_json(manifest, {"schema": "glm-replay-tokenizer-assets-v1", "revision": TARGET_REVISION,
                                  "files": assets})
            vocabulary = {"ordinary": 0, "last-token": 154855}
            codec = SimpleNamespace(get_vocab=lambda: vocabulary, get_chat_template=lambda: "pinned-template")
            loader = SimpleNamespace(from_pretrained=lambda *a, **kw: codec)
            with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=loader)}):
                self.assertEqual(PinnedTokenizer(directory, manifest).vocab_size, 154880)
                vocabulary["invalid"] = 154880
                with self.assertRaisesRegex(ReplayError, "IDs exceed pinned model"):
                    PinnedTokenizer(directory, manifest)

    def test_immutable_fixture_cli_and_load(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.json"
            self.assertEqual(main([str(path), "--synthetic", "--repetitions", "2"]), 0)
            self.assertEqual(load_fixture(path, True)["tokens_sha256"], self.base["tokens_sha256"])
            with self.assertRaisesRegex(ReplayError, "overwrite immutable"):
                main([str(path), "--synthetic", "--repetitions", "2"])
            path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
            with self.assertRaisesRegex(ReplayError, "duplicate JSON key"):
                load_fixture(path, True)


if __name__ == "__main__":
    unittest.main()
