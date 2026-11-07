"""Generate a fixed fixture matrix with one tokenization per family/context history."""
import argparse
import copy
from pathlib import Path

from .common import file_digest, integer, require, text_digest, write_json
from .fixture import TARGET_REVISION, seal_fixture, validate_fixture
from .generate import PinnedTokenizer, SyntheticTokenizer, _schedule, build_fixture


def cases(screening):
    result = [(active, (width,) * 4, (width,) * 4, "balanced", False)
              for active in ((0,), (0, 1, 2, 3)) for width in ((1, 4, 8) if screening else (1, 4))]
    if screening:
        result.extend([((0, 1, 2, 3), (8, 4, 2, 1), (2, 1, 1, 1), "balanced", True),
                       ((0, 1, 2, 3), (4,) * 4, (4,) * 4, "reversed", False),
                       ((2,), (4,) * 4, (4,) * 4, "balanced", False)])
    return result


def generate_bundle(output, tokenizer, pool_h, configured_capacity, rounds,
                    contexts=(8192, 32768, 131072), repetitions=None, asset_provenance=None):
    output = Path(output)
    require(not output.exists(), "refusing to overwrite immutable fixture bundle")
    require(contexts and len(set(contexts)) == len(contexts), "distinct contexts required")
    for context in contexts:
        integer(context, "context", 1)
    integer(rounds, "rounds", 1)
    integer(pool_h, "pool_h", 1)
    integer(configured_capacity, "configured_capacity", 1)
    output.mkdir(parents=True)
    entries = []
    for family in ("prose", "code", "pi"):
        for index, context in enumerate(contexts):
            count = repetitions if repetitions is not None else max(160, context // 55 + 16)
            base = build_fixture(family, tokenizer, context_tokens=context, rounds=rounds,
                                 preparation_rounds=1, repetitions=count, pool_h=pool_h,
                                 configured_capacity=configured_capacity)
            for active, widths, keeps, placement, permute in cases(index == 0):
                fixture = copy.deepcopy(base)
                fixture["config"]["active_streams"] = list(active)
                fixture["config"]["placement"] = placement
                owners = [0, 0, 1, 1] if placement == "balanced" else [1, 1, 0, 0]
                for stream, owner in zip(fixture["streams"], owners, strict=True):
                    stream["owner"] = owner
                fixture["preparation"], fixture["forwards"] = _schedule(
                    fixture["streams"], list(active), list(widths), list(keeps), 1, rounds, permute)
                tag = (f"{family}-{context}-s{'-'.join(map(str, active))}-w{'-'.join(map(str, widths))}"
                       f"-k{'-'.join(map(str, keeps))}-{placement}")
                fixture["fixture_id"] = tag
                validate_fixture(seal_fixture(fixture), allow_synthetic=True)
                path = output / (tag + ".json")
                write_json(path, fixture)
                entries.append({"path": path.name, "sha256": file_digest(path),
                                **{key: fixture[key] for key in ("fixture_sha256", "tokens_sha256",
                                                                 "schedule_sha256", "config_sha256")},
                                "family": family, "context_tokens": context, "active_streams": list(active),
                                "widths": list(widths), "keeps": list(keeps), "placement": placement})
    manifest = {"schema": "glm-replay-fixture-bundle-v1", "kind": base["kind"],
                "target_revision": TARGET_REVISION, "template_sha256": text_digest(tokenizer.template),
                "asset_provenance_sha256": file_digest(asset_provenance) if asset_provenance else None,
                "tokenization_rule": "one encode each prompt/tape per family/context; shape variants reuse arrays",
                "encode_calls": 8 * 3 * len(contexts), "pool_h": pool_h,
                "configured_capacity": configured_capacity, "rounds": rounds, "fixture_count": len(entries),
                "device_checks": "UNRUN", "fixtures": entries}
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--asset-provenance", type=Path)
    parser.add_argument("--pool-h", type=int, required=True)
    parser.add_argument("--configured-capacity", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args(argv)
    require(args.synthetic or args.tokenizer_manifest is not None, "real generator requires tokenizer manifest")
    codec = SyntheticTokenizer() if args.synthetic else PinnedTokenizer(args.tokenizer_dir, args.tokenizer_manifest)
    manifest = generate_bundle(args.output, codec, args.pool_h, args.configured_capacity, args.rounds,
                               asset_provenance=args.asset_provenance)
    print(f"{manifest['kind']}: {manifest['fixture_count']} fixtures; device checks UNRUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
