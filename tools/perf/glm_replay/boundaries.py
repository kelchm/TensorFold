"""Derive page-crossing qualification controls from immutable one-time token arrays."""
import argparse
import copy
import shutil
from pathlib import Path, PurePosixPath

from .common import file_digest, read_json, require, write_json
from .fixture import load_fixture, seal_fixture, validate_fixture
from .generate import _schedule


def derive_bundle(source, output):
    source, output = Path(source), Path(output)
    require(not output.exists(), "refusing to overwrite immutable derived bundle")
    parent = source / "manifest.json"
    manifest = read_json(parent)
    require(manifest["schema"] in ("glm-replay-fixture-bundle-v1", "glm-replay-verified-fixture-bundle-v3")
            and manifest["fixture_count"] == len(manifest["fixtures"]), "unsupported source bundle")
    bases, seen, kinds, rounds = {}, set(), set(), set()
    for entry in manifest["fixtures"]:
        name = entry["path"]
        require(PurePosixPath(name).name == name and name not in (".", "..", "manifest.json")
                and "\\" not in name and "\x00" not in name and name not in seen,
                "unsafe/duplicate bundle path")
        seen.add(name)
        path = source / name
        require(not path.is_symlink() and file_digest(path) == entry["sha256"], "source bundle file changed")
        fixture = load_fixture(path, allow_synthetic=True)
        kinds.add(fixture["kind"])
        rounds.add(len(fixture["forwards"]))
        for key in ("fixture_sha256", "tokens_sha256", "schedule_sha256", "config_sha256"):
            require(entry[key] == fixture[key], "source bundle entry identity mismatch")
        size = len(fixture["streams"][0]["prompt_tokens"])
        if size >= 512 and size % 512 == 0 and fixture["family"] not in bases:
            bases[fixture["family"]] = fixture
    require(set(bases) == {"prose", "code", "pi"}, "source needs all three page-aligned histories")
    require(len(kinds) == 1 and len(rounds) == 1 and manifest.get("kind", next(iter(kinds))) in kinds,
            "source bundle changed kind or round count")
    count = next(iter(rounds))
    require(manifest.get("rounds", count) == count, "source round count mismatch")
    output.mkdir(parents=True)
    for name in seen:
        shutil.copyfile(source / name, output / name)
    entries = copy.deepcopy(manifest["fixtures"])
    for family, base in bases.items():
        fixture = copy.deepcopy(base)
        fixture["config"]["active_streams"] = [0, 1, 2, 3]
        for stream in fixture["streams"]:
            size = len(stream["prompt_tokens"]) - 2
            stream["prompt_tokens"] = stream["prompt_tokens"][:size]
            stream["prompt_encoding"]["used_tokens"] = size
        fixture["preparation"], fixture["forwards"] = _schedule(
            fixture["streams"], [0, 1, 2, 3], [4] * 4, [2] * 4, 1, count, True)
        tag = f"qualification-page-edge-{family}-{size}"
        fixture["fixture_id"] = tag
        validate_fixture(seal_fixture(fixture), allow_synthetic=True)
        path = output / (tag + ".json")
        write_json(path, fixture)
        entries.append({"path": path.name, "sha256": file_digest(path),
                        **{k: fixture[k] for k in ("fixture_sha256", "tokens_sha256", "schedule_sha256", "config_sha256")},
                        "family": family, "context_tokens": size, "active_streams": [0, 1, 2, 3],
                        "widths": [4] * 4, "keeps": [2] * 4, "placement": fixture["config"]["placement"],
                        "qualification_only": True})
    schema = ("glm-replay-verified-fixture-bundle-v4" if manifest["schema"].endswith("v3")
              else "glm-replay-fixture-bundle-v2")
    result = {**manifest, "schema": schema, "fixtures": entries, "kind": next(iter(kinds)), "rounds": count,
              "status": "generated; strict derived-bundle checks pending; device UNRUN",
              "fixture_count": len(entries), "parent_manifest_sha256": file_digest(parent),
              "additional_encode_calls": 0, "device_checks": "UNRUN",
              "derivation": "retain parent bytes; qualification controls slice two prompt tokens and reuse all encodings"}
    write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    result = derive_bundle(args.source, args.output)
    print(f"{result['fixture_count']} fixtures; three qualification controls; no retokenization; device UNRUN")


if __name__ == "__main__":
    main()
