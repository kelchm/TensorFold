"""Validate immutable fixtures or produce a strict offline paired-start report."""
import argparse
from pathlib import Path

from . import load_fixture, load_results, summarize_paired, validate_results
from .capacity import validate_pool
from .common import canonical_bytes, require, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate-fixture", "validate-results", "summarize"))
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--results", type=Path, action="append", default=[])
    parser.add_argument("--anchors-root", type=Path, action="append", help="retained native/Python phase anchor directories")
    parser.add_argument("--allow-synthetic", action="store_true", help="host controls only; no device qualification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    fixtures = {}
    for path in args.fixture:
        fixture = load_fixture(path, allow_synthetic=args.allow_synthetic)
        require(fixture["fixture_sha256"] not in fixtures, "duplicate fixture input")
        fixtures[fixture["fixture_sha256"]] = fixture
    if args.command == "validate-fixture":
        result = [{"fixture_sha256": key, "kind": value["kind"], "capacity": validate_pool(value)}
                  for key, value in fixtures.items()]
    else:
        records = load_results(args.results)
        if args.command == "validate-results":
            validate_results(records, fixtures, allow_synthetic=args.allow_synthetic, anchors_root=args.anchors_root)
            result = {"validated_rank_receipts": len(records)}
        else:
            result = summarize_paired(records, fixtures, allow_synthetic=args.allow_synthetic, anchors_root=args.anchors_root)
    if args.output:
        require(not args.output.exists(), "refusing to overwrite report")
        write_json(args.output, result)
    else:
        print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
