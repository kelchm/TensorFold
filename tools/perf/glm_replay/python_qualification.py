"""Offline suite qualification from revalidated device receipts and typed external records."""
import argparse
import sys
from pathlib import Path

from .common import ReplayError, canonical_bytes, fields, hex_id, read_json, require
from .fixture import load_fixture, validate_fixture
from .python_evidence import artifact_receipt, validate_receipt
from .python_external import common_binding, installed_listing, validate_activation, validate_listing
from .python_provenance import gate_binding, metadata_check, relative_file

SCHEMA = "glm-replay-python-qualification-v2"
REQUIRED_COVERAGE = ("solo", "shared", "full", "partial", "permutation", "page_edge", "repeat_reset")


def suite_coverage(fixtures):
    result = {key: False for key in REQUIRED_COVERAGE}
    orders = {}
    for fixture in fixtures:
        for forward in fixture["forwards"]:
            parts = forward["parts"]
            order = tuple(p["stream_id"] for p in parts)
            result["solo"] |= len(parts) == 1
            result["shared"] |= len(parts) > 1
            result["full"] |= all(p["kept"] == p["offered"] for p in parts)
            result["partial"] |= any(p["kept"] < p["offered"] for p in parts)
            result["page_edge"] |= any(p["position"] // 512 < (p["position"] + p["kept"]) // 512 for p in parts)
            if len(order) > 1:
                orders.setdefault(tuple(sorted(order)), set()).add(order)
        result["repeat_reset"] = True
    result["permutation"] = any(len(values) > 1 for values in orders.values())
    return result


def validate_suite(entries, fixture, meta, runtime_sha256, root, allow_synthetic=False):
    require(isinstance(entries, list) and entries, "qualification needs actual validation receipts and fixtures")
    fixtures, seen, rank_hashes = [], set(), None
    for entry in entries:
        fields(entry, "fixture validation", "qualification suite entry")
        case = validate_fixture(artifact_receipt(entry["fixture"], root), allow_synthetic=allow_synthetic)
        sha = case["fixture_sha256"]
        require(sha not in seen, "duplicate qualification fixture")
        seen.add(sha)
        validation = artifact_receipt(entry["validation"], root)
        validate_receipt(validation, case, meta, runtime_sha256)
        hashes = [r["runtime_rank_sha256"] for r in validation["ranks"]]
        require(rank_hashes is None or hashes == rank_hashes, "suite changed rank runtime identities")
        require(case["config"]["pool"] == fixture["config"]["pool"], "qualification changed common pool")
        rank_hashes = hashes
        fixtures.append(case)
    require(fixture["fixture_sha256"] in seen, "measured fixture absent from qualification suite")
    coverage = suite_coverage(fixtures)
    require(all(coverage.values()), "qualification suite lacks derived coverage: " +
            ",".join(k for k, v in coverage.items() if not v))
    return coverage, rank_hashes


def build_qualification(entries, activation, installed_sources, fixture, meta, runtime_sha256, root,
                        allow_synthetic=False):
    hex_id(runtime_sha256, "qualification runtime digest")
    binding = gate_binding(fixture, meta, runtime_sha256)
    coverage, ranks = validate_suite(entries, fixture, meta, runtime_sha256, root, allow_synthetic)
    require(isinstance(installed_sources, list) and len(installed_sources) == 2, "both installed listings required")
    for rank, spec in enumerate(installed_sources):
        validate_listing(artifact_receipt(spec, root), meta, rank)
    validate_activation(artifact_receipt(activation, root), binding, meta, ranks, root)
    return {"schema": SCHEMA, "status": "pass", "bindings": common_binding(binding), "completed_ranks": [0, 1],
            "suite": entries, "coverage": coverage, "activation": activation, "installed_sources": installed_sources}


def validate_qualification(value, fixture, meta, runtime_sha256, root, allow_synthetic=False):
    fields(value, "schema status bindings completed_ranks suite coverage activation installed_sources", "qualification")
    require(value["schema"] == SCHEMA and value["status"] == "pass", "old opaque/unrun qualification refused")
    rebuilt = build_qualification(value["suite"], value["activation"], value["installed_sources"], fixture, meta,
                                  runtime_sha256, root, allow_synthetic)
    require(value == rebuilt, "qualification bindings/coverage differ from actual evidence")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("installed-list", help="hash an already extracted installed-image tree without drivers")
    listing.add_argument("--installed-root", required=True)
    listing.add_argument("--rank", type=int, choices=(0, 1), required=True)
    listing.add_argument("--image-digest", required=True)
    listing.add_argument("--device-image-digest", required=True)
    build = sub.add_parser("build", help="authenticate retained suite and external observations; never contact a device")
    build.add_argument("--request", required=True)
    build.add_argument("--fixture", required=True)
    build.add_argument("--runtime-sha256", required=True)
    build.add_argument("--receipts-root", required=True)
    for cmd in (listing, build):
        cmd.add_argument("--metadata", required=True)
        cmd.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        meta = read_json(args.metadata)
        if args.command == "installed-list":
            result = installed_listing(args.installed_root, meta, args.rank, args.image_digest, args.device_image_digest)
        else:
            request = read_json(args.request)
            fields(request, "suite activation installed_sources", "qualification build request")
            fixture = load_fixture(args.fixture)
            metadata_check(meta, fixture)
            result = build_qualification(request["suite"], request["activation"], request["installed_sources"],
                                         fixture, meta, args.runtime_sha256, args.receipts_root)
        with Path(args.output).open("xb") as stream:
            stream.write(canonical_bytes(result) + b"\n")
        return 0
    except (ReplayError, OSError, KeyError, TypeError) as exc:
        print(f"Python qualification refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
