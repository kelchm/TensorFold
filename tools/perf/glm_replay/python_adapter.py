"""External frozen-engine replay: host validation is separate from explicitly selected device execution."""
import argparse
import sys
from pathlib import Path

from .common import ReplayError, canonical_bytes, hex_id, require
from .fixture import load_fixture
from .python_provenance import gate_binding, install_import_boundary, preflight


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    host = sub.add_parser("host-check", help="validate only the immutable input; never import GPU libraries")
    host.add_argument("fixture")
    host.add_argument("--allow-synthetic", action="store_true")
    for name in ("preflight", "gpu-run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("fixture")
        cmd.add_argument("--source-root", required=True, help="frozen src/tensorfold directory")
        cmd.add_argument("--metadata")
        for role in ("artifacts", "target", "drafter"):
            cmd.add_argument(f"--{role}-root")
        cmd.add_argument("--rank", type=int, choices=(0, 1), default=0)
        if name == "preflight":
            cmd.add_argument("--allow-synthetic", action="store_true")
        else:
            cmd.add_argument("--mode", choices=("validate", "measure"), required=True)
            cmd.add_argument("--master", required=True)
            cmd.add_argument("--port", type=int, required=True)
            cmd.add_argument("--receipts-root", required=True)
            cmd.add_argument("--output", required=True)
    campaign = sub.add_parser("campaign", help="prevalidated cases/blocks in one production rank session")
    campaign.add_argument("campaign")
    campaign.add_argument("--campaign-root", required=True)
    campaign.add_argument("--source-root", required=True)
    campaign.add_argument("--receipts-root", required=True)
    for role in ("artifacts", "target", "drafter"):
        campaign.add_argument(f"--{role}-root", required=True)
    campaign.add_argument("--rank", type=int, choices=(0, 1), default=0)
    campaign.add_argument("--mode", choices=("preflight", "validate", "measure"), required=True)
    campaign.add_argument("--master")
    campaign.add_argument("--port", type=int)
    campaign.add_argument("--output")
    return p


def gpu_run(args, fixture, meta, report, roots):
    from .python_receipts import (artifact_receipt, build_record, measurement_gates, measure_block,
                                  publish_local, retain_anchor, write_exclusive)
    from .python_runtime import Runtime, exchange
    from .python_validation import validate_device

    require(meta is not None, "device mode requires complete pinned metadata")
    require(0 < args.port < 65536 and args.master.strip(), "invalid bootstrap address/port")
    require(not Path(args.output).exists(), "output already exists")
    require(not Path(args.output + ".anchors").exists(), "anchor output already exists")
    from .results import CONFIG_FIELDS, RECORD_FIELDS

    require("runtime_sha256" in CONFIG_FIELDS.split(), "common runtime-binding integration is required")
    require("timing_source" in RECORD_FIELDS.split(), "common honest timing-source integration is required")
    if args.mode == "measure":
        gate = artifact_receipt(meta["validation_gate"], args.receipts_root)
        expected_runtime = gate.get("bindings", {}).get("runtime_sha256")
        hex_id(expected_runtime, "prevalidated device runtime digest")
        measurement_gates(fixture, meta, expected_runtime, args.receipts_root)
    install_import_boundary(args.source_root)
    runtime = Runtime(fixture, meta, roots, args.rank, args.master, args.port, report)
    binding = gate_binding(fixture, meta, runtime.runtime_sha256)
    agreed = exchange(runtime.nccl.store, "binding", runtime.rank, binding)
    require(agreed[0] == agreed[1], "ranks disagree on effective runtime/arm/source binding")
    if args.mode == "measure":
        require(runtime.runtime_sha256 == expected_runtime, "actual loaded runtime differs from passed device gates")
        checks = measurement_gates(fixture, meta, runtime.runtime_sha256, args.receipts_root)
    if args.mode == "validate":
        validation = validate_device(runtime, binding)
        if args.rank == 0:
            write_exclusive(args.output, [validation])
        return
    block, anchor = measure_block(runtime, checks, args.receipts_root)
    telemetry = artifact_receipt(meta["telemetry"], args.receipts_root)
    require(telemetry.get("bindings") == binding and telemetry.get("completed_ranks") == [0, 1]
            and telemetry.get("block_id") == meta["ids"]["block_id"], "telemetry does not bind this block")
    record = build_record(runtime, block, checks, meta["telemetry"]["sha256"], anchor)
    if runtime.rank == 0:
        retain_anchor(args.output, anchor)
    publish_local(runtime, args.output, record)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "campaign":
            from .python_campaign import preflight_campaign, run_campaign

            roots = {key: getattr(args, f"{key}_root") for key in ("artifacts", "target", "drafter")}
            campaign, cases, report = preflight_campaign(args.campaign, args.campaign_root, args.source_root,
                                    roots, args.rank, args.receipts_root, args.mode)
            if args.mode != "preflight":
                require(args.output and args.master and args.port and 0 < args.port < 65536,
                        "device campaign requires output and valid bootstrap address/port")
                run_campaign(args, campaign, cases, report, roots)
                return 0
        elif args.command == "host-check":
            fixture = load_fixture(args.fixture, allow_synthetic=args.allow_synthetic)
            report = {"schema": "glm-replay-python-host-check-v1", "status": "host-checked",
                      "fixture_sha256": fixture["fixture_sha256"], "hardware_checks": "unrun"}
        else:
            roots = {key: getattr(args, f"{key}_root") for key in ("artifacts", "target", "drafter")}
            if args.metadata:
                require(all(roots.values()), "metadata preflight requires all artifact/checkpoint roots")
            fixture, meta, report = preflight(args.fixture, args.source_root, args.metadata, roots,
                                             args.rank, getattr(args, "allow_synthetic", False))
            if args.command == "gpu-run":
                gpu_run(args, fixture, meta, report, roots)
                return 0
        print(canonical_bytes(report).decode("utf-8"))
        return 0
    except (ReplayError, OSError, KeyError, TypeError) as exc:
        print(f"Python replay refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
