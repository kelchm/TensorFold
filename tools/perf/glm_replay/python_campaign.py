"""An ordered developer case loop inside one real production engine/rank session."""
import copy
from pathlib import Path

from .capacity import reservations
from .common import canonical_bytes, digest, fields, file_digest, hex_id, read_json, require
from .fixture import load_fixture
from .python_provenance import graph_rows, metadata_check, preflight, relative_file
from .python_receipts import artifact_receipt, measurement_gates
from .telemetry import reference, request_path


def bound_file(root, spec):
    fields(spec, "path sha256", "campaign input")
    hex_id(spec["sha256"], "campaign input digest")
    path = relative_file(Path(root), spec["path"])
    require(file_digest(path) == spec["sha256"], "campaign input identity mismatch")
    return path


def preflight_campaign(path, root, source_root, roots, rank, receipts_root, mode):
    campaign = read_json(path)
    fields(campaign, "schema status arm run_id pair_id start_id start_order cases", "campaign")
    require(campaign["schema"] == "glm-replay-python-campaign-v2" and campaign["status"] == "ready"
            and campaign["arm"] == "P", "unrun/unsupported Python campaign")
    require(isinstance(campaign["cases"], list) and campaign["cases"], "empty campaign")
    cases, cells, block_ids, telemetry_paths = [], {}, set(), set()
    counts = {}
    common = None
    expected_runtime = None
    for case in campaign["cases"]:
        fields(case, "fixture metadata blocks", "campaign case")
        fixture_path = bound_file(root, case["fixture"])
        metadata_path = bound_file(root, case["metadata"])
        fixture, meta = load_fixture(fixture_path), read_json(metadata_path)
        metadata_check(meta, fixture)
        identity = {key: meta[key] for key in ("source", "source_tree_sha256", "artifacts", "checkpoints", "launch")}
        identity["pool"] = fixture["config"]["pool"]
        require(common is None or identity == common, "campaign changed pool/capacity/source/runtime launch")
        common = identity
        for key in ("run_id", "pair_id", "start_id", "start_order"):
            require(type(meta["ids"][key]) is type(campaign[key]) and meta["ids"][key] == campaign[key],
                    f"campaign changed genuine process-start identity: {key}")
        cell = meta["ids"]["cell_id"]
        require(cell not in cells or cells[cell] == fixture["fixture_sha256"], "campaign cell changed fixture")
        cells[cell] = fixture["fixture_sha256"]
        require(isinstance(case["blocks"], list) and case["blocks"], "campaign entry needs explicit blocks")
        for block in case["blocks"]:
            fields(block, "block_id telemetry", "campaign block")
            bid = block["block_id"]
            require(isinstance(bid, str) and bid.strip() and (cell, bid) not in block_ids,
                    "duplicate/invalid cell/block ID")
            block_ids.add((cell, bid))
            counts[cell] = counts.get(cell, 0) + 1
            path = reference(block["telemetry"], receipts_root)
            request = request_path(block["telemetry"], receipts_root)
            require(path not in telemetry_paths and request not in telemetry_paths, "reused telemetry/request path")
            telemetry_paths.update((path, request))
            if mode == "measure":
                require(not path.exists() and not request.exists(), "future telemetry/request path already exists")
        if mode == "measure":
            gate = artifact_receipt(meta["validation_gate"], receipts_root)
            runtime = gate.get("bindings", {}).get("runtime_sha256")
            hex_id(runtime, "prevalidated device runtime digest")
            require(expected_runtime is None or runtime == expected_runtime, "campaign gates changed session runtime")
            expected_runtime = runtime
            measurement_gates(fixture, meta, runtime, receipts_root)
        cases.append({"fixture": fixture, "metadata": meta, "fixture_path": fixture_path,
                      "metadata_path": metadata_path, "blocks": case["blocks"]})
    require(mode != "measure" or all(n >= 5 for n in counts.values()),
            "campaign measurement needs at least five explicit blocks per cell/start")
    first = cases[0]
    _, _, report = preflight(first["fixture_path"], source_root, first["metadata_path"], roots, rank)
    bank = sorted({row for case in cases for row in graph_rows(case["fixture"])})
    sizes = [max(reservations(c["fixture"], 512, 1)["stream_extents"][sid] for c in cases) for sid in range(4)]
    pool = first["fixture"]["config"]["pool"]
    require(sum((n + 2047) // 2048 * 2048 for n in sizes) <= 2 * pool["h"] * 512,
            "campaign maximum all-slot reservations exceed fixed pool")
    report.update({"campaign_sha256": digest(campaign), "cases": len(cases), "blocks": len(block_ids),
                   "needed_graph_rows": bank, "graph_rows": list(range(1, 33)), "expected_runtime_sha256": expected_runtime})
    return campaign, cases, report


def run_campaign(args, campaign, cases, report, roots):
    from .python_provenance import gate_binding, install_import_boundary
    from .python_receipts import block_telemetry, build_record, collect_records, measure_block, retain_anchor, write_exclusive
    from .python_evidence import validate_receipt
    from .python_runtime import Runtime, exchange
    from .python_validation import validate_device
    from .results import CONFIG_FIELDS, RECORD_FIELDS

    require("runtime_sha256" in CONFIG_FIELDS.split(), "common runtime-binding integration is required")
    require("timing_source" in RECORD_FIELDS.split(), "common honest timing-source integration is required")
    require(not Path(args.output).exists(), "campaign output already exists")
    require(not Path(args.output + ".anchors").exists(), "campaign anchor output already exists")
    first = cases[0]
    install_import_boundary(args.source_root)
    runtime = Runtime(first["fixture"], first["metadata"], roots, args.rank, args.master, args.port, report,
                      fixtures=list({c["fixture"]["fixture_sha256"]: c["fixture"] for c in cases}.values()),
                      campaign_identity=report["campaign_sha256"])
    if args.mode == "measure":
        require(runtime.runtime_sha256 == report["expected_runtime_sha256"], "campaign actual runtime differs from gates")
    validations, output = {}, None
    destination = Path(args.output)
    if args.mode == "validate" and runtime.rank == 0:
        destination.mkdir()
    try:
        for index, case in enumerate(cases):
            runtime.select_case(case["fixture"], case["metadata"])
            binding = gate_binding(runtime.fixture, runtime.meta, runtime.runtime_sha256)
            if args.mode == "validate":
                sha = runtime.fixture["fixture_sha256"]
                path = destination / f"validation-{sha}.json"
                if sha in validations:
                    validation = validations[sha]
                    validate_receipt(validation, runtime.fixture, runtime.meta, runtime.runtime_sha256)
                    if runtime.rank == 0:
                        require(path.read_bytes() == canonical_bytes(validation) + b"\n",
                                "previous successful validation receipt changed")
                else:
                    validation = validate_device(runtime, binding)
                    validate_receipt(validation, runtime.fixture, runtime.meta, runtime.runtime_sha256)
                    if runtime.rank == 0:
                        write_exclusive(path, [validation])
                    validations[sha] = validation
                exchange(runtime.nccl.store, f"campaign-validation/{index}", runtime.rank,
                         {"fixture_sha256": sha, "validation_sha256": digest(validation)})
                continue
            checks = measurement_gates(runtime.fixture, runtime.meta, runtime.runtime_sha256, args.receipts_root)
            for block_index, spec in enumerate(case["blocks"]):
                runtime.meta = copy.deepcopy(case["metadata"])
                runtime.meta["ids"]["block_id"] = spec["block_id"]
                result, anchor = measure_block(runtime, checks, args.receipts_root)
                telemetry_sha = block_telemetry(runtime, result, spec["telemetry"], args.receipts_root)
                record = build_record(runtime, result, checks, telemetry_sha, anchor)
                records = collect_records(runtime, record, anchor)
                if runtime.rank == 0:
                    retain_anchor(args.output, anchor)
                    if output is None:
                        output = Path(args.output).open("xb")
                    output.write(b"".join(canonical_bytes(r) + b"\n" for r in records))
                    output.flush()
                exchange(runtime.nccl.store, f"campaign-block/{index}/{block_index}", runtime.rank,
                         {"written": True, "block_id": spec["block_id"]})
    finally:
        if output is not None:
            output.close()
