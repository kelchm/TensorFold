"""Reauthenticate actual validation receipts and derive fixture coverage without device imports."""
from pathlib import Path

import hashlib

from .common import digest, fields, hex_id, integer, parse_json, require
from .python_provenance import gate_binding, relative_file, require_gate

VALIDATION_SCHEMA = "glm-replay-python-validation-v2"


def artifact_receipt(spec, root):
    fields(spec, "path sha256", "bound receipt artifact")
    hex_id(spec["sha256"], "receipt artifact digest")
    path = relative_file(Path(root), spec["path"])
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "receipt artifact identity mismatch")
    return parse_json(raw.decode("utf-8"))


def fixture_coverage(fixture):
    forwards = fixture["forwards"]
    parts = [p for f in forwards for p in f["parts"]]
    return {"fixture_only": True, "full": any(all(p["kept"] == p["offered"] for p in f["parts"]) for f in forwards),
            "partial": any(p["kept"] < p["offered"] for p in parts), "repeat_reset": True}


def poisoned_rows(fixture):
    return sum(p["offered"] - p["kept"] for f in fixture["forwards"] for p in f["parts"])


def validate_receipt(validation, fixture, meta, runtime_sha256):
    from .python_validation import compare_oracle

    binding = gate_binding(fixture, meta, runtime_sha256)
    require_gate(validation, binding, ("correctness", "timing_neutral", "fresh_prefill"), VALIDATION_SCHEMA)
    fields(validation, "schema status execution bindings completed_ranks ranks logical_evidence_sha256 logical_evidence "
           "checks coverage performance", "validation receipt")
    require(all(type(r) is int for r in validation["completed_ranks"]), "invalid completed rank types")
    fields(validation["checks"], "correctness timing_neutral fresh_prefill default_server_neutral hardware_qualification",
           "validation checks")
    require(validation["checks"]["default_server_neutral"] == "not_applicable_unmodified_engine"
            and validation["checks"]["hardware_qualification"] == "unrun", "validation overclaimed external qualification")
    logical = validation.get("logical_evidence")
    require(isinstance(logical, dict) and logical.get("schema") == "glm-replay-python-logical-evidence-v2"
            and logical.get("bindings") == binding and logical.get("ranks") == validation.get("ranks")
            and digest(logical) == validation.get("logical_evidence_sha256"), "logical evidence hash/binding mismatch")
    ranks = validation.get("ranks")
    require(isinstance(ranks, list) and len(ranks) == 2 and [r.get("rank") for r in ranks] == [0, 1]
            and all(r.get("status") == "pass" for r in ranks), "validation lacks actual both-rank receipts")
    for rank in ranks:
        fields(rank, "rank status runtime_rank_sha256 snapshot_sha256 oracle_sha256 dispatch_sha256 "
               "evidence_sha256 evidence", "rank validation")
        integer(rank["rank"], "validation rank", 0, 1)
        hex_id(rank.get("runtime_rank_sha256"), "rank runtime digest")
    require(digest([r["runtime_rank_sha256"] for r in ranks]) == runtime_sha256,
            "validation runtime rank identities disagree with combined digest")
    require(digest(validation.get("coverage")) == digest(fixture_coverage(fixture)), "validation fixture coverage mismatch")
    for rank in ranks:
        evidence = rank.get("evidence")
        require(isinstance(evidence, dict) and digest(evidence) == rank.get("evidence_sha256"),
                "rank logical evidence hash mismatch")
        fields(evidence, "initial_states oracle_initial_states validated_forwards oracle_rows expected_kept_outputs "
               "expected_final dispatch poisoned_rows runs", "rank logical evidence")
        require(integer(evidence["poisoned_rows"], "poisoned rows") == poisoned_rows(fixture),
                "poisoned discarded-row count mismatch")
        require((evidence["poisoned_rows"] > 0) == validation["coverage"]["partial"], "partial/poison count mismatch")
        require(evidence["initial_states"] == evidence["oracle_initial_states"], "fresh oracle state mismatch")
        snapshots, oracle, dispatch = evidence["validated_forwards"], evidence["oracle_rows"], evidence["dispatch"]
        require(digest(snapshots) == rank.get("snapshot_sha256") and digest(oracle) == rank.get("oracle_sha256")
                and digest(dispatch) == rank.get("dispatch_sha256"), "rank snapshot/oracle/dispatch hash mismatch")
        validate_snapshots(fixture, evidence)
        compare_oracle(fixture, snapshots, oracle, evidence["oracle_initial_states"])
        outputs = [{sid: item["selected"] for sid, item in row["kept"].items()} for row in snapshots]
        require(digest(outputs) == digest(evidence["expected_kept_outputs"])
                and digest(snapshots[-1]) == digest(evidence["expected_final"]),
                "logical kept/final evidence mismatch")
        require(len(dispatch) == len(fixture["forwards"]), "incomplete dispatch evidence")
        for forward, row in zip(fixture["forwards"], dispatch):
            kept = sum(p["kept"] for p in forward["parts"])
            require(row == [1, 0, kept, kept, 1] and all(type(n) is int for n in row), "invalid graph/commit dispatch")
        runs = evidence["runs"]
        require(isinstance(runs, list) and len(runs) == 5, "missing repeated/timed/unsynchronized runs")
        for name, run in zip(("baseline", "repeat", "timed", "poisoned_whole", "natural_whole"), runs):
            fields(run, "name snapshots_sha256 dispatch_sha256 kept_outputs_sha256 final_sha256 poisoned_rows", "run evidence")
            require(run["name"] == name and run["dispatch_sha256"] == digest(dispatch)
                    and run["kept_outputs_sha256"] == digest(outputs)
                    and run["final_sha256"] == digest(evidence["expected_final"]), "run evidence mismatch")
            require(run["snapshots_sha256"] == (None if name.endswith("whole") else digest(snapshots)),
                    "repeated/timed snapshot mismatch")
            require(type(run["poisoned_rows"]) is int
                    and run["poisoned_rows"] == (0 if name == "natural_whole" else poisoned_rows(fixture)),
                    "run poison count mismatch")
    return validation


def validate_snapshots(fixture, evidence):
    from .python_logical import initial_positions, validate_capture, validate_states

    positions = initial_positions(fixture)
    validate_states(evidence["initial_states"], fixture, positions)
    snapshots = evidence["validated_forwards"]
    require(isinstance(snapshots, list) and len(snapshots) == len(fixture["forwards"]), "incomplete capture evidence")
    for forward, capture in zip(fixture["forwards"], snapshots):
        for p in forward["parts"]:
            positions[str(p["stream_id"])] = p["position"] + p["kept"]
        validate_capture(capture, fixture, forward["parts"], positions)
    rows = [(p, r) for f in fixture["forwards"] for p in f["parts"] for r in range(p["kept"])]
    oracle = evidence["oracle_rows"]
    require(isinstance(oracle, list) and len(oracle) == len(rows), "incomplete oracle evidence")
    for capture, (part, row) in zip(oracle, rows):
        one = {**part, "kept": 1, "position": part["position"] + row}
        sid = str(part["stream_id"])
        validate_capture(capture, fixture, [one], {sid: one["position"] + 1},
                         state_slots={sid} if row + 1 == part["kept"] else set())
