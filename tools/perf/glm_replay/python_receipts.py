"""Bound device gates and common rank receipts; no output before both ranks complete."""
import hashlib
from pathlib import Path

from .common import canonical_bytes, digest, require
from .python_control import forward_receipt
from .python_provenance import HASHES, gate_binding
from .python_evidence import artifact_receipt, validate_receipt
from .python_runtime import exchange, module


def measurement_gates(fixture, meta, runtime_sha256, root):
    from .python_qualification import validate_qualification

    validation = artifact_receipt(meta["validation_gate"], root)
    validate_receipt(validation, fixture, meta, runtime_sha256)
    external = artifact_receipt(meta["external_gate"], root)
    validate_qualification(external, fixture, meta, runtime_sha256, root)
    require(meta["server_gate"] is None, "P server gate must be null: unmodified frozen engine")
    return {"correctness": True, "timing_neutral": True,
            "default_server_neutral": "not_applicable_unmodified_engine",
            "fresh_prefill": True, "proposer_calls": 0, "lost": 0, "carries": 0, "twins": 0}


def block_telemetry(runtime, block, spec, root, timeout_s=30):
    from .telemetry import block_window, wait_telemetry, write_request

    binding = gate_binding(runtime.fixture, runtime.meta, runtime.runtime_sha256)
    ranks = exchange(runtime.nccl.store, f"block-window/{digest(runtime.meta['ids'])}", runtime.rank,
                     block["realtime_window"])
    window = block_window(binding, runtime.meta["ids"], ranks, runtime.session_nonce,
                          runtime.campaign_sha256, runtime.checkpoints)
    if runtime.rank == 0:
        write_request(spec, root, window)
    telemetry, sha = wait_telemetry(spec, root, window, timeout_s)
    qualification = artifact_receipt(runtime.meta["external_gate"], root)
    activation = artifact_receipt(qualification["activation"], root)
    require([r["device_uuid"] for r in telemetry["ranks"]] ==
            [r["device"]["uuid"] for r in activation["ranks"]], "telemetry device identity mismatch")
    agreed = exchange(runtime.nccl.store, f"telemetry/{digest(runtime.meta['ids'])}", runtime.rank, sha)
    require(agreed == [sha, sha], "ranks read different telemetry bytes")
    return sha


def measure_block(runtime, checks, root):
    require(checks.get("correctness") is True and checks.get("timing_neutral") is True,
            "measurement requires passed correctness/timing gates")
    validation = artifact_receipt(runtime.meta["validation_gate"], root)
    expected = validation["ranks"][runtime.rank]["evidence"]
    anchor = runtime.run(enabled=True, final_capture=True)
    headline = runtime.run(enabled=False, final_capture=True)
    for block in (anchor, headline):
        require(block["kept_outputs"] == expected["expected_kept_outputs"], "measured kept outputs differ from validation")
        require(block["final"] == expected["expected_final"], "measured final logical state/taps differ from validation")
    local = {"rank": runtime.rank, "wall_ns": anchor["wall_ns"], "phases_ns": anchor["phases"],
             "kept_outputs_sha256": digest(anchor["kept_outputs"]), "final_sha256": digest(anchor["final"])}
    ranks = exchange(runtime.nccl.store, f"phase-anchor/{digest(runtime.meta['ids'])}", runtime.rank, local)
    require([r.get("rank") for r in ranks] == [0, 1], "phase anchor lacks both ranks")
    combined = {"schema": "glm-replay-python-phase-anchor-v1", "bindings": validation["bindings"],
                "session_nonce": runtime.session_nonce, "campaign_sha256": runtime.campaign_sha256,
                "checkpoints": runtime.checkpoints,
                "ids": runtime.meta["ids"], "logical_evidence_sha256": validation["logical_evidence_sha256"],
                "timing_source": "separate_observer_on_anchor", "completed_ranks": [0, 1], "ranks": ranks}
    return headline, combined


def retain_anchor(output, anchor):
    sha = hashlib.sha256(canonical_bytes(anchor) + b"\n").hexdigest()
    directory = Path(str(output) + ".anchors")
    directory.mkdir(exist_ok=True)
    write_exclusive(directory / f"{sha}.json", [anchor])
    return sha


def effective_config(runtime):
    cfg = runtime.fixture["config"]
    w = runtime.w
    tp = module("tp").of(w)
    dsa = len([l for l in w.layers if l.kind == "dsa"])
    require((dsa, tp.heads, w.cfg.kv_lora) == (11, 32, 512), "loaded geometry differs from pinned GLM")
    require(runtime.e.caches.kv == "fp8" and runtime.e.slots.count == 4, "effective cache/slots mismatch")
    row_bytes = tp.heads * w.cfg.kv_lora * 2
    return {"engine": "python", "speed": "standard", "tp": 2, "slots": 4, "dense_bits": 4,
            "head_bits": 4, "kv": "fp8", "kv_shard": False, "multi_lone": 0,
            "pool_h": cfg["pool"]["h"], "page_argument": 2 * cfg["pool"]["h"], "page_tokens": 512,
            "configured_capacity": cfg["pool"]["configured_capacity"], "dsa_layers": dsa,
            "local_heads": tp.heads, "latent_width": w.cfg.kv_lora, "row_dtype_bytes": 2,
            "query_bytes_per_row": row_bytes, "output_bytes_per_row": row_bytes,
            "graph_rows": runtime.graph_rows, "runtime_sha256": runtime.runtime_sha256}


def memory_receipt(runtime):
    caches = runtime.e.caches
    page = 512
    planes = caches.arena.planes
    latents = set(caches.kc)
    pooled = {trio[2] for trio in caches.index}
    def row_bytes(t):
        return t[0].numel() * t.element_size()
    key = sum(page // planes[i].div * row_bytes(planes[i].tensor) for i in pooled)
    latent = sum(page * row_bytes(planes[i].tensor) for i in latents)
    return {"logical_pages": 2 * runtime.fixture["config"]["pool"]["h"], "key_page_bytes": key,
            "latent_page_bytes": latent, "row_page_bytes": caches.arena.nbytes(),
            "tables_bytes": None, "scratch_bytes": None, "other_bytes": None, "peak_physical_bytes": None}


def build_record(runtime, block, checks, telemetry_sha256, anchor=None):
    fixture, meta = runtime.fixture, runtime.meta
    require(block["completed_ranks"] == [0, 1], "rank completion required before result construction")
    require(anchor is not None and block["phases"] is None, "missing separate anchor or headline observer enabled")
    phases = anchor["ranks"][runtime.rank]["phases_ns"]
    require(phases is not None and len(phases) == len(block["frames"])
            == len(fixture["forwards"]), "missing measured timings or incomplete schedule")
    for key in ("correctness", "timing_neutral", "fresh_prefill"):
        require(checks.get(key) is True, f"measurement gate not passed: {key}")
    require(checks.get("default_server_neutral") == "not_applicable_unmodified_engine",
            "P requires explicit unmodified-engine server N/A")
    config = effective_config(runtime)
    return {"schema": "glm-replay-result-v1", **{k: fixture[k] for k in HASHES.split()},
            "session_nonce": runtime.session_nonce, "campaign_sha256": runtime.campaign_sha256,
            "checkpoints": runtime.checkpoints, "layout_checks": 0,
            **meta["ids"], "arm": "P", "rank": runtime.rank, "status": "complete", "exclusion_reason": None,
            "source": meta["source"], "effective_config": config, "effective_config_sha256": digest(config),
            "preparation_ns": block["preparation_ns"], "wall_ns": block["wall_ns"], "completed_ranks": [0, 1],
            "timing_source": {"wall": "observer_off", "phases": "separate_observer_on_anchor",
                              "anchor_sha256": hashlib.sha256(canonical_bytes(anchor) + b"\n").hexdigest()},
            "forwards": [forward_receipt(f, t) for f, t in zip(block["frames"], phases)],
            "checks": checks, "memory": memory_receipt(runtime), "telemetry_sha256": telemetry_sha256}


def write_exclusive(path, records):
    data = b"".join(canonical_bytes(r) + b"\n" for r in records)
    with Path(path).open("xb") as stream:
        stream.write(data)


def collect_records(runtime, record, anchor):
    records = exchange(runtime.nccl.store, f"output/{digest(runtime.meta['ids'])}", runtime.rank, record)
    require([r.get("rank") for r in records] == [0, 1], "missing/duplicate rank output")
    for key in ("effective_config_sha256", "fixture_sha256", "source", "checks", "completed_ranks"):
        require(records[0][key] == records[1][key], f"rank output disagreement: {key}")
    from .results import validate_results

    sha = record["timing_source"]["anchor_sha256"]
    validate_results(records, {runtime.fixture["fixture_sha256"]: runtime.fixture},
                     anchors={sha: canonical_bytes(anchor) + b"\n"})
    return records


def publish_local(runtime, output, record, anchor):
    records = collect_records(runtime, record, anchor)
    if runtime.rank == 0:
        write_exclusive(output, records)
