"""Driver-free post-block window requests and strictly bound sampled telemetry receipts."""
import hashlib
import os
import secrets
import time
from pathlib import Path, PurePosixPath

from .common import canonical_bytes, digest, fields, hex_id, integer, parse_json, require, string

WINDOW_SCHEMA = "glm-replay-block-window-v1"
TELEMETRY_SCHEMA = "glm-replay-telemetry-v1"
IDS = "run_id pair_id start_id start_order cell_id block_id"
BINDINGS = ("fixture_sha256 config_sha256 schedule_sha256 tokens_sha256 arm source_sha256 "
            "source_tree_sha256 runtime_sha256 checkpoints_sha256")


def relative_path(root, name):
    name = path_name(name)
    root = Path(root).resolve()
    result = root / name
    require(not any(p.is_symlink() for p in (result, *result.parents)), "symlink telemetry path")
    require(result.resolve().is_relative_to(root), "telemetry escaped receipts root")
    return result


def path_name(name):
    require(isinstance(name, str) and name, "unsafe telemetry path")
    path = PurePosixPath(name)
    require(str(path) == name and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and name != ".", "unsafe telemetry path")
    return name


def reference_name(spec):
    fields(spec, "path", "future telemetry reference")
    return path_name(spec["path"])


def reference(spec, root):
    return relative_path(root, reference_name(spec))


def validate_window(window):
    fields(window, "schema bindings ids completed_ranks ranks session_nonce campaign_sha256 checkpoints", "block window")
    require(window["schema"] == WINDOW_SCHEMA and window["completed_ranks"] == [0, 1], "incomplete block window")
    require(all(type(r) is int for r in window["completed_ranks"]), "invalid window rank types")
    hex_id(window["session_nonce"], "session nonce")
    hex_id(window["campaign_sha256"], "campaign digest")
    fields(window["bindings"], BINDINGS, "window bindings")
    fields(window["checkpoints"], "scheme target drafter", "window checkpoints")
    scheme = "file-manifest-v1" if window["bindings"]["arm"] == "P" else "tensorfold-full-identity-v1"
    require(window["checkpoints"]["scheme"] == scheme, "unsupported checkpoint identity scheme")
    for role in ("target", "drafter"):
        hashes = window["checkpoints"][role]
        require(isinstance(hashes, list) and len(hashes) == 2, "window checkpoints need both ranks")
        for sha in hashes:
            hex_id(sha, "checkpoint digest")
    for key, value in window["bindings"].items():
        if key == "arm":
            require(value in ("A", "B", "P"), "invalid window arm")
        else:
            hex_id(value, key)
    fields(window["ids"], IDS, "window identifiers")
    for key, value in window["ids"].items():
        integer(value, key, 0, 1) if key == "start_order" else string(value, key)
    require(isinstance(window["ranks"], list) and len(window["ranks"]) == 2, "missing rank windows")
    for rank, item in enumerate(window["ranks"]):
        fields(item, "rank realtime_start_ns realtime_end_ns", "rank window")
        require(integer(item["rank"], "rank", 0, 1) == rank, "rank window order mismatch")
        integer(item["realtime_start_ns"], "realtime start", 1)
        require(integer(item["realtime_end_ns"], "realtime end", 1) > item["realtime_start_ns"],
                "empty/reversed realtime window")
    return window


def block_window(bindings, ids, ranks, session_nonce, campaign_sha256, checkpoints):
    return validate_window({"schema": WINDOW_SCHEMA, "bindings": bindings, "ids": ids,
                            "session_nonce": session_nonce, "campaign_sha256": campaign_sha256, "checkpoints": checkpoints,
                            "completed_ranks": [0, 1], "ranks": ranks})


def validate_telemetry(value, window):
    validate_window(window)
    fields(value, "schema status window_sha256 bindings ids completed_ranks ranks "
           "session_nonce campaign_sha256 checkpoints", "telemetry")
    require(value["schema"] == TELEMETRY_SCHEMA and value["status"] == "measured", "telemetry unmeasured/wrong schema")
    require(value["window_sha256"] == digest(window) and value["bindings"] == window["bindings"]
            and value["ids"] == window["ids"] and value["completed_ranks"] == [0, 1], "telemetry window binding mismatch")
    require(all(type(r) is int for r in value["completed_ranks"]), "invalid telemetry rank types")
    fields(value["ids"], IDS, "telemetry identifiers")
    integer(value["ids"]["start_order"], "start order", 0, 1)
    require(all(value[key] == window[key] for key in ("session_nonce", "campaign_sha256", "checkpoints")),
            "telemetry session/campaign/checkpoint mismatch")
    require(isinstance(value["ranks"], list) and len(value["ranks"]) == 2, "telemetry lacks both ranks")
    for rank, row in enumerate(value["ranks"]):
        fields(row, "rank device_uuid realtime_start_ns realtime_end_ns max_sample_gap_ns samples", "rank telemetry")
        require(integer(row["rank"], "rank", 0, 1) == rank, "telemetry rank mismatch")
        string(row["device_uuid"], "device UUID")
        expected = window["ranks"][rank]
        integer(row["realtime_start_ns"], "telemetry start", 1)
        integer(row["realtime_end_ns"], "telemetry end", 1)
        require(row["realtime_start_ns"] == expected["realtime_start_ns"]
                and row["realtime_end_ns"] == expected["realtime_end_ns"], "telemetry rank window mismatch")
        gap = integer(row["max_sample_gap_ns"], "maximum sample gap", 1, 5_000_000_000)
        require(isinstance(row["samples"], list) and len(row["samples"]) >= 2, "empty telemetry sample coverage")
        stamps = []
        for sample in row["samples"]:
            fields(sample, "realtime_ns sm_clock_mhz memory_clock_mhz temperature_millicelsius "
                   "power_milliwatts used_memory_bytes", "telemetry sample")
            stamps.append(integer(sample["realtime_ns"], "sample time", 1))
            for name in ("sm_clock_mhz", "memory_clock_mhz"):
                integer(sample[name], name, 1)
            integer(sample["temperature_millicelsius"], "temperature", 0, 150000)
            integer(sample["power_milliwatts"], "power", 1)
            integer(sample["used_memory_bytes"], "used memory", 1)
        require(stamps[0] <= expected["realtime_start_ns"] < expected["realtime_end_ns"] <= stamps[-1],
                "telemetry samples do not cover rank window")
        require(any(expected["realtime_start_ns"] <= t <= expected["realtime_end_ns"] for t in stamps),
                "telemetry lacks an in-window sample")
        require(all(0 < b - a <= gap for a, b in zip(stamps, stamps[1:])), "telemetry sample gap/order failure")
    return value


def request_path(spec, root):
    path = reference(spec, root)
    return relative_path(root, spec["path"] + ".window.json")


def write_request(spec, root, window):
    validate_window(window)
    path = request_path(spec, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".glm-window-" + secrets.token_hex(16))
    stream = temporary.open("xb")
    try:
        with stream:
            stream.write(canonical_bytes(window) + b"\n")
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def wait_telemetry(spec, root, window, timeout_s=30, expected_sha256=None):
    require(type(timeout_s) in (int, float) and 0 <= timeout_s <= 60, "invalid telemetry wait bound")
    if expected_sha256 is not None:
        hex_id(expected_sha256, "telemetry digest")
    path = reference(spec, root)
    deadline = time.monotonic() + timeout_s
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    require(path.is_file(), "post-block telemetry unavailable before bounded deadline")
    reference(spec, root)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    require(expected_sha256 is None or sha == expected_sha256, "telemetry hash mismatch")
    return validate_telemetry(parse_json(raw.decode("utf-8")), window), sha
