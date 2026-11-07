"""Typed external activation and complete both-rank installed-image source evidence."""
from pathlib import Path

from .common import digest, fields, hex_id, integer, read_json, require, string
from .python_evidence import artifact_receipt
from .python_provenance import SOURCE_MANIFEST, environment, verify_files, verify_sources

LISTING_SCHEMA = "glm-replay-python-installed-source-v1"
ACTIVATION_SCHEMA = "glm-replay-python-activation-v1"


def common_binding(binding):
    return {k: binding[k] for k in ("arm", "source_sha256", "source_tree_sha256", "runtime_sha256", "checkpoints_sha256")}


def source_pins(meta):
    pins = read_json(SOURCE_MANIFEST)
    require(len(pins["files"]) == 439 and digest(pins) == meta["source_tree_sha256"], "complete 439-file source pin required")
    require(pins["engine_commit"] == meta["source"]["engine_commit"]
            and pins["kit_commit"] == meta["source"]["kit_commit"]
            and pins["device_image"] == meta["source"]["device_image_digest"], "installed source/image pin mismatch")
    return pins


def validate_listing(value, meta, rank):
    fields(value, "schema status rank source source_tree_sha256 files", "installed-image listing")
    require(value["schema"] == LISTING_SCHEMA and value["status"] == "listed"
            and integer(value["rank"], "listing rank", 0, 1) == rank, "installed listing schema/rank mismatch")
    require(value["source"] == meta["source"] and value["source_tree_sha256"] == meta["source_tree_sha256"],
            "installed source/image identity mismatch")
    expected = [{k: a[k] for k in ("path", "bytes", "sha256")} for a in source_pins(meta)["files"]]
    require(value["files"] == expected, "installed listing must match all 439 source pins")
    return value


def installed_listing(installed_root, meta, rank, image_digest, device_image_digest):
    integer(rank, "listing rank", 0, 1)
    require(image_digest == meta["source"]["image_digest"]
            and device_image_digest == meta["source"]["device_image_digest"], "extracted image identity mismatch")
    pins = source_pins(meta)
    require(verify_sources(Path(installed_root)) == pins, "installed extraction differs from frozen source")
    value = {"schema": LISTING_SCHEMA, "status": "listed", "rank": rank, "source": meta["source"],
             "source_tree_sha256": digest(pins),
             "files": [{k: a[k] for k in ("path", "bytes", "sha256")} for a in pins["files"]]}
    return validate_listing(value, meta, rank)


def validate_activation(value, binding, meta, runtime_ranks, root):
    fields(value, "schema status execution bindings completed_ranks ranks", "external activation")
    require(value["schema"] == ACTIVATION_SCHEMA and value["status"] == "observed"
            and value["execution"] == "device" and value["completed_ranks"] == [0, 1], "activation remains UNRUN")
    require(all(type(r) is int for r in value["completed_ranks"]), "invalid activation completed ranks")
    require(value["bindings"] == common_binding(binding), "activation source/runtime/checkpoint binding mismatch")
    ranks = value["ranks"]
    require(isinstance(ranks, list) and len(ranks) == 2, "missing both-rank activation")
    devices = []
    for rank, item in enumerate(ranks):
        fields(item, "rank runtime_rank_sha256 source environment device transport kernels", "rank activation")
        require(integer(item["rank"], "activation rank", 0, 1) == rank
                and item["runtime_rank_sha256"] == runtime_ranks[rank] and item["source"] == meta["source"],
                "activation rank/source/runtime mismatch")
        device = item["device"]
        fields(device, "uuid pci_bus_id compute_capability driver_version", "device activation")
        for key in ("uuid", "pci_bus_id", "driver_version"):
            string(device[key], key)
        require(isinstance(device["compute_capability"], list) and len(device["compute_capability"]) == 2,
                "missing device capability")
        for n in device["compute_capability"]:
            integer(n, "compute capability")
        devices.append(device["uuid"])
        env = item["environment"]
        require(environment(env) == env and digest(env) == meta["launch"]["environment_sha256"][rank],
                "activation launch environment mismatch")
        validate_transport(item["transport"], meta)
        require(item["transport"]["peer_rank"] == 1 - rank
                and item["transport"]["kind"] == env.get("TF_GLM_COMM", "nccl"), "activation transport peer/mode mismatch")
        validate_kernels(item["kernels"], meta, root)
    require(len(set(devices)) == 2, "activation reused one device for both ranks")
    return value


def validate_transport(value, meta):
    fields(value, "kind implementation bootstrap world peer_rank collective_counts sent_bytes received_bytes "
           "route_devices", "production transport")
    require(value["kind"] in ("nccl", "roce") and value["bootstrap"] == "tensorfold.cuda.comm.NCCL",
            "nonproduction transport/bootstrap")
    expected = "tensorfold.cuda.comm.NCCL" if value["kind"] == "nccl" else "tensorfold.cuda.roce.RoceComm"
    require(value["implementation"] == expected and integer(value["world"], "world", 2, 2) == 2,
            "production transport implementation/world mismatch")
    integer(value["peer_rank"], "transport peer", 0, 1)
    counts = value["collective_counts"]
    fields(counts, "all_gather barrier", "observed production collectives")
    for n in counts.values():
        integer(n, "observed collective count", 1)
    for key in ("sent_bytes", "received_bytes"):
        integer(value[key], key, 1)
    require(isinstance(value["route_devices"], list) and value["route_devices"], "missing observed transport routes")
    for route in value["route_devices"]:
        string(route, "transport route")


def validate_kernels(value, meta, root):
    fields(value, "bundle loaded", "loaded kernel evidence")
    require(value["bundle"] == meta["artifacts"]["fatbin"], "loaded kernel bundle identity mismatch")
    require(value["bundle"]["sha256"] == meta["source"]["fatbin_sha256"], "kernel/source identity mismatch")
    verify_files(Path(root), [value["bundle"]])
    bundle = artifact_receipt({k: value["bundle"][k] for k in ("path", "sha256")}, root)
    fields(bundle, "schema files", "kernel bundle")
    require(bundle["schema"] == "glm-replay-python-kernels-v1", "wrong kernel bundle schema")
    verify_files(Path(root), bundle["files"])
    require(isinstance(value["loaded"], list) and value["loaded"], "missing loaded kernel observations")
    observed = []
    for item in value["loaded"]:
        fields(item, "artifact symbols launch_count", "loaded kernel")
        require(item["artifact"] in bundle["files"], "loaded kernel outside pinned bundle")
        observed.append(item["artifact"]["path"])
        require(isinstance(item["symbols"], list) and item["symbols"], "missing launched kernel symbols")
        for symbol in item["symbols"]:
            string(symbol, "launched kernel symbol")
        integer(item["launch_count"], "kernel launch count", 1)
    require(len(observed) == len(set(observed)) and set(observed) == {a["path"] for a in bundle["files"]},
            "loaded kernel observations do not cover exact bundle")
