"""Strict offline rank receipts, shared-forward accounting and paired-work validation."""
from collections import defaultdict

from .common import digest, fields, hex_id, integer, number, parse_json, require, string
from .fixture import NATIVE_BASE, PYTHON_BASE, validate_fixture

RESULT_SCHEMA = "glm-replay-result-v1"
PHASES = "staging target tap_bookkeeping selection target_commit context_commit"
SOURCE_FIELDS = ("engine_commit harness_commit binary_sha256 fatbin_sha256 compiler_sha256 "
                 "kit_commit image_digest device_image_digest")
CONFIG_FIELDS = ("engine speed tp slots dense_bits head_bits kv kv_shard multi_lone pool_h page_argument "
                 "page_tokens configured_capacity dsa_layers local_heads latent_width row_dtype_bytes "
                 "query_bytes_per_row output_bytes_per_row graph_rows")
RECORD_FIELDS = ("schema fixture_sha256 schedule_sha256 tokens_sha256 config_sha256 run_id pair_id start_id "
                 "start_order block_id cell_id arm rank status exclusion_reason source effective_config "
                 "effective_config_sha256 preparation_ns wall_ns completed_ranks forwards checks memory telemetry_sha256")
FORWARD_FIELDS = ("forward_id round_id parts phases_ns owner graph_replays eager_fallbacks target_commits "
                  "context_commits ingest_calls proposer_calls lost")


def load_results(paths):
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                require(bool(line.strip()), f"{path}:{line_number}: blank JSONL record")
                try:
                    records.append(parse_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    require(records, "empty result set")
    return records


def _source(record):
    source, arm = record["source"], record["arm"]
    fields(source, SOURCE_FIELDS, "source")
    for key in ("engine_commit", "harness_commit", "kit_commit"):
        hex_id(source[key], key, 40)
    require(source["engine_commit"] == (PYTHON_BASE if arm == "P" else NATIVE_BASE), "engine base mismatch")
    expected_kit = ("bf5f94523a8251029cedae86bf61f8ce37fac218" if arm == "P" else
                    "13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7")
    require(source["kit_commit"] == expected_kit, "kit revision mismatch")
    for key in ("binary_sha256", "fatbin_sha256", "compiler_sha256"):
        hex_id(source[key], key)
    for key in ("image_digest", "device_image_digest"):
        require(isinstance(source[key], str) and source[key].startswith("sha256:"), f"invalid {key}")
        hex_id(source[key][7:], key)


def _config(record, fixture):
    cfg, base, arm = record["effective_config"], fixture["config"], record["arm"]
    fields(cfg, CONFIG_FIELDS, "effective config")
    require(record["effective_config_sha256"] == digest(cfg), "effective config hash mismatch")
    for key in ("speed", "tp", "slots", "kv"):
        require(type(cfg[key]) is type(base[key]) and cfg[key] == base[key], f"effective {key} mismatch")
    require(cfg["engine"] == ("python" if arm == "P" else "native"), "arm/engine mismatch")
    require(type(cfg["kv_shard"]) is bool and cfg["kv_shard"] == (arm == "B"), "arm/sharding mismatch")
    integer(cfg["dense_bits"], "dense bits")
    integer(cfg["head_bits"], "head bits")
    require(cfg["dense_bits"] == (4 if arm == "P" else 8), "arm/dense precision mismatch")
    require(cfg["head_bits"] in ((4, 8, 16) if arm == "P" else (8,)), "arm/head precision mismatch")
    require(integer(cfg["multi_lone"], "multi_lone") == 0, "TF_GLM_MULTI_LONE must be zero")
    for key, name in (("pool_h", "h"), ("page_tokens", "page_tokens"), ("configured_capacity", "configured_capacity")):
        require(integer(cfg[key], key, 1) == base["pool"][name], f"effective {key} mismatch")
    require(integer(cfg["page_argument"], "page argument", 1) == 2 * cfg["pool_h"] + (4 if arm == "B" else 0),
            "unequal logical page capacity")
    for key, expected in (("dsa_layers", 11), ("local_heads", 32), ("latent_width", 512), ("row_dtype_bytes", 2)):
        require(integer(cfg[key], key, 1) == expected, f"pinned GLM geometry mismatch: {key}")
    row_bytes = cfg["local_heads"] * cfg["latent_width"] * cfg["row_dtype_bytes"]
    for key in ("query_bytes_per_row", "output_bytes_per_row"):
        require(integer(cfg[key], key, 1) == row_bytes, f"owner row geometry mismatch: {key}")
    require(isinstance(cfg["graph_rows"], list), "graph_rows must be an array")
    for rows in cfg["graph_rows"]:
        integer(rows, "graph row key", 1, 32)
    require(len(set(cfg["graph_rows"])) == len(cfg["graph_rows"]), "duplicate graph key")
    if arm == "P":
        needed = {sum(p["offered"] for p in f["parts"]) for f in fixture["forwards"]}
        require(needed <= set(cfg["graph_rows"]), "missing expected Python graph")
    else:
        require(cfg["graph_rows"] == [], "native eager path must not claim Python graphs")


def _memory(record):
    memory, cfg = record["memory"], record["effective_config"]
    fields(memory, "logical_pages key_page_bytes latent_page_bytes row_page_bytes tables_bytes scratch_bytes "
           "other_bytes peak_physical_bytes", "memory")
    for key, value in memory.items():
        if key not in ("tables_bytes", "scratch_bytes", "other_bytes", "peak_physical_bytes") or value is not None:
            integer(value, f"memory.{key}")
    require(memory["logical_pages"] == 2 * cfg["pool_h"], "logical capacity accounting mismatch")
    if record["arm"] != "P":
        require(memory["key_page_bytes"] == cfg["dsa_layers"] * cfg["page_tokens"] // 4 * 128 * 2,
                "key page geometry mismatch")
        require(memory["latent_page_bytes"] == cfg["dsa_layers"] * cfg["page_tokens"] * (cfg["latent_width"] + 4),
                "latent page geometry mismatch")
        h, key, lat = cfg["pool_h"], memory["key_page_bytes"], memory["latent_page_bytes"]
        expected = 2 * h * key + (h + 1) * lat if cfg["kv_shard"] else 2 * h * (key + lat)
        require(memory["row_page_bytes"] == expected, "physical row-page byte accounting mismatch")
    if memory["peak_physical_bytes"] is not None:
        known = sum(memory[k] or 0 for k in ("tables_bytes", "scratch_bytes", "other_bytes", "row_page_bytes"))
        require(memory["peak_physical_bytes"] >= known, "physical footprint below recorded allocations")


def _forward(record, actual, scheduled, fixture):
    fields(actual, FORWARD_FIELDS, "result forward")
    for key in ("forward_id", "round_id"):
        integer(actual[key], key)
        require(actual[key] == scheduled[key], "result forward/round ID mismatch")
    require(digest(actual["parts"]) == digest(scheduled["parts"]), "result parts differ from exact schedule")
    fields(actual["phases_ns"], PHASES, "phase timings")
    for key, value in actual["phases_ns"].items():
        number(value, key)
    kept = sum(part["kept"] for part in scheduled["parts"])
    for key, value in {"target_commits": kept, "context_commits": kept, "ingest_calls": 1,
                       "proposer_calls": 0, "lost": 0, "eager_fallbacks": 0,
                       "graph_replays": int(record["arm"] == "P")}.items():
        require(integer(actual[key], key) == value, f"incorrect per-forward {key}")
    owner, cfg = actual["owner"], record["effective_config"]
    fields(owner, "layer_visits rows query_send_bytes output_send_bytes", "owner counters")
    total = sum(part["offered"] for part in scheduled["parts"])
    own = sum(part["offered"] for part in scheduled["parts"]
              if fixture["streams"][part["stream_id"]]["owner"] == record["rank"])
    d = cfg["dsa_layers"] if cfg["kv_shard"] else 0
    expected = {"layer_visits": d, "rows": d * total,
                "query_send_bytes": d * (total - own) * cfg["query_bytes_per_row"],
                "output_send_bytes": d * own * cfg["output_bytes_per_row"]}
    for key, value in expected.items():
        require(integer(owner[key], key) == value, f"owner {key} mismatch at forward {scheduled['forward_id']}")


def _record(record, fixtures):
    fields(record, RECORD_FIELDS, "result record")
    require(record["schema"] == RESULT_SCHEMA, "unsupported result schema")
    require(record["fixture_sha256"] in fixtures, "unknown fixture hash")
    fixture = fixtures[record["fixture_sha256"]]
    for key in ("fixture_sha256", "schedule_sha256", "tokens_sha256", "config_sha256"):
        require(record[key] == fixture[key], f"result {key} mismatch")
    for key in ("run_id", "pair_id", "start_id", "block_id", "cell_id"):
        string(record[key], key)
    integer(record["rank"], "rank", 0, 1)
    integer(record["start_order"], "start_order", 0, 1)
    require(record["arm"] in ("A", "B", "P"), "invalid arm")
    require(record["status"] in ("complete", "excluded"), "invalid result status")
    complete = record["status"] == "complete"
    if complete:
        require(record["exclusion_reason"] is None, "complete receipt has exclusion reason")
    else:
        string(record["exclusion_reason"], "exclusion_reason")
    require(record["completed_ranks"] == [0, 1] and all(type(r) is int for r in record["completed_ranks"]),
            "missing both-rank completion receipt")
    integer(record["preparation_ns"], "preparation_ns")
    integer(record["wall_ns"], "wall_ns", 1)
    hex_id(record["telemetry_sha256"], "telemetry_sha256")
    _source(record)
    _config(record, fixture)
    _memory(record)
    checks = record["checks"]
    fields(checks, "correctness timing_neutral default_server_neutral fresh_prefill proposer_calls lost carries twins", "checks")
    for key in ("correctness", "timing_neutral", "default_server_neutral", "fresh_prefill"):
        require(type(checks[key]) is bool and (checks[key] or not complete), f"failed or unrun {key}")
    for key in ("proposer_calls", "lost", "carries", "twins"):
        integer(checks[key], key)
        require(not complete or checks[key] == 0, f"nonzero {key} in complete block")
    require(isinstance(record["forwards"], list), "result forwards must be an array")
    count, expected = len(record["forwards"]), len(fixture["forwards"])
    require(count == expected if complete else count <= expected, "incomplete or extra result forwards")
    for index, forward in enumerate(record["forwards"]):
        _forward(record, forward, fixture["forwards"][index], fixture)


def block_key(record):
    return tuple(record[k] for k in ("run_id", "pair_id", "cell_id", "block_id", "arm"))


def rank_blocks(records):
    groups = defaultdict(list)
    for record in records:
        groups[block_key(record)].append(record)
    return groups


def _rank_agreement(group):
    require(len(group) == 2 and {r["rank"] for r in group} == {0, 1}, "incomplete or duplicate rank block")
    left, right = sorted(group, key=lambda r: r["rank"])
    keys = ("fixture_sha256 schedule_sha256 tokens_sha256 config_sha256 run_id pair_id start_id start_order "
            "block_id cell_id arm status exclusion_reason source effective_config effective_config_sha256 checks")
    for key in keys.split():
        require(digest(left[key]) == digest(right[key]), f"rank disagreement: {key}")
    require(len(left["forwards"]) == len(right["forwards"]), "rank schedule completion disagreement")
    return left


def _pair_agreement(left, right):
    require(left["start_id"] != right["start_id"], "native arms reused process start")
    require(left["start_order"] != right["start_order"], "native paired order mismatch")
    for key in ("fixture_sha256", "schedule_sha256", "tokens_sha256", "config_sha256", "source"):
        require(digest(left[key]) == digest(right[key]), f"unmatched native work/build: {key}")
    configs = [{k: v for k, v in r["effective_config"].items() if k not in ("kv_shard", "page_argument")}
               for r in (left, right)]
    require(digest(configs[0]) == digest(configs[1]), "native arms changed more than sharding/page argument")


def validate_results(records, fixtures, allow_synthetic=False):
    require(isinstance(records, list) and records, "empty result set")
    for key, fixture in fixtures.items():
        validate_fixture(fixture, allow_synthetic=allow_synthetic)
        require(key == fixture["fixture_sha256"], "fixture map hash mismatch")
    seen, starts, cells, pairs = set(), {}, {}, defaultdict(dict)
    native_builds = {}
    for record in records:
        _record(record, fixtures)
        if record["arm"] != "P":
            build = digest(record["source"])
            require(record["run_id"] not in native_builds or native_builds[record["run_id"]] == build,
                    "native build changed across paired starts")
            native_builds[record["run_id"]] = build
        identity = tuple(record[k] for k in ("run_id", "start_id", "cell_id", "block_id", "rank"))
        require(identity not in seen, "duplicate rank/block receipt")
        seen.add(identity)
        start = (record["run_id"], record["start_id"])
        definition = (record["pair_id"], record["arm"], record["start_order"], digest(record["source"]),
                      record["effective_config_sha256"])
        require(start not in starts or starts[start] == definition, "inconsistent process-start definition")
        starts[start] = definition
        cell = (record["run_id"], record["cell_id"])
        require(cell not in cells or cells[cell] == record["fixture_sha256"], "cell changed fixture")
        cells[cell] = record["fixture_sha256"]
    for key, group in rank_blocks(records).items():
        rank0 = _rank_agreement(group)
        if rank0["arm"] != "P":
            pair_key = key[:4]
            pairs[pair_key][rank0["arm"]] = rank0
    start_pairs, orders = {}, defaultdict(dict)
    for key, arms in pairs.items():
        require(set(arms) == {"A", "B"}, "unmatched native A/B block")
        _pair_agreement(arms["A"], arms["B"])
        pair = key[:2]
        definition = tuple((arms[a]["start_id"], arms[a]["start_order"]) for a in ("A", "B"))
        require(pair not in start_pairs or start_pairs[pair] == definition, "pair reused different native starts")
        start_pairs[pair] = definition
        orders[key[0]][key[1]] = arms["A"]["start_order"]
    for run_orders in orders.values():
        counts = list(run_orders.values())
        require(abs(counts.count(0) - counts.count(1)) <= 1, "native start order is not counterbalanced")
    return records
