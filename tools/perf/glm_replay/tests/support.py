"""Invented host receipts with known wall times; never hardware measurements."""
import copy

from tools.perf.glm_replay.common import digest
from tools.perf.glm_replay.fixture import NATIVE_BASE, PYTHON_BASE
from tools.perf.glm_replay.results import PHASES, RESULT_SCHEMA


def receipt(fixture, arm="A", pair=0, block=0, rank=0, wall=1_000_000):
    h = fixture["config"]["pool"]["h"]
    cfg = {"engine": "python" if arm == "P" else "native", "speed": "standard", "tp": 2, "slots": 4,
           "dense_bits": 4 if arm == "P" else 8, "head_bits": 4 if arm == "P" else 8,
           "kv": "fp8", "kv_shard": arm == "B", "multi_lone": 0, "pool_h": h,
           "page_argument": 2 * h + (4 if arm == "B" else 0), "page_tokens": 512,
           "configured_capacity": fixture["config"]["pool"]["configured_capacity"],
           "dsa_layers": 11, "local_heads": 32, "latent_width": 512, "row_dtype_bytes": 2,
           "query_bytes_per_row": 32768, "output_bytes_per_row": 32768,
           "graph_rows": [1, 4, 8, 16, 32] if arm == "P" else [], "runtime_sha256": "9" * 64}
    source = {"engine_commit": PYTHON_BASE if arm == "P" else NATIVE_BASE, "harness_commit": "1" * 40,
              "binary_sha256": "2" * 64, "fatbin_sha256": "3" * 64, "compiler_sha256": "4" * 64,
              "kit_commit": "bf5f94523a8251029cedae86bf61f8ce37fac218" if arm == "P"
              else "13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7",
              "image_digest": "sha256:" + "5" * 64, "device_image_digest": "sha256:" + "6" * 64}
    forwards = []
    for scheduled in fixture["forwards"]:
        offered = sum(p["offered"] for p in scheduled["parts"])
        kept = sum(p["kept"] for p in scheduled["parts"])
        own = sum(p["offered"] for p in scheduled["parts"] if fixture["streams"][p["stream_id"]]["owner"] == rank)
        d = 11 if arm == "B" else 0
        forwards.append({**copy.deepcopy(scheduled), "phases_ns": {phase: 100 for phase in PHASES.split()},
                         "owner": {"layer_visits": d, "rows": d * offered,
                                   "query_send_bytes": d * (offered - own) * 32768,
                                   "output_send_bytes": d * own * 32768},
                         "graph_replays": int(arm == "P"), "eager_fallbacks": 0,
                         "target_commits": kept, "context_commits": kept, "ingest_calls": 1,
                         "proposer_calls": 0, "lost": 0})
    key, latent = 11 * 512 // 4 * 128 * 2, 11 * 512 * 516
    rows = 2 * h * key + (h + 1) * latent if arm == "B" else 2 * h * (key + latent)
    return {"schema": RESULT_SCHEMA, **{k: fixture[k] for k in
            ("fixture_sha256", "schedule_sha256", "tokens_sha256", "config_sha256")},
            "run_id": "synthetic-host-test", "pair_id": f"pair-{pair}", "start_id": f"{arm}-start-{pair}",
            "start_order": (pair + int(arm == "B")) % 2, "block_id": f"block-{block}", "cell_id": "test-cell",
            "arm": arm, "rank": rank, "status": "complete", "exclusion_reason": None,
            "source": source, "effective_config": cfg, "effective_config_sha256": digest(cfg),
            "preparation_ns": 500_000, "wall_ns": wall, "completed_ranks": [0, 1], "forwards": forwards,
            "checks": {"correctness": True, "timing_neutral": True, "default_server_neutral": True,
                       "fresh_prefill": True, "proposer_calls": 0, "lost": 0, "carries": 0, "twins": 0},
            "memory": {"logical_pages": 2 * h, "key_page_bytes": key, "latent_page_bytes": latent,
                       "row_page_bytes": rows, "tables_bytes": 0, "scratch_bytes": 0, "other_bytes": 0,
                       "peak_physical_bytes": rows}, "telemetry_sha256": "7" * 64}


def campaign(fixture, ratios=(1.1, 1.2, 1.3), blocks=5):
    return [receipt(fixture, arm, pair, block, rank,
                    round((1_000_000 + pair * 2_000_000 + block * 1000) * (ratio if arm == "B" else 1)))
            for pair, ratio in enumerate(ratios) for block in range(blocks) for arm in ("A", "B") for rank in (0, 1)]
