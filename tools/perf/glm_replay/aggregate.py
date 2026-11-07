"""Summarize blocks within starts before estimating independent native paired-start effects."""
import math
import statistics
from collections import defaultdict

from .common import integer, require
from .results import rank_blocks, validate_results

T95 = (0, 12.706205, 4.302653, 3.182446, 2.776445, 2.570582, 2.446912, 2.364624,
       2.306004, 2.262158, 2.228139, 2.200985, 2.178813, 2.160369, 2.144787, 2.131450,
       2.119905, 2.109816, 2.100922, 2.093024, 2.085963, 2.079614, 2.073873, 2.068658,
       2.063899, 2.059539, 2.055529, 2.051831, 2.048407, 2.045230, 2.042272)


def _interval(values, logarithmic=False):
    samples = [math.log(v) for v in values] if logarithmic else values
    count = len(samples)
    require(count >= 2, "interval requires independent process-start pairs")
    mean = statistics.mean(samples)
    critical = T95[min(count - 1, len(T95) - 1)]
    half_width = critical * statistics.stdev(samples) / math.sqrt(count)
    transform = math.exp if logarithmic else float
    return {"estimate": transform(mean), "ci95": [transform(mean - half_width), transform(mean + half_width)],
            "independent_pairs": count, "method": "paired-start Student-t on log ratios" if logarithmic
            else "paired-start Student-t on absolute differences"}


def _start_summary(blocks, tokens):
    require(len({b["start_id"] for b in blocks}) == 1, "different process starts pooled")
    mean_wall = statistics.mean(b["wall_ns"] for b in blocks)
    forward_count = len(blocks[0]["phases_ns_by_rank"][0])
    return {"source": blocks[0]["source"], "effective_config": blocks[0]["effective_config"],
            "start_id": blocks[0]["start_id"], "start_order": blocks[0]["start_order"],
            "block_count": len(blocks), "mean_wall_ns": mean_wall,
            "milliseconds_per_forward": mean_wall / 1e6 / forward_count,
            "forced_committed_tokens_per_second": tokens * 1e9 / mean_wall,
            "blocks": [{"block_id": b["block_id"], "wall_ns": b["wall_ns"],
                        "preparation_ns_by_rank": b["preparation_ns_by_rank"],
                        "phases_ns_by_rank": b["phases_ns_by_rank"],
                        "owner_counters_by_rank": b["owner_counters_by_rank"],
                        "memory_by_rank": b["memory_by_rank"], "telemetry_sha256_by_rank": b["telemetry_sha256_by_rank"]} for b in blocks]}


def _block(group):
    ranks = sorted(group, key=lambda r: r["rank"])
    first = ranks[0]
    return {"source": first["source"], "effective_config": first["effective_config"],
            "start_id": first["start_id"], "start_order": first["start_order"], "block_id": first["block_id"],
            "wall_ns": max(r["wall_ns"] for r in ranks),
            "preparation_ns_by_rank": [r["preparation_ns"] for r in ranks],
            "memory_by_rank": [r["memory"] for r in ranks],
            "telemetry_sha256_by_rank": [r["telemetry_sha256"] for r in ranks],
            "phases_ns_by_rank": [[f["phases_ns"] for f in r["forwards"]] for r in ranks],
            "owner_counters_by_rank": [[f["owner"] for f in r["forwards"]] for r in ranks]}


def summarize_paired(records, fixtures, minimum_pairs=3, minimum_blocks=5, allow_synthetic=False):
    integer(minimum_pairs, "minimum pairs", 3)
    integer(minimum_blocks, "minimum blocks", 5)
    validate_results(records, fixtures, allow_synthetic=allow_synthetic)
    grouped = rank_blocks(records)
    native, python, exclusions = defaultdict(lambda: defaultdict(dict)), defaultdict(list), []
    all_pairs = defaultdict(set)
    for key, ranks in grouped.items():
        run, pair, cell, block, arm = key
        if arm == "P":
            python[(run, cell, ranks[0]["start_id"])].append(ranks)
        else:
            native[(run, cell)][pair].setdefault(block, {})[arm] = ranks
            all_pairs[run].add(pair)
    cells = []
    for (run, cell), pairs in sorted(native.items()):
        require(set(pairs) == all_pairs[run], "cell missing independent native start pair")
        require(len(pairs) >= minimum_pairs, "insufficient independent native paired starts")
        first_group = next(iter(next(iter(pairs.values())).values()))["A"]
        fixture = fixtures[first_group[0]["fixture_sha256"]]
        tokens = sum(p["kept"] for f in fixture["forwards"] for p in f["parts"])
        starts = []
        for pair, blocks in sorted(pairs.items()):
            valid = {"A": [], "B": []}
            for block, arms in sorted(blocks.items()):
                if any(r["status"] != "complete" for group in arms.values() for r in group):
                    exclusions.append({"run_id": run, "pair_id": pair, "cell_id": cell, "block_id": block,
                                       "attempts": [{"arm": a, "rank": r["rank"], "status": r["status"],
                                                     "reason": r["exclusion_reason"]}
                                                    for a, group in arms.items() for r in group]})
                    continue
                for arm in ("A", "B"):
                    valid[arm].append(_block(arms[arm]))
            require(len(valid["A"]) >= minimum_blocks, "insufficient complete matched blocks within start pair")
            left, right = (_start_summary(valid[a], tokens) for a in ("A", "B"))
            starts.append({"pair_id": pair, "A": left, "B": right,
                           "B_over_A": right["mean_wall_ns"] / left["mean_wall_ns"],
                           "B_minus_A_ns": right["mean_wall_ns"] - left["mean_wall_ns"]})
        ratio = _interval([s["B_over_A"] for s in starts], logarithmic=True)
        difference = _interval([s["B_minus_A_ns"] for s in starts])
        cells.append({"run_id": run, "cell_id": cell, "fixture_sha256": fixture["fixture_sha256"],
                      "unique_forwards_per_block": len(fixture["forwards"]),
                      "rounds_per_block": len(fixture["forwards"]), "committed_tokens_per_block": tokens,
                      "target_commit_tokens_per_block": tokens, "context_commit_tokens_per_block": tokens,
                      "tokens_sha256": fixture["tokens_sha256"], "schedule_sha256": fixture["schedule_sha256"],
                      "config_sha256": fixture["config_sha256"],
                      "paired_starts": starts, "B_over_A": ratio, "B_minus_A_ns": difference,
                      "interpretation": "inconclusive" if ratio["ci95"][0] <= 1 <= ratio["ci95"][1]
                      else "resolved in these paired starts; small-sample uncertainty remains"})
    context, python_starts = [], defaultdict(set)
    for (run, cell, start), groups in sorted(python.items()):
        valid = []
        for ranks in groups:
            if ranks[0]["status"] == "complete":
                valid.append(_block(ranks))
            else:
                exclusions.append({"run_id": run, "cell_id": cell, "start_id": start,
                                   "block_id": ranks[0]["block_id"], "arm": "P",
                                   "reason": ranks[0]["exclusion_reason"]})
        require(len(valid) >= minimum_blocks, "insufficient Python blocks within start")
        fixture = fixtures[groups[0][0]["fixture_sha256"]]
        tokens = sum(p["kept"] for f in fixture["forwards"] for p in f["parts"])
        python_starts[(run, cell)].add(start)
        context.append({"run_id": run, "cell_id": cell, "arm": "P", **_start_summary(valid, tokens)})
    for starts in python_starts.values():
        require(len(starts) >= 2, "Python context requires at least two independent starts")
    return {"schema": "glm-replay-summary-v1", "metric": "synchronized replay-block wall time",
            "rate_label": "forced committed-token replay throughput", "synthetic_host_only": allow_synthetic,
            "independent_unit": "paired process starts; blocks, streams and rounds are not replicates",
            "interval_note": "Student-t intervals are assumption-dependent with only three paired starts",
            "cells": cells, "python_context": context, "excluded_attempts": exclusions}
