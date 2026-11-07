"""Untimed logical-state checks against the production one-row teacher-forced path."""
import hashlib

from .common import digest, require
from .python_runtime import exchange


def oracle_schedule(schedule):
    out = []
    for forward in schedule:
        for part in forward["parts"]:
            for row in range(part["kept"]):
                p = {**part, "position": part["position"] + row, "tape_offset": part["tape_offset"] + row,
                     "offered": 1, "kept": 1}
                out.append({"forward_id": len(out), "round_id": len(out), "parts": [p],
                            "oracle_part_end": row + 1 == part["kept"]})
    return out


def tensor_hash(tensor, torch):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def logical_state(runtime, sid):
    torch, w = runtime.torch, runtime.w
    st, ctx = runtime.states[sid], runtime.drafts.contexts[sid]
    pos = st.pos
    require(int(st.pos_dev.item()) == pos == ctx.context_end == int(ctx.pos_dev.item()),
            "host/device target/context positions diverged")
    live_from = max(0, pos - min(2048, ctx.ring or ctx.cap))
    ring_rows = torch.arange(live_from, pos, dtype=torch.int64, device=w.device)
    if ctx.ring:
        ring_rows = ring_rows % ctx.ring
    tail_from = pos // 4 * 4
    state = {"position": pos, "device_position": int(st.pos_dev.item()), "ring_from": live_from,
             "index_from": tail_from, "conv": tensor_hash(st.conv, torch),
             "recurrent": tensor_hash(st.rec[st.parity], torch),
             "latent_codes": [tensor_hash(t[:pos, :w.cfg.kv_lora], torch) for t in st.kc],
             "latent_scales": [tensor_hash(t[:pos, w.cfg.kv_lora:w.cfg.kv_lora + 4], torch) for t in st.kc],
             "index_keys": [], "index_gates": [], "pooled_keys": [],
             "ring_k": [tensor_hash(t.index_select(1, ring_rows), torch) for t in ctx.kc],
             "ring_v": [tensor_hash(t.index_select(1, ring_rows), torch) for t in ctx.vc]}
    require(st.index is not None, "production long-context index missing")
    for ik, ig, pk in st.index:
        logical = torch.arange(tail_from, pos, dtype=torch.int64, device=w.device)
        if st.caches.ring is not None:
            logical = logical % st.caches.ring
        state["index_keys"].append(tensor_hash(ik.index_select(0, logical), torch))
        state["index_gates"].append(tensor_hash(ig.index_select(0, logical), torch))
        state["pooled_keys"].append(tensor_hash(pk[:pos // 4, :w.cfg.index_dim + 4], torch))
    return state


def capture_forward(runtime, frame, slots=range(4)):
    torch = runtime.torch
    kept = {}
    for i, part in enumerate(frame.scheduled["parts"]):
        sid, keep = part["stream_id"], part["kept"]
        logits, taps = frame.verified.logits[i], frame.verified.taps[i]
        kept[str(sid)] = {"logits": [tensor_hash(logits[j], torch) for j in range(keep)],
                     "taps": [tensor_hash(taps[j], torch) for j in range(keep)],
                     "selected": frame.sampled[i][:keep]}
    return {"states": {str(sid): logical_state(runtime, sid) for sid in slots}, "kept": kept}


def compare_oracle(fixture, batched, oracle, initial_states=None):
    at = 0
    states = dict(initial_states or {})
    require(len(batched) == len(fixture["forwards"]), "incomplete batched validation capture")
    for scheduled, actual in zip(fixture["forwards"], batched):
        expected = {}
        for part in scheduled["parts"]:
            sid, count = str(part["stream_id"]), part["kept"]
            expected[sid] = {"logits": [], "taps": [], "selected": []}
            for _ in range(count):
                require(at < len(oracle), "oracle ended before all committed rows")
                row = oracle[at]
                require(set(row["kept"]) == {sid}, "oracle did not use true solo one-row forwards")
                for key in expected[sid]:
                    require(len(row["kept"][sid][key]) == 1, "oracle offered multiple rows")
                    expected[sid][key].extend(row["kept"][sid][key])
                at += 1
            require(sid in oracle[at - 1]["states"], "oracle part-end full state missing")
            states[sid] = oracle[at - 1]["states"][sid]
        require(actual["kept"] == expected, f"one-row logits/taps/selection mismatch at {scheduled['forward_id']}")
        require(actual["states"] == states,
                f"one-row logical cache/ring mismatch at {scheduled['forward_id']}")
    require(at == len(oracle), "oracle contains extra/stale rows")


def validate_device(runtime, binding):
    baseline = runtime.run(enabled=False, capture=True, poison=True)
    repeated = runtime.run(enabled=False, capture=True, poison=True)
    timed = runtime.run(enabled=True, capture=True, poison=True)
    oracle = runtime.run(enabled=False, capture=True, oracle=True)
    require(baseline["initial_states"] == oracle["initial_states"], "fresh preparation differs from one-row oracle")
    compare_oracle(runtime.fixture, baseline["snapshots"], oracle["snapshots"], oracle["initial_states"])
    require(baseline["snapshots"] == repeated["snapshots"], "fresh reset/repeated-run state mismatch")
    require(baseline["snapshots"] == timed["snapshots"], "timing changed outputs or logical state")
    def dispatch(run):
        return [(f.graph_replays, f.eager_fallbacks, f.target_commits, f.context_commits, f.ingest_calls)
                for f in run["frames"]]
    require(dispatch(baseline) == dispatch(repeated) == dispatch(timed), "timing changed dispatch/context lifecycle")
    for poison in (True, False):
        whole = runtime.run(enabled=False, poison=poison, final_capture=True)
        require(whole["kept_outputs"] == baseline["kept_outputs"], "unsynchronized schedule changed kept outputs")
        require(whole["final"] == baseline["snapshots"][-1], "unsynchronized schedule changed final state/taps/logits")
        require(dispatch(whole) == dispatch(baseline), "unsynchronized schedule changed dispatch")
    evidence = {"initial_states": baseline["initial_states"], "validated_forwards": baseline["snapshots"],
                "oracle_rows": oracle["snapshots"], "expected_kept_outputs": baseline["kept_outputs"],
                "expected_final": baseline["snapshots"][-1], "dispatch": dispatch(baseline),
                "poisoned_lookahead": True, "unsynchronized_poisoned_and_natural": True}
    rank_receipt = {"rank": runtime.rank, "status": "pass", "snapshot_sha256": digest(baseline["snapshots"]),
                    "oracle_sha256": digest(oracle["snapshots"]), "dispatch_sha256": digest(dispatch(baseline)),
                    "runtime_rank_sha256": runtime.runtime_ranks[runtime.rank],
                    "evidence_sha256": digest(evidence), "evidence": evidence}
    ranks = exchange(runtime.nccl.store, f"validation/{runtime.block_number}", runtime.rank, rank_receipt)
    require([r["rank"] for r in ranks] == [0, 1] and all(r["status"] == "pass" for r in ranks),
            "device validation missing peer completion")
    logical = {"schema": "glm-replay-python-logical-evidence-v1", "bindings": binding, "ranks": ranks}
    return {"schema": "glm-replay-python-validation-v1", "status": "pass", "execution": "device",
            "bindings": binding, "completed_ranks": [0, 1], "ranks": ranks,
            "logical_evidence_sha256": digest(logical), "logical_evidence": logical,
            "checks": {"correctness": True, "timing_neutral": True, "fresh_prefill": True,
                       "default_server_neutral": "unrun", "hardware_qualification": "unrun"},
            "coverage": {"fixture_only": True, "full": all(p["kept"] == p["offered"]
                         for f in runtime.fixture["forwards"] for p in f["parts"]),
                         "partial": any(p["kept"] < p["offered"]
                         for f in runtime.fixture["forwards"] for p in f["parts"]), "repeat_reset": True},
            "performance": "unrun; validation reads occur outside individually completed forwards"}
