"""Symmetric ranks using production bootstrap, transport, loaders, graphs and cache pools."""
import dataclasses
import importlib
import math
import secrets
import sys
import time
from datetime import timedelta
from pathlib import Path

from .capacity import reservations
from .common import canonical_bytes, digest, file_digest, hex_id, read_json, require
from .python_control import Controller, ProposerGuard, QuietTrace, frames
from .python_provenance import environment, graph_rows
from .python_startup import capture_bank, launch_settings, open_transport

PREFIX = "tensorfold.families.glm5_next.cuda."


def module(name):
    return importlib.import_module(PREFIX + name)


def exchange(store, label, rank, value):
    from .common import parse_json

    key = f"glm-replay/{label}"
    store.set(f"{key}/{rank}", canonical_bytes(value))
    store.wait([f"{key}/{r}" for r in (0, 1)], timedelta(seconds=120))
    return [parse_json(store.get(f"{key}/{r}").decode("utf-8")) for r in (0, 1)]


def completed(payloads, expected):
    require(len(expected) == 4 and all(type(n) is int and n >= 0 for n in expected)
            and expected[0] > 0 and expected[1] > 0 and expected[2:] == [0, 0],
            "invalid empty/proposer/lost completion audit")
    require(len(payloads) == 2, "missing rank completion")
    for rank, got in enumerate(payloads):
        require(got == [rank, *expected], "rank completion/count disagreement")
    return [0, 1]


def session_provenance(store, rank, meta, campaign_sha256=None):
    nonces = exchange(store, "session-nonce", rank, secrets.token_hex(32))
    require(isinstance(nonces, list) and len(nonces) == 2, "missing rank session nonces")
    for nonce in nonces:
        hex_id(nonce, "session nonce")
    campaign = campaign_sha256 or digest(meta)
    hex_id(campaign, "campaign digest")
    return {"session_nonce": digest(nonces), "campaign_sha256": campaign,
            "checkpoints": {"scheme": "file-manifest-v1",
                **{role: [digest(files) for files in meta["checkpoints"][role]["ranks"]]
                   for role in ("target", "drafter")}}}


class EventTrace:
    """Preallocated rank-local CUDA intervals, resolved only after both ranks complete."""

    def __init__(self, torch, count, enabled=True):
        from .python_control import PHASES

        self.enabled = enabled
        self.events = [{p: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        for p in PHASES} for _ in range(count)]
        for row in self.events:
            for start, end in row.values():
                start.record()
                end.record()
        torch.cuda.synchronize()

    def begin(self, index, phase):
        if self.enabled:
            self.events[index][phase][0].record()

    def end(self, index, phase):
        if self.enabled:
            self.events[index][phase][1].record()

    def resolve(self):
        require(self.enabled, "disabled events have no measured phase durations")
        return [{p: round(a.elapsed_time(b) * 1_000_000) for p, (a, b) in row.items()} for row in self.events]


class Runtime:
    def __init__(self, fixture, meta, roots, rank, master, port, preflight, fixtures=None, campaign_identity=None):
        import torch

        from tensorfold.cuda.comm import NCCL

        self.torch, self.fixture, self.meta, self.rank = torch, fixture, meta, rank
        self.fixtures = fixtures or [fixture]
        torch.cuda.set_device(0)
        self.nccl = NCCL(rank, 2, master, port)
        self.comm = self.nccl
        provenance = session_provenance(self.nccl.store, rank, meta, campaign_identity)
        self.session_nonce, self.campaign_sha256 = provenance["session_nonce"], provenance["campaign_sha256"]
        self.checkpoints = provenance["checkpoints"]
        identity = {"bindings": preflight["bindings"], "sources": preflight["source_tree_sha256"],
                    "metadata": digest(meta), "campaign": campaign_identity,
                    "execution_environment": {k: v for k, v in environment().items()
                    if k not in {"NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "TF_ROCE_HCAS", "TF_NCCL_LIB",
                                 "CUDA_VISIBLE_DEVICES", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"}}}
        agreed = exchange(self.nccl.store, "preload", rank, identity)
        require(agreed[0] == agreed[1], "ranks disagree before engine load")
        self.comm = open_transport(self.nccl, rank, environment().get("TF_GLM_COMM", "nccl"))
        production = module("engine")
        self.startup = launch_settings(production, module("display_kv"), meta["launch"])
        agreed = exchange(self.nccl.store, "startup", rank, self.startup)
        require(agreed[0] == agreed[1], "ranks disagree on derived startup settings")
        require(production.MAX_ROWS == 16 and production.DRAFT_RING is True
                and module("shard").ON is False, "recorded production window/ring/shard settings required")
        self.split = module("hcsplit").SplitSettings.from_env()
        if self.split.split and self.split.exchange == "p2p":
            a = torch.zeros((1,), dtype=torch.float32, device="cuda")
            self.comm.exchange([a], [torch.empty_like(a)], 1 - rank)
            torch.cuda.synchronize()
        weights = module("weights")
        cfg = weights.Config.read(Path(roots["target"]))
        require(cfg.quant == "mlx" and cfg.bits == 4, "contextual Python arm requires pinned affine q4 checkpoint")
        self.w = weights.load(Path(roots["target"]), rank=rank, world=2, mtp=False)
        self.w.comm = self.comm
        self.comm.ready("loading")
        self.comm.barrier()
        q4 = module("qmm").Q4
        projections = [self.w.head] + [l.kda.proj if l.kind == "kda" else l.dsa.proj for l in self.w.layers]
        require(all(isinstance(p, q4) for p in projections), "loaded dense/head q4 identity mismatch")
        require(self.w.mtp is None, "target MTP must not be loaded with four configured slots")
        needed_rows = sorted({row for case in self.fixtures for row in graph_rows(case)})
        pool = fixture["config"]["pool"]
        self.pool_rows = 2 * pool["h"] * pool["page_tokens"]
        require(self.pool_rows % module("pool").ALIGN == 0, "pool must tile production aligned extents")
        self.drafter = module("dflash2").Drafter(Path(roots["drafter"]), self.w,
                    capacity=self.pool_rows, tap_rows=16, ring=True)
        launch = self.startup
        self.e = module("decode").Engine(self.w, capacity=self.pool_rows, max_rows=production.MAX_ROWS,
                    prefill_rows=launch["prefill_rows"], graphs=True,
                    graph_rows=production.GRAPH_ROWS + production.WIDE_GRAPHS,
                    long_context=True, taps=self.drafter.tap_layers, grid=launch["grid"], split=self.split,
                    streams=4, pool_rows=self.pool_rows, kv="fp8", display=launch["display"])
        if self.e.pbuf.split is not None:
            self.e.pbuf.split.warm()
        self.drafter.capture()
        self.drafts = module("dflash2_multi").MultiDrafter(self.drafter, streams=4)
        self.drafts.capture()
        self.verifier = module("verify").BatchedVerify(self.e, taps=self.drafter.tap_layers, rows=32, segs=4)
        self.graph_rows = capture_bank(self.verifier, needed_rows)
        require(all(isinstance(g, torch.cuda.CUDAGraph) for g in self.verifier.graphs.values()),
                "target graph implementation was substituted")
        require(self.drafts.tap_graphs and self.drafts.block_graphs, "production drafter graph allocation missing")
        require(self.drafter.ring >= 2048 and len(self.drafts.contexts) == self.e.slots.count == 4,
                "four real production rings/slots required")
        self.guard = ProposerGuard([self.drafter, self.drafts, *self.drafts.contexts])
        self.controller = Controller(self.verifier, self.drafts, module("multi").sample_streams, self.w, self.guard)
        self.controller.wrap_graphs()
        self.layout = self.reserve_layout()
        if hasattr(self.comm, "settle"):
            self.comm.settle()
        self.finish_host = torch.zeros((5,), dtype=torch.int64, pin_memory=True)
        self.finish_send = torch.zeros_like(self.finish_host, device="cuda")
        self.finish_recv = torch.zeros((10,), dtype=torch.int64, device="cuda")
        self.block_number = 0
        for case in self.fixtures:
            self.fixture = case
            self.run(enabled=False)
        self.fixture = fixture
        self.runtime_ranks, self.runtime_sha256 = self.fingerprint(roots)

    def fingerprint(self, roots):
        torch = self.torch
        loaded = [{"name": k, "class": type(v).__module__ + "." + type(v).__qualname__}
                  for k, v in sorted(self.w.meta.items()) if v is not None]
        transport = {"class": type(self.comm).__module__ + "." + type(self.comm).__qualname__,
                     "roce_limit": getattr(getattr(self.comm, "roce", None), "max_bytes", None),
                     "roce_routes": getattr(getattr(self.comm, "roce", None), "routes", None),
                     "roce_hcas": getattr(getattr(self.comm, "roce", None), "hcas", None),
                     "settled": getattr(self.comm, "settled", None)}
        libraries = []
        extension_hashes = set()
        for name, mod in sorted(sys.modules.items()):
            path = Path(getattr(mod, "__file__", "") or "")
            if path.is_file() and (".so" in path.name or path.suffix == ".pyd"):
                sha = file_digest(path)
                libraries.append({"module": name, "sha256": sha})
                if name.startswith("tensorfold"):
                    extension_hashes.add(sha)
        process_libraries = []
        mapped = {line.rsplit(maxsplit=1)[-1] for line in Path("/proc/self/maps").read_text().splitlines()}
        for name in sorted(mapped):
            path = Path(name)
            if path.is_absolute() and ".so" in path.name and path.is_file():
                sha = file_digest(path)
                process_libraries.append({"name": path.name, "sha256": sha})
                if "tensorfold" in path.name:
                    extension_hashes.add(sha)
        bundle = read_json(Path(roots["artifacts"]) / self.meta["artifacts"]["fatbin"]["path"])
        require(extension_hashes <= {f["sha256"] for f in bundle["files"]},
                "actual loaded engine extension missing from pinned kernel bundle")
        flags = {}
        for name, mod in sorted(sys.modules.items()):
            if name.startswith("tensorfold"):
                flags[name] = {k: v for k, v in vars(mod).items() if k.isupper()
                               and (v is None or type(v) in (str, bool, int)
                                    or type(v) is float and math.isfinite(v))}
        actual = {"environment": environment(), "launch": self.meta["launch"], "startup": self.startup, "loaded": loaded,
                  "transport": transport, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(), "capability": torch.cuda.get_device_capability(),
                  "split": dataclasses.asdict(self.split), "pool_rows": self.e.caches.rows,
                  "slots": self.e.slots.count, "kv": self.e.caches.kv, "q4": True,
                  "tap_layers": self.drafter.tap_layers, "ring": self.drafter.ring,
                  "multi": dataclasses.asdict(module("multi_tune").MultiSettings.from_env()),
                  "source_tree_sha256": self.meta["source_tree_sha256"],
                  "artifacts": self.meta["artifacts"], "checkpoints": self.meta["checkpoints"],
                  "graph_rows": self.graph_rows, "pool_layout": self.layout, "module_flags": flags,
                  "native_modules": libraries, "process_libraries": process_libraries,
                  "weight_config": dataclasses.asdict(self.w.cfg),
                  "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                  "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                  "deterministic": torch.are_deterministic_algorithms_enabled(),
                  "matmul_precision": torch.get_float32_matmul_precision()}
        ranks = exchange(self.nccl.store, "runtime", self.rank, digest(actual))
        return ranks, digest(ranks)

    def reserve_layout(self):
        pool = module("pool").Pool(self.pool_rows)
        needs = {sid: max(reservations(case, 512, 1)["stream_extents"][sid] for case in self.fixtures)
                 for sid in range(4)}
        layout = []
        for sid in range(4):
            extent = pool.alloc(needs[sid])
            require(extent is not None, "aligned four-slot extents exceed real production pool")
            extent.owner = sid
            layout.append({"stream_id": sid, "base": extent.base, "size": extent.size})
        return layout

    def select_case(self, fixture, meta):
        require(self.controller.current is None and not self.controller.broken, "cannot switch an active/failed case")
        require(fixture["fixture_sha256"] in {f["fixture_sha256"] for f in self.fixtures},
                "case was not prevalidated before session startup")
        for key in ("source", "source_tree_sha256", "artifacts", "checkpoints", "launch"):
            require(meta[key] == self.meta[key], f"campaign changed shared session identity: {key}")
        require(fixture["config"]["pool"] == self.fixture["config"]["pool"], "campaign changed fixed pool/capacity")
        self.fixture, self.meta = fixture, meta

    def fresh(self):
        torch, e = self.torch, self.e
        torch.cuda.synchronize()
        for plane in e.caches.arena.planes:
            plane.tensor.zero_()
        for pair in e.caches.rings or []:
            for tensor in pair:
                tensor.zero_()
        for tensor in (e.slots.conv, e.slots.rec, e.slots.proj, e.slots.pos_dev, e.slots.mtp_pos_dev):
            tensor.zero_()
        for tensor in (*self.drafts.kc, *self.drafts.vc, self.drafts.pos):
            tensor.zero_()
        self.drafts._pre = None
        self.drafts.reset()
        self.pool = module("pool").Pool(self.pool_rows)
        self.states, self.extents = [], []
        for stream in self.fixture["streams"]:
            sid = stream["stream_id"]
            extent = self.pool.alloc(self.layout[sid]["size"])
            require(extent is not None and extent.base % 2048 == 0, "production pool cannot reserve all four streams")
            extent.owner = sid
            self.extents.append(extent)
            require(self.layout[sid] == {"stream_id": sid, "base": extent.base, "size": extent.size},
                    "fresh production pool layout changed")
            st = module("forward").State(self.w, extent.size, e.rows, caches=e.caches, base=extent.base,
                                         slots=e.slots, slot=sid)
            self.states.append(st)
            previous = e.use(st)
            try:
                module("decode").prefill(e, stream["prompt_tokens"], None, mtp=False,
                                         drafter=self.drafts.contexts[sid], sample=False)
            finally:
                e.use(previous)
            require(st.pos == self.drafts.contexts[sid].context_end == len(stream["prompt_tokens"]),
                    "fresh all-slot prefill incomplete")
        torch.cuda.synchronize()
        require(len({s.base for s in self.states}) == 4 and not any(x.twin for x in self.pool.extents),
                "fresh pool aliases or retains twins")

    def plan(self, schedule, poison=False):
        return frames(self.fixture, schedule, self.states, module("verify").Segment, poison)

    def finish(self, plan, start_ns):
        torch = self.torch
        expected = [len(plan), sum(f.target_commits for f in plan), self.guard.calls, sum(f.lost for f in plan)]
        require(expected[0] > 0 and all(f.target_commits == f.context_commits > 0 for f in plan),
                "empty or incomplete replay block")
        torch.cuda.synchronize()
        self.finish_host.numpy()[:] = [self.rank, *expected]
        self.finish_send.copy_(self.finish_host, non_blocking=True)
        self.nccl.all_gather(self.finish_send, self.finish_recv)
        torch.cuda.synchronize()
        wall = time.perf_counter_ns() - start_ns
        completed(self.finish_recv.cpu().view(2, 5).tolist(), expected)
        if hasattr(self.comm, "check"):
            self.comm.check()
        return wall

    def prepare(self, oracle=False):
        begin = time.perf_counter_ns()
        self.controller.trace = QuietTrace()
        self.fresh()
        if oracle:
            from .python_validation import oracle_schedule

            plan = self.plan(oracle_schedule(self.fixture["preparation"]))
            verifier = module("verify").SerialVerify(self.e, taps=True)
            ctl = Controller(verifier, self.drafts, module("multi").sample_streams, self.w, self.guard)
        else:
            plan, ctl = self.plan(self.fixture["preparation"]), self.controller
        ctl.start()
        for i, frame in enumerate(plan):
            ctl.execute(frame, i, graph=not oracle)
        self.torch.cuda.synchronize()
        self.comm.barrier()
        return time.perf_counter_ns() - begin

    def run(self, enabled=False, capture=False, oracle=False, poison=False, final_capture=False):
        from .python_validation import capture_forward, logical_state, oracle_schedule

        prep_begin = time.perf_counter_ns()
        self.prepare(oracle)
        schedule = oracle_schedule(self.fixture["forwards"]) if oracle else self.fixture["forwards"]
        plan = self.plan(schedule, poison)
        initial = {str(s): logical_state(self, s) for s in range(4)} if capture else None
        trace = EventTrace(self.torch, len(plan), enabled=enabled) if not oracle else None
        ctl = self.controller
        if oracle:
            ctl = Controller(module("verify").SerialVerify(self.e, taps=True), self.drafts,
                             module("multi").sample_streams, self.w, self.guard)
        elif trace is not None:
            ctl.trace = trace
        self.comm.barrier()
        ctl.start()
        snapshots = []
        prep = time.perf_counter_ns() - prep_begin
        realtime_start = time.time_ns()
        begin = time.perf_counter_ns()
        wall = 0
        for index, frame in enumerate(plan):
            ctl.execute(frame, index, graph=not oracle)
            if capture:
                wall += self.finish([frame], begin)
                slots = range(4)
                if oracle:
                    part = frame.scheduled["parts"][0]
                    slots = [part["stream_id"]] if frame.scheduled["oracle_part_end"] else []
                snapshots.append(capture_forward(self, frame, slots))
                self.comm.barrier()
                begin = time.perf_counter_ns()
            if not final_capture or index + 1 < len(plan):
                frame.verified = None
        if not capture:
            wall = self.finish(plan, begin)
        realtime_end = time.time_ns()
        final = capture_forward(self, plan[-1], range(4)) if final_capture else None
        plan[-1].verified = None
        phases = trace.resolve() if trace is not None and enabled else None
        self.block_number += 1
        return {"preparation_ns": prep, "wall_ns": wall, "frames": plan, "phases": phases,
                "snapshots": snapshots, "initial_states": initial, "final": final,
                "poisoned_rows": sum(f.poisoned_rows for f in plan),
                "realtime_window": {"rank": self.rank, "realtime_start_ns": realtime_start,
                                    "realtime_end_ns": realtime_end},
                "kept_outputs": [{str(p["stream_id"]): f.sampled[i][:p["kept"]]
                                 for i, p in enumerate(f.scheduled["parts"])} for f in plan],
                "completed_ranks": [0, 1]}
