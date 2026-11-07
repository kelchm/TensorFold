"""Prescribed replay transactions over the existing verifier and context-ingest boundaries."""
from dataclasses import dataclass

from .common import require
from .fixture import offered_tokens

PHASES = ("staging", "target", "tap_bookkeeping", "selection", "target_commit", "context_commit")


class QuietTrace:
    def begin(self, index, phase):
        pass

    def end(self, index, phase):
        pass


class ProposerGuard:
    """Fail immediately if any production proposal boundary is reached."""

    def __init__(self, owners):
        self.calls = 0
        self.saved = []
        for owner in owners:
            for name in ("propose", "prelaunch", "candidates", "_launch_block"):
                if hasattr(owner, name):
                    original = getattr(owner, name)
                    self.saved.append((owner, name, original))
                    setattr(owner, name, self.refuse)

    def refuse(self, *args, **kwargs):
        self.calls += 1
        raise ValueError("draft proposer is forbidden in fixed-work replay")


@dataclass
class Frame:
    scheduled: dict
    segments: list
    keeps: list
    positions: list
    sampled: object = None
    verified: object = None
    graph_replays: int = 0
    eager_fallbacks: int = 0
    target_commits: int = 0
    context_commits: int = 0
    ingest_calls: int = 0
    lost: int = 0
    poisoned_rows: int = 0


def frames(fixture, schedule, states, segment_type, poison=False):
    out = []
    for forward in schedule:
        segments = [segment_type(states[p["stream_id"]], offered_tokens(fixture, p)) for p in forward["parts"]]
        poisoned = 0
        if poison:
            vocab = fixture["provenance"]["tokenizer"]["vocab_size"]
            require(vocab > 1, "lookahead poisoning requires at least two tokens")
            for seg, part in zip(segments, forward["parts"]):
                for row in range(part["kept"], part["offered"]):
                    seg.tokens[row] = (seg.tokens[row] + 1) % vocab
                    poisoned += 1
        out.append(Frame(forward, segments, [p["kept"] for p in forward["parts"]],
                         [list(range(p["position"] + 1, p["position"] + 1 + p["offered"]))
                          for p in forward["parts"]], poisoned_rows=poisoned))
    return out


class GraphBoundary:
    """Observe the real captured replay without substituting its implementation."""

    def __init__(self, graph, controller):
        self.graph, self.controller = graph, controller

    def replay(self):
        ctl = self.controller
        if ctl.current is None:
            return self.graph.replay()
        i, _ = ctl.current
        ctl.trace.end(i, "staging")
        ctl.trace.begin(i, "target")
        result = self.graph.replay()
        ctl.graph_calls += 1
        ctl.trace.end(i, "target")
        ctl.trace.begin(i, "tap_bookkeeping")
        return result


class Controller:
    """One verify, selection, all target commits and one full context group before another forward."""

    def __init__(self, verifier, drafts, sample_streams, weights, guard, trace=None):
        self.verifier, self.drafts, self.sample = verifier, drafts, sample_streams
        self.weights, self.guard = weights, guard
        self.trace = trace or QuietTrace()
        self.current = None
        self.graph_calls = 0
        self.next_id = 0
        self.broken = False

    def wrap_graphs(self):
        for key, graph in self.verifier.graphs.items():
            require(not isinstance(graph, GraphBoundary), "graph observer already installed")
            self.verifier.graphs[key] = GraphBoundary(graph, self)

    def start(self):
        require(not self.broken and self.current is None, "stale or failed replay transaction")
        self.next_id = 0

    def execute(self, frame, index, graph=True):
        require(not self.broken and self.current is None, "stale or failed replay transaction")
        self.current = (index, frame)
        try:
            self._execute(frame, index, graph)
            self.next_id += 1
        except BaseException:
            self.broken = True
            raise
        finally:
            self.current = None

    def _execute(self, frame, index, graph):
        scheduled, segments, keeps = frame.scheduled, frame.segments, frame.keeps
        require(scheduled["forward_id"] == self.next_id, "duplicate or stale forward ID")
        require(len(segments) == len(keeps) == len(frame.positions) == len(scheduled["parts"]) > 0,
                "truncated replay part/count vector")
        require(len({s.st.slot for s in segments}) == len(segments), "duplicate stream slot")
        require(not frame.target_commits and not frame.context_commits, "forward already consumed")
        rows = sum(len(s.tokens) for s in segments)
        if graph:
            require(rows in self.verifier.graphs, f"missing captured total-row graph: {rows}")
        calls = self.guard.calls
        require(calls == 0 and frame.lost == 0, "proposer/lost audit already failed")
        contexts = []
        for seg, part, keep, positions in zip(segments, scheduled["parts"], keeps, frame.positions):
            ctx = self.drafts.contexts[part["stream_id"]]
            require(seg.st.slot == part["stream_id"] and seg.st.pos == part["position"], "stale target state")
            require(ctx.context_end == seg.st.pos, "stale/lost drafter context before forward")
            require(len(seg.tokens) == part["offered"] and keep == part["kept"]
                    and 1 <= keep <= len(seg.tokens), "invalid replay keep/window")
            require(positions == list(range(part["position"] + 1, part["position"] + 1 + part["offered"])),
                    "selection position vector differs from prescribed rows")
            contexts.append(ctx)
        before = dict(self.verifier.replays) if graph else None
        boundary = self.graph_calls
        self.trace.begin(index, "staging")
        verified = self.verifier.forward(segments)
        if graph:
            self.trace.end(index, "tap_bookkeeping")
            frame.graph_replays = self.verifier.replays["graph"] - before["graph"]
            frame.eager_fallbacks = self.verifier.replays["eager"] - before["eager"]
            require(frame.graph_replays == 1 and frame.eager_fallbacks == 0
                    and self.graph_calls - boundary == 1, "eager fallback or missing real graph replay")
        require(len(verified.logits) == len(verified.taps) == len(segments), "truncated verified outputs")
        for seg, logits, taps in zip(segments, verified.logits, verified.taps):
            require(logits.shape[0] == len(seg.tokens) and taps is not None
                    and taps.shape[0] == len(seg.tokens), "missing normal logits/taps")
        self.trace.begin(index, "selection")
        sampled = self.sample(self.weights, [(v, pos, None) for v, pos in zip(verified.logits, frame.positions)])
        self.trace.end(index, "selection")
        require(len(sampled) == len(segments) and all(len(v) == len(s.tokens) for v, s in zip(sampled, segments)),
                "selection did not complete every row")
        self.trace.begin(index, "target_commit")
        self.verifier.commit(segments, keeps)
        self.trace.end(index, "target_commit")
        for seg, part, keep in zip(segments, scheduled["parts"], keeps):
            require(seg.st.pos == part["position"] + keep, "incomplete target commit")
        frame.target_commits = sum(keeps)
        self.trace.begin(index, "context_commit")
        self.drafts.commit([(c, taps[:keep]) for c, taps, keep in zip(contexts, verified.taps, keeps)])
        self.trace.end(index, "context_commit")
        frame.ingest_calls = 1
        for seg, ctx in zip(segments, contexts):
            require(ctx.context_end == seg.st.pos, "partial/lost context commit")
        frame.context_commits = sum(keeps)
        require(self.guard.calls == calls == 0, "proposer was invoked")
        frame.verified, frame.sampled = verified, sampled


def forward_receipt(frame, phases):
    require(frame.target_commits == frame.context_commits > 0 and frame.ingest_calls == 1,
            "incomplete context transaction")
    return {**frame.scheduled, "phases_ns": phases,
            "owner": {"layer_visits": 0, "rows": 0, "query_send_bytes": 0, "output_send_bytes": 0},
            "graph_replays": frame.graph_replays, "eager_fallbacks": frame.eager_fallbacks,
            "target_commits": frame.target_commits, "context_commits": frame.context_commits,
            "ingest_calls": frame.ingest_calls, "proposer_calls": 0, "lost": frame.lost}
