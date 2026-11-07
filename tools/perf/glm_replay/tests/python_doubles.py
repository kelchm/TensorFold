"""Host boundary doubles; these are not arithmetic, transport or GPU correctness evidence."""
from types import SimpleNamespace

from tools.perf.glm_replay.python_control import Controller, Frame, ProposerGuard


class Rows:
    def __init__(self, rows):
        self.shape = (rows, 8)

    def __getitem__(self, item):
        return Rows(len(range(self.shape[0])[item]))


class Graph:
    def __init__(self, log):
        self.log = log

    def replay(self):
        self.log.append("real_graph")


class Verify:
    def __init__(self, log):
        self.log = log
        self.graphs = {n: Graph(log) for n in (1, 4, 8, 16, 32)}
        self.replays = {"graph": 0, "eager": 0}
        self.force_eager = False
        self.truncate = False

    def forward(self, segments):
        self.log.append("forward")
        rows = sum(len(s.tokens) for s in segments)
        if self.force_eager:
            self.replays["eager"] += 1
        else:
            self.replays["graph"] += 1
            self.graphs[rows].replay()
        out = [Rows(len(s.tokens)) for s in segments]
        if self.truncate:
            out = out[:-1]
        return SimpleNamespace(logits=out, taps=out)

    def commit(self, segments, keeps):
        self.log.append(("target_commit", tuple(keeps)))
        for seg, keep in zip(segments, keeps):
            seg.st.pos += keep


class Drafts:
    def __init__(self, log, contexts):
        self.log, self.contexts = log, contexts
        self.partial = False

    def commit(self, items):
        self.log.append(("context_commit", tuple(t.shape[0] for _, t in items)))
        for ctx, taps in items[:-1] if self.partial else items:
            ctx.context_end += taps.shape[0]

    def propose(self):
        raise AssertionError("production proposer body must never be reached")


def setup(widths=(4, 4, 4, 4), keeps=(1, 2, 3, 4)):
    log = []
    states = [SimpleNamespace(slot=i, pos=12 + i) for i in range(4)]
    contexts = [SimpleNamespace(context_end=s.pos) for s in states]
    segments = [SimpleNamespace(st=states[i], tokens=list(range(n))) for i, n in enumerate(widths)]
    parts = [{"stream_id": i, "position": states[i].pos, "tape_offset": 0, "offered": n, "kept": k}
             for i, (n, k) in enumerate(zip(widths, keeps))]
    frame = Frame({"forward_id": 0, "round_id": 0, "parts": parts}, segments, list(keeps),
                  [list(range(s.st.pos, s.st.pos + len(s.tokens))) for s in segments])
    drafts, verify = Drafts(log, contexts), Verify(log)
    guard = ProposerGuard([drafts])
    def sample(weights, sampled_parts):
        log.append("selection")
        assert all(p[2] is None for p in sampled_parts)
        return [[99] * v.shape[0] for v, _, _ in sampled_parts]
    ctl = Controller(verify, drafts, sample, object(), guard)
    ctl.wrap_graphs()
    return ctl, frame, log
