"""Host lifecycle doubles exercise full-pool clearing and fresh prefill of inactive slots too."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.perf.glm_replay.python_runtime import Runtime


class LifecycleTests(unittest.TestCase):
    def test_realtime_end_is_read_after_wall_stop_and_before_completion_readback(self):
        calls = []
        runtime = Runtime.__new__(Runtime)
        runtime.rank, runtime.guard = 0, SimpleNamespace(calls=0)
        runtime.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")))
        runtime.finish_host = SimpleNamespace(numpy=lambda: [0] * 5)
        runtime.finish_send = SimpleNamespace(copy_=lambda *a, **k: None)
        def readback():
            self.assertEqual(runtime.last_realtime_end_ns, 1234)
            calls.append("readback")
            return SimpleNamespace(view=lambda *a: SimpleNamespace(tolist=lambda: [[r, 1, 1, 0, 0] for r in (0, 1)]))
        runtime.finish_recv = SimpleNamespace(cpu=readback)
        runtime.nccl = SimpleNamespace(all_gather=lambda *a: None)
        runtime.comm = SimpleNamespace()
        frame = SimpleNamespace(target_commits=1, context_commits=1, lost=0)
        def stop():
            calls.append("wall_stop")
            return 100
        def realtime():
            calls.append("realtime_end")
            return 1234
        with patch("tools.perf.glm_replay.python_runtime.time.perf_counter_ns", side_effect=stop), \
             patch("tools.perf.glm_replay.python_runtime.time.time_ns", side_effect=realtime):
            self.assertEqual(runtime.finish([frame], 10), 90)
        self.assertEqual(calls[-3:], ["wall_stop", "realtime_end", "readback"])

    def test_fresh_prefill_clears_every_cache_and_all_four_contexts(self):
        calls = []
        def tensor(name):
            return SimpleNamespace(zero_=lambda: calls.append(("clear", name)))
        planes = [SimpleNamespace(tensor=tensor("latent")), SimpleNamespace(tensor=tensor("pooled_keys"))]
        caches = SimpleNamespace(arena=SimpleNamespace(planes=planes),
                                 rings=[(tensor("index_keys"), tensor("index_gates"))])
        slots = SimpleNamespace(**{k: tensor(k) for k in ("conv", "rec", "proj", "pos_dev", "mtp_pos_dev")})
        e = SimpleNamespace(caches=caches, slots=slots, rows=16, st=None)
        def use(st):
            previous, e.st = e.st, st
            return previous
        e.use = use
        contexts = [SimpleNamespace(context_end=999, slot=i) for i in range(4)]
        drafts = SimpleNamespace(kc=[tensor("draft_k")], vc=[tensor("draft_v")], pos=tensor("draft_pos"),
                                 contexts=contexts, _pre="stale")
        def reset():
            for c in contexts:
                c.context_end = 0
            calls.append("reset_contexts")
        drafts.reset = reset
        class Pool:
            def __init__(self, rows):
                self.extents = []
            def alloc(self, size):
                extent = SimpleNamespace(base=len(self.extents) * 2048, size=2048, twin=False)
                self.extents.append(extent)
                return extent
        def state(weights, capacity, rows, **kwargs):
            return SimpleNamespace(base=kwargs["base"], pos=0, slot=kwargs["slot"])
        def prefill(engine, prompt, sampling, *, mtp, drafter, sample):
            self.assertFalse(mtp or sample)
            self.assertEqual(drafter.context_end, 0)
            self.assertIn(("clear", "latent"), calls)
            self.assertIn(("clear", "pooled_keys"), calls)
            self.assertIn(("clear", "draft_v"), calls)
            engine.st.pos = drafter.context_end = len(prompt)
            calls.append(("prefill", engine.st.slot))
        modules = {"pool": SimpleNamespace(Pool=Pool), "forward": SimpleNamespace(State=state),
                   "decode": SimpleNamespace(prefill=prefill)}
        runtime = Runtime.__new__(Runtime)
        runtime.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("sync")))
        runtime.e, runtime.drafts, runtime.w, runtime.pool_rows = e, drafts, object(), 8192
        runtime.layout = [{"stream_id": i, "base": i * 2048, "size": 2048} for i in range(4)]
        runtime.fixture = {"streams": [{"stream_id": i, "owner": i // 2,
                                       "prompt_tokens": [i + 1] * (i + 2)} for i in range(4)],
                           "preparation": [], "forwards": [], "config": {"active_streams": [0]}}
        with patch("tools.perf.glm_replay.python_runtime.module", side_effect=modules.__getitem__):
            runtime.fresh()
            first_states = runtime.states
            runtime.fresh()
        self.assertEqual([v for v in calls if isinstance(v, tuple) and v[0] == "prefill"],
                         [("prefill", i) for i in range(4)] * 2)
        self.assertTrue(all(a is not b for a, b in zip(first_states, runtime.states)))
        self.assertIsNone(runtime.drafts._pre)
        self.assertEqual([c.context_end for c in contexts], [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
