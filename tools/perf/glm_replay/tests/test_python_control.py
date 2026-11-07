"""Failure-oriented tests of the external replay boundary and global tap lifecycle."""
import unittest

from tools.perf.glm_replay.common import ReplayError
from tools.perf.glm_replay.python_control import forward_receipt, frames
from tools.perf.glm_replay.python_validation import compare_oracle, oracle_schedule
from tools.perf.glm_replay.tests.python_doubles import setup


class ControlTests(unittest.TestCase):
    def test_all_target_commits_then_one_full_context_group(self):
        ctl, frame, log = setup()
        ctl.execute(frame, 0)
        self.assertEqual(log, ["forward", "real_graph", "selection", ("target_commit", (1, 2, 3, 4)),
                               ("context_commit", (1, 2, 3, 4))])
        self.assertEqual(frame.target_commits, frame.context_commits)
        self.assertEqual((frame.graph_replays, frame.eager_fallbacks, frame.ingest_calls), (1, 0, 1))
        self.assertEqual(ctl.guard.calls, 0)

    def test_predictions_never_advance_prescribed_input(self):
        ctl, frame, _ = setup()
        before = [list(s.tokens) for s in frame.segments]
        ctl.execute(frame, 0)
        self.assertEqual(before, [s.tokens for s in frame.segments])
        self.assertTrue(all(99 not in s.tokens for s in frame.segments))

    def test_missing_graph_refused_before_forward(self):
        ctl, frame, log = setup()
        del ctl.verifier.graphs[16]
        with self.assertRaisesRegex(ReplayError, "missing captured"):
            ctl.execute(frame, 0)
        self.assertEqual(log, [])

    def test_eager_fallback_is_not_a_measurement(self):
        ctl, frame, log = setup()
        ctl.verifier.force_eager = True
        with self.assertRaisesRegex(ReplayError, "fallback"):
            ctl.execute(frame, 0)
        self.assertNotIn("selection", log)

    def test_graph_counter_without_replay_boundary_is_refused(self):
        ctl, frame, _ = setup()
        ctl.verifier.graphs[16] = ctl.verifier.graphs[16].graph
        with self.assertRaisesRegex(ReplayError, "missing real graph"):
            ctl.execute(frame, 0)

    def test_short_keep_vector_cannot_zip_truncate(self):
        ctl, frame, log = setup()
        frame.keeps.pop()
        with self.assertRaisesRegex(ReplayError, "truncated"):
            ctl.execute(frame, 0)
        self.assertEqual(log, [])

    def test_short_verified_vector_cannot_zip_truncate(self):
        ctl, frame, log = setup()
        ctl.verifier.truncate = True
        with self.assertRaisesRegex(ReplayError, "truncated verified"):
            ctl.execute(frame, 0)
        self.assertNotIn("selection", log)

    def test_stale_selection_positions_refused_before_forward(self):
        ctl, frame, log = setup()
        frame.positions[0][0] += 1
        with self.assertRaisesRegex(ReplayError, "selection position"):
            ctl.execute(frame, 0)
        self.assertEqual(log, [])

    def test_existing_lost_audit_refused_before_forward(self):
        ctl, frame, log = setup()
        frame.lost = 1
        with self.assertRaisesRegex(ReplayError, "audit already failed"):
            ctl.execute(frame, 0)
        self.assertEqual(log, [])

    def test_observer_brackets_real_staging_graph_and_tap_boundaries(self):
        ctl, frame, log = setup()
        class Trace:
            def begin(self, index, phase):
                log.append(("begin", phase))
            def end(self, index, phase):
                log.append(("end", phase))
        ctl.trace = Trace()
        ctl.execute(frame, 0)
        self.assertEqual(log[:8], [("begin", "staging"), "forward", ("end", "staging"),
                                  ("begin", "target"), "real_graph", ("end", "target"),
                                  ("begin", "tap_bookkeeping"), ("end", "tap_bookkeeping")])
        self.assertEqual(frame.graph_replays, 1)

    def test_short_sample_vector_prevents_commit(self):
        ctl, frame, log = setup()
        ctl.sample = lambda *_: [[99]]
        with self.assertRaisesRegex(ReplayError, "selection did not complete"):
            ctl.execute(frame, 0)
        self.assertEqual(log, ["forward", "real_graph"])

    def test_partial_context_commit_breaks_session(self):
        ctl, frame, _ = setup()
        ctl.drafts.partial = True
        with self.assertRaisesRegex(ReplayError, "partial/lost"):
            ctl.execute(frame, 0)
        self.assertTrue(ctl.broken)
        with self.assertRaisesRegex(ReplayError, "failed replay"):
            ctl.start()
        with self.assertRaisesRegex(ReplayError, "incomplete context"):
            forward_receipt(frame, {})

    def test_stale_context_refused_before_shared_taps_overwritten(self):
        ctl, frame, log = setup()
        ctl.drafts.contexts[3].context_end -= 1
        with self.assertRaisesRegex(ReplayError, "stale/lost"):
            ctl.execute(frame, 0)
        self.assertEqual(log, [])

    def test_stale_target_position_refused(self):
        ctl, frame, log = setup()
        frame.segments[0].st.pos += 1
        with self.assertRaisesRegex(ReplayError, "stale target"):
            ctl.execute(frame, 0)
        self.assertEqual(log, [])

    def test_duplicate_forward_is_refused(self):
        ctl, frame, _ = setup()
        ctl.execute(frame, 0)
        with self.assertRaisesRegex(ReplayError, "stale forward"):
            ctl.execute(frame, 1)

    def test_duplicate_slots_are_refused(self):
        ctl, frame, _ = setup()
        frame.segments[-1].st.slot = 0
        with self.assertRaisesRegex(ReplayError, "duplicate stream"):
            ctl.execute(frame, 0)

    def test_proposer_fails_without_entering_production_body(self):
        ctl, _, _ = setup()
        with self.assertRaisesRegex(ValueError, "proposer is forbidden"):
            ctl.drafts.propose()
        self.assertEqual(ctl.guard.calls, 1)

    def test_solo_still_uses_batched_graph_with_four_context_slots(self):
        ctl, frame, log = setup((8,), (4,))
        ctl.execute(frame, 0)
        self.assertEqual(len(ctl.drafts.contexts), 4)
        self.assertEqual(log[-1], ("context_commit", (4,)))

    def test_oracle_only_visits_kept_rows_through_single_row_calls(self):
        _, frame, _ = setup()
        schedule = oracle_schedule([frame.scheduled])
        self.assertEqual(len(schedule), 10)
        self.assertTrue(all(len(f["parts"]) == f["parts"][0]["offered"] == f["parts"][0]["kept"] == 1
                            for f in schedule))
        self.assertEqual([f["parts"][0]["stream_id"] for f in schedule], [0, 1, 1, 2, 2, 2, 3, 3, 3, 3])

    def test_oracle_rejects_live_ring_mismatch(self):
        fixture = {"forwards": [{"forward_id": 0, "parts": [{"stream_id": 0, "kept": 1}]}]}
        kept = {"0": {"logits": ["same"], "taps": ["same"], "selected": [1]}}
        batch = [{"kept": kept, "states": {"0": {"ring_k": "wrong"}}}]
        oracle = [{"kept": kept, "states": {"0": {"ring_k": "fresh"}}}]
        with self.assertRaisesRegex(ReplayError, "cache/ring mismatch"):
            compare_oracle(fixture, batch, oracle)

    def test_validation_poison_changes_only_discarded_tokens_and_not_fixture(self):
        from types import SimpleNamespace

        fixture = {"provenance": {"tokenizer": {"vocab_size": 8}},
                   "streams": [{"tape_tokens": [0, 1, 7, 3, 4]}]}
        schedule = [{"forward_id": 0, "round_id": 0,
                     "parts": [{"stream_id": 0, "position": 2, "tape_offset": 0, "offered": 4, "kept": 2}]}]
        segment = lambda st, tokens: SimpleNamespace(st=st, tokens=tokens)
        ordinary = frames(fixture, schedule, [object()], segment)
        poisoned = frames(fixture, schedule, [object()], segment, poison=True)
        self.assertEqual(ordinary[0].segments[0].tokens, [0, 1, 7, 3])
        self.assertEqual(poisoned[0].segments[0].tokens, [0, 1, 0, 4])
        self.assertEqual(fixture["streams"][0]["tape_tokens"], [0, 1, 7, 3, 4])

    def test_oracle_part_end_marks_do_not_merge_adjacent_solo_forwards(self):
        _, frame, _ = setup((4,), (2,))
        second = {**frame.scheduled, "forward_id": 1, "round_id": 1,
                  "parts": [{**frame.scheduled["parts"][0], "position": 14, "tape_offset": 2}]}
        schedule = oracle_schedule([frame.scheduled, second])
        self.assertEqual([f["oracle_part_end"] for f in schedule], [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
