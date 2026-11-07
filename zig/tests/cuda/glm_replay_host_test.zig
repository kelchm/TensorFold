//! Host fixture, schedule and ownership negative controls; no driver is opened.
const std = @import("std");
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");
test {
    _ = @import("glm_replay_metadata.zig");
}
const clean =
    \\{
    \\ "schema": "glm-replay-fixture-v1",
    \\ "fixture_id": "host-control",
    \\ "kind": "synthetic",
    \\ "family": "prose",
    \\ "provenance": {
    \\  "target_revision": "078455ffe6472f9a52fbc1139f58b9db2881b25c",
    \\  "drafter_revision": "bf582e4eacc1810f76656d1811693ff6c6737d2a",
    \\  "tokenizer": {
    \\   "kind": "synthetic-byte",
    \\   "revision": "host-v1",
    \\   "vocab_size": 256,
    \\   "files": [],
    \\   "template": "",
    \\   "template_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    \\  },
    \\  "sources": [
    \\   {
    \\    "source_id": "authored",
    \\    "attribution": "Authored synthetic host control",
    \\    "license": "Apache-2.0",
    \\    "text": "synthetic",
    \\    "sha256": "b3cc0475bb78a5026098858e9889acf666d31062d513d303314eca31d36e72f2"
    \\   }
    \\  ]
    \\ },
    \\ "config": {
    \\  "speed": "standard",
    \\  "tp": 2,
    \\  "slots": 4,
    \\  "dense_bits": 8,
    \\  "head_bits": 8,
    \\  "kv": "fp8",
    \\  "reset": "fresh_prefill",
    \\  "selection": "greedy",
    \\  "proposer": false,
    \\  "active_streams": [
    \\   0,
    \\   1,
    \\   2,
    \\   3
    \\  ],
    \\  "placement": "balanced",
    \\  "pool": {
    \\   "h": 256,
    \\   "page_tokens": 512,
    \\   "configured_capacity": 1024,
    \\   "prompt_twins": 1
    \\  }
    \\ },
    \\ "streams": [
    \\  {
    \\   "stream_id": 0,
    \\   "owner": 0,
    \\   "source_id": "authored",
    \\   "rendered_prompt": "prompt0",
    \\   "rendered_tape": "tape0",
    \\   "prompt_encoding": {
    \\    "token_ids": [
    \\     1,
    \\     2
    \\    ],
    \\    "used_tokens": 2
    \\   },
    \\   "prompt_tokens": [
    \\    1,
    \\    2
    \\   ],
    \\   "tape_tokens": [
    \\    5,
    \\    6,
    \\    7,
    \\    8,
    \\    9
    \\   ],
    \\   "rendered_prompt_sha256": "c942b8911d1d1582829de70db66841f0fb363a8b95b831fa9f5f155e76130ba4",
    \\   "rendered_tape_sha256": "04b8e3acd8bbe1f21d5ca64c07122850edba3903d7ffb9e9bcbfd8d151b6d926"
    \\  },
    \\  {
    \\   "stream_id": 1,
    \\   "owner": 0,
    \\   "source_id": "authored",
    \\   "rendered_prompt": "prompt1",
    \\   "rendered_tape": "tape1",
    \\   "prompt_encoding": {
    \\    "token_ids": [
    \\     2,
    \\     3
    \\    ],
    \\    "used_tokens": 2
    \\   },
    \\   "prompt_tokens": [
    \\    2,
    \\    3
    \\   ],
    \\   "tape_tokens": [
    \\    6,
    \\    7,
    \\    8,
    \\    9,
    \\    10
    \\   ],
    \\   "rendered_prompt_sha256": "ba703f8abdd5b1300c39b95f3892b64a83b98efa6fdd737d5d15b021844a79b7",
    \\   "rendered_tape_sha256": "0ae4770ae334535d419d045d88452a206a76b0017c06a3f0cd14c4b1161885d0"
    \\  },
    \\  {
    \\   "stream_id": 2,
    \\   "owner": 1,
    \\   "source_id": "authored",
    \\   "rendered_prompt": "prompt2",
    \\   "rendered_tape": "tape2",
    \\   "prompt_encoding": {
    \\    "token_ids": [
    \\     3,
    \\     4
    \\    ],
    \\    "used_tokens": 2
    \\   },
    \\   "prompt_tokens": [
    \\    3,
    \\    4
    \\   ],
    \\   "tape_tokens": [
    \\    7,
    \\    8,
    \\    9,
    \\    10,
    \\    11
    \\   ],
    \\   "rendered_prompt_sha256": "9e4eb8e7f619d63865316eab7d47d530d5baaa7ee0fc6687c78493d8868e3997",
    \\   "rendered_tape_sha256": "1c60e837f67c733864c1b402b10bdc68f3861e3b5734988c7d4ad584dbfeedda"
    \\  },
    \\  {
    \\   "stream_id": 3,
    \\   "owner": 1,
    \\   "source_id": "authored",
    \\   "rendered_prompt": "prompt3",
    \\   "rendered_tape": "tape3",
    \\   "prompt_encoding": {
    \\    "token_ids": [
    \\     4,
    \\     5
    \\    ],
    \\    "used_tokens": 2
    \\   },
    \\   "prompt_tokens": [
    \\    4,
    \\    5
    \\   ],
    \\   "tape_tokens": [
    \\    8,
    \\    9,
    \\    10,
    \\    11,
    \\    12
    \\   ],
    \\   "rendered_prompt_sha256": "311e35746b6fc8a65ca7f7e0db61cc11119f66bb2c4c3378ba536ba4a66bfaf2",
    \\   "rendered_tape_sha256": "6008b27fe5a4599c8a8d9613141c817df4850996c3475d43b79b3eb9424f08ca"
    \\  }
    \\ ],
    \\ "preparation": [
    \\  {
    \\   "round_id": 0,
    \\   "forward_id": 0,
    \\   "parts": [
    \\    {
    \\     "stream_id": 0,
    \\     "position": 2,
    \\     "tape_offset": 0,
    \\     "offered": 2,
    \\     "kept": 1
    \\    },
    \\    {
    \\     "stream_id": 1,
    \\     "position": 2,
    \\     "tape_offset": 0,
    \\     "offered": 1,
    \\     "kept": 1
    \\    },
    \\    {
    \\     "stream_id": 2,
    \\     "position": 2,
    \\     "tape_offset": 0,
    \\     "offered": 1,
    \\     "kept": 1
    \\    },
    \\    {
    \\     "stream_id": 3,
    \\     "position": 2,
    \\     "tape_offset": 0,
    \\     "offered": 1,
    \\     "kept": 1
    \\    }
    \\   ]
    \\  }
    \\ ],
    \\ "forwards": [
    \\  {
    \\   "round_id": 0,
    \\   "forward_id": 0,
    \\   "parts": [
    \\    {
    \\     "stream_id": 0,
    \\     "position": 3,
    \\     "tape_offset": 1,
    \\     "offered": 2,
    \\     "kept": 1
    \\    },
    \\    {
    \\     "stream_id": 1,
    \\     "position": 3,
    \\     "tape_offset": 1,
    \\     "offered": 1,
    \\     "kept": 1
    \\    },
    \\    {
    \\     "stream_id": 2,
    \\     "position": 3,
    \\     "tape_offset": 1,
    \\     "offered": 1,
    \\     "kept": 1
    \\    },
    \\    {
    \\     "stream_id": 3,
    \\     "position": 3,
    \\     "tape_offset": 1,
    \\     "offered": 1,
    \\     "kept": 1
    \\    }
    \\   ]
    \\  }
    \\ ],
    \\ "config_sha256": "a6953e6a5cdb2da9e53a3f84f46d5f937ef86de5ff752ae6286fad47a714cd21",
    \\ "tokens_sha256": "7dd6bbd1683461d2ceb5a248948475c72c2998533829d588be833609bb714bd4",
    \\ "schedule_sha256": "ac2b20df16fba7e3fa7fd27c61fc22957eee2f2facaea2856927276b170d22f8",
    \\ "fixture_sha256": "ca8181f5c85e9819b5375a65eb9653494b0f2c2c57206b76ce9795cf2d3b62e2"
    \\}
;

test "replay fixture accepts valid hashed partial fixture with equal capacity arms" {
    var off = try fixture.load(std.testing.allocator, clean, true, false);
    defer off.deinit();
    var on = try fixture.load(std.testing.allocator, clean, true, true);
    defer on.deinit();
    try std.testing.expectEqual(off.budget.logical_pages, on.budget.logical_pages);
    try std.testing.expectEqual(off.budget.page_argument + 4, on.budget.page_argument);
    try std.testing.expectError(error.SyntheticFixture, fixture.load(std.testing.allocator, clean, false, false));
}

fn tamper(comptime field: []const u8, replacement: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const needle = "\"" ++ field ++ "\": ";
    const at = std.mem.indexOf(u8, clean, needle).? + needle.len;
    var end = at;
    while (end < clean.len and clean[end] != ',' and clean[end] != '\n') : (end += 1) {}
    const raw = try std.mem.concat(a, u8, &.{ clean[0..at], replacement, clean[end..] });
    if (fixture.load(a, raw, true, false)) |loaded| {
        var owned = loaded;
        owned.deinit();
        return error.NegativeControlPassed;
    } else |_| {}
}

test "replay fixture rejects hash, width, keep, owner, config and vocabulary corruption" {
    try tamper("fixture_sha256", "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"");
    try tamper("offered", "17");
    try tamper("kept", "0");
    try tamper("owner", "1");
    try tamper("tp", "1");
    try tamper("vocab_size", "2");
    try tamper("position", "3");
    try tamper("tape_offset", "1");
    try tamper("offered", "true");
    try tamper("h", "1");
}

test "replay control forbids un-ingested and duplicate rounds and reoffers discarded rows" {
    var l = try fixture.load(std.testing.allocator, clean, true, false);
    defer l.deinit();
    const streams = l.fixture.streamsFor();
    var c = ctl.Cursor.init(streams);
    const f: ctl.Forward = .{ .forward_id = 0, .parts = l.fixture.preparation[0].parts };
    try c.submitted(f, streams);
    try std.testing.expectError(error.IngestPending, c.check(f, streams));
    try c.ingested(f);
    try std.testing.expectEqual(@as(usize, 1), c.offsets[0]);
    try std.testing.expectError(error.DuplicateIngest, c.ingested(f));
    const next: ctl.Forward = .{ .forward_id = 1, .parts = l.fixture.forwards[0].parts };
    try c.check(next, streams);
}

test "replay owner deltas count one shared forward on each rank" {
    const g: ctl.Geometry = .{ .layers = 11, .heads = 32, .latent = 512 };
    const expected: ctl.Counters = .{ .owner_layers = 11, .owner_rows = 88, .owner_bytes = 88 * 32 * 512 * 2 };
    try ctl.checkCounters(.{}, expected, 8, true, g);
    try ctl.checkCounters(.{}, .{}, 8, false, g);
    var bad = expected;
    bad.owner_layers *= 4;
    try std.testing.expectError(error.OwnerCounters, ctl.checkCounters(.{}, bad, 8, true, g));
    bad = expected;
    bad.lost = 1;
    try std.testing.expectError(error.ContextLost, ctl.checkCounters(.{}, bad, 8, true, g));
    bad = expected;
    bad.calls = 1;
    try std.testing.expectError(error.ProposerCalled, ctl.checkCounters(.{}, bad, 8, true, g));
}
