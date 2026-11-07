//! Force all additive correctness API bodies into an inert cross-compiled object.
const std = @import("std");
const replay = @import("glm_cuda").cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const exec = @import("glm_replay_execute.zig");
const evidence = @import("glm_replay_evidence.zig");

export fn compileReplayCorrectness(a: *std.mem.Allocator, session: *replay.Session, f: *const fixture.Fixture, raw: [*]const u8, raw_len: usize, binary: [*]const u8, compiled: [*]const exec.Compiled, count: usize, observations: [*]const evidence.Observed, completion: *const replay.Completion) void {
    _ = evidence.load(a.*, raw[0..raw_len], f.*, binary[0..64], false) catch return;
    evidence.checkMeasured(compiled[0..count], f.forwards, observations[0..count]) catch return;
    const saved = completion.copy(a.*) catch return;
    defer saved.deinitCopy(a.*);
    _ = session.captureRows(0, 1) catch return;
    _ = session.capture(0, 1) catch return;
}
