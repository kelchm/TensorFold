//! Identical fresh work for a separate phase anchor and the observer-off headline.
const std = @import("std");
const replay = @import("glm_cuda").cuda_replay;
const exec = @import("glm_replay_execute.zig");
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");
const validation = @import("glm_replay_validate.zig");
const clock = @import("check.zig").now;

pub const Measured = struct {
    headline: replay.Completion,
    anchor: replay.Completion,
    preparation_ns: u64,

    pub fn deinit(m: Measured, a: std.mem.Allocator) void {
        m.headline.deinitCopy(a);
        m.anchor.deinitCopy(a);
    }
};

pub fn run(a: std.mem.Allocator, io: std.Io, s: anytype, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, sharded: bool, g: ctl.Geometry, reference: []const validation.Observed) !Measured {
    try s.enableTiming(false);
    try exec.prepare(s, f, b, prep);
    _ = try checked(s, f, compiled, sharded, g, reference);
    try s.enableTiming(true);
    try exec.prepare(s, f, b, prep);
    const profile = try checked(s, f, compiled, sharded, g, reference);
    try checkObserver(profile, f.forwards.len, true);
    const anchor = try profile.copy(a);
    errdefer anchor.deinitCopy(a);
    try s.enableTiming(false);
    const before = clock(io);
    try exec.prepare(s, f, b, prep);
    const preparation_ns: u64 = @intCast(clock(io) - before);
    const wall = try checked(s, f, compiled, sharded, g, reference);
    try checkObserver(wall, f.forwards.len, false);
    return .{ .headline = try wall.copy(a), .anchor = anchor, .preparation_ns = preparation_ns };
}

fn checked(s: anytype, f: fixture.Fixture, compiled: []exec.Compiled, sharded: bool, g: ctl.Geometry, reference: []const validation.Observed) !replay.Completion {
    const completed = try exec.block(s, compiled);
    try exec.checkCompletion(completed, f.forwards, sharded, g);
    try validation.checkMeasured(compiled, f.forwards, reference);
    _ = try validation.checkFinal(s, f, reference);
    return completed;
}

pub fn checkObserver(c: replay.Completion, forwards: usize, enabled: bool) !void {
    if (c.timings.len != (if (enabled) 2 * forwards else @as(usize, 0))) return error.HeadlineObserver;
}
