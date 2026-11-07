//! Untimed poisoned-lookahead, one-row oracle and synchronized whole-schedule validation.
const std = @import("std");
const replay = @import("glm_cuda").cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");
const exec = @import("glm_replay_execute.zig");
pub const evidence = @import("glm_replay_evidence.zig");
pub const Observed = evidence.Observed;
pub const checkMeasured = evidence.checkMeasured;
pub const Evidence = struct { observations: []const Observed, comparisons: u64 };

pub fn run(a: std.mem.Allocator, s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, sharded: bool) !Evidence {
    const expected = try a.alloc(Observed, compiled.len);
    const actual = try a.alloc(Observed, compiled.len);
    const poison_prep = try exec.compileValidation(a, f, f.preparation);
    const poisoned = try exec.compileValidation(a, f, f.forwards);
    const g = try exec.geometry(s.owned.e);
    try s.enableTiming(false);
    try batch(s, f, b, prep, compiled, expected, sharded, g);
    var comparisons = try oracle(s, f, b, prep, expected);
    try batch(s, f, b, poison_prep.rounds, poisoned.rounds, actual, sharded, g);
    comparisons += try compareBatches(expected, actual, f.forwards);
    try s.enableTiming(true);
    try batch(s, f, b, poison_prep.rounds, poisoned.rounds, actual, sharded, g);
    comparisons += try compareBatches(expected, actual, f.forwards);
    try s.enableTiming(false);
    comparisons += try checkWhole(s, f, b, prep, compiled, expected, sharded, g, false);
    comparisons += try checkWhole(s, f, b, poison_prep.rounds, poisoned.rounds, expected, sharded, g, true);
    return .{ .observations = expected, .comparisons = comparisons };
}

fn prepare(s: anytype, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled) !void {
    try exec.prepareStreams(s, f, b);
    _ = try validationBlock(s, prep);
}

fn validationBlock(s: anytype, compiled: []exec.Compiled) !replay.Completion {
    try s.beginBlock();
    for (compiled, 0..) |*c, i| try s.round(i, c.parts[0..c.n]);
    return s.complete(compiled.len);
}

pub fn checkWhole(s: anytype, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, expected: []const Observed, sharded: bool, g: ctl.Geometry, poisoned: bool) !u64 {
    try prepare(s, f, b, prep);
    const completed = if (poisoned) try validationBlock(s, compiled) else try exec.block(s, compiled);
    try exec.checkCompletion(completed, f.forwards, sharded, g);
    if (poisoned) try evidence.checkOutputs(compiled, f.forwards, expected) else try checkMeasured(compiled, f.forwards, expected);
    const last = f.forwards[f.forwards.len - 1];
    var seen: [4]bool = @splat(false);
    for (last.parts, 0..) |p, i| {
        seen[p.stream_id] = true;
        const got = try s.capture(p.stream_id, p.position + p.kept);
        for (0..2) |rank| try evidence.compare(expected[expected.len - 1].shots[i][rank], got[rank], p.kept);
    }
    for (f.config.active_streams) |slot| if (!seen[slot]) return error.FinalForwardMissingStream;
    return 2 * last.parts.len;
}

fn batch(s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, out: []Observed, sharded: bool, g: ctl.Geometry) !void {
    try prepare(s, f, b, prep);
    for (compiled, f.forwards, out) |*c, r, *o| {
        o.* = std.mem.zeroes(Observed);
        try s.beginBlock();
        try s.round(0, c.parts[0..c.n]);
        const completed = try s.complete(1);
        var one = r;
        one.forward_id = 0;
        one.round_id = 0;
        try exec.checkCompletion(completed, &.{one}, sharded, g);
        @memcpy(&o.records, completed.records);
        for (r.parts, 0..) |p, i| {
            o.shots[i] = try s.capture(p.stream_id, p.position + p.kept);
            @memcpy(o.outputs[i][0..p.kept], c.parts[i].out[0..p.kept]);
        }
    }
}

fn compareBatches(expected: []const Observed, actual: []const Observed, rounds: []const fixture.Round) !u64 {
    var comparisons: u64 = 0;
    for (expected, actual, rounds) |want, got, r| {
        for (r.parts, 0..) |p, i| {
            for (0..2) |rank| try evidence.compare(want.shots[i][rank], got.shots[i][rank], p.kept);
            comparisons += 2;
            if (!std.mem.eql(u32, want.outputs[i][0..p.kept], got.outputs[i][0..p.kept])) return error.ValidationSelection;
        }
        for (0..2) |rank| if (!std.meta.eql(want.records[rank], got.records[rank])) return error.TimingChangedDispatch;
    }
    return comparisons;
}

fn oracle(s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, expected: []const Observed) !u64 {
    var comparisons: u64 = 0;
    try exec.prepare(s, f, b, prep);
    for (f.forwards, expected) |r, want| for (r.parts, 0..) |p, i| {
        var last: [2]replay.Snapshot = undefined;
        for (0..p.kept) |row| {
            var selected: [1]u32 = undefined;
            const offset = p.tape_offset + row;
            const tape = f.streams[p.stream_id].tape_tokens;
            const one: replay.Part = .{ .slot = p.stream_id, .position = p.position + row, .ids = tape[offset..][0..1], .keep = 1, .anchor = tape[offset + 1], .out = &selected };
            try s.beginBlock();
            try s.round(0, &.{one});
            _ = try s.complete(1);
            const end = row + 1 == p.kept;
            last = if (end) try s.capture(p.stream_id, p.position + row + 1) else try s.captureRows(p.stream_id, p.position + row + 1);
            if (selected[0] != want.outputs[i][row]) return error.ReferenceSelection;
            for (0..2) |rank| try evidence.rows(last[rank], 0, want.shots[i][rank], row);
            comparisons += 2;
        }
        for (0..2) |rank| try evidence.logical(last[rank], want.shots[i][rank]);
        comparisons += 2;
    };
    return comparisons;
}
