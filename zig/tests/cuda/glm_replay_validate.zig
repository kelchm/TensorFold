//! Untimed one-row teacher-forced oracle and timing instrumentation neutrality.
const std = @import("std");
const replay = @import("glm_cuda").cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");
const exec = @import("glm_replay_execute.zig");

pub const Observed = struct {
    shots: [4][2]replay.Snapshot = undefined,
    outputs: [4][16]u32 = undefined,
    records: [2]replay.Forward = undefined,
};

pub const Evidence = struct { observations: []const Observed, comparisons: u64 };

pub fn run(a: std.mem.Allocator, s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, sharded: bool) !Evidence {
    const expected = try a.alloc(Observed, compiled.len);
    const actual = try a.alloc(Observed, compiled.len);
    const g = try exec.geometry(s.owned.e);
    try s.enableTiming(false);
    try batch(s, f, b, prep, compiled, expected, sharded, g);
    try oracle(s, f, b, prep, compiled, expected);
    try s.enableTiming(true);
    try batch(s, f, b, prep, compiled, actual, sharded, g);
    for (expected, actual, f.forwards) |want, got, r| {
        for (r.parts, 0..) |p, i| {
            for (0..2) |rank| try compare(want.shots[i][rank], got.shots[i][rank], p.kept);
            if (!std.mem.eql(u32, want.outputs[i][0..p.kept], got.outputs[i][0..p.kept])) return error.TimingChangedSelection;
        }
        for (0..2) |rank| if (!std.meta.eql(want.records[rank], got.records[rank])) return error.TimingChangedDispatch;
    }
    return .{ .observations = expected, .comparisons = 2 * compiled.len };
}

fn batch(s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, out: []Observed, sharded: bool, g: ctl.Geometry) !void {
    try exec.prepare(s, f, b, prep);
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

fn oracle(s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []exec.Compiled, compiled: []exec.Compiled, expected: []const Observed) !void {
    try exec.prepare(s, f, b, prep);
    for (compiled, f.forwards, expected) |*c, r, want| for (r.parts, 0..) |p, i| {
        var last: [2]replay.Snapshot = undefined;
        for (0..p.kept) |row| {
            var selected: [1]u32 = undefined;
            const offset = p.tape_offset + row;
            const tape = f.streams[p.stream_id].tape_tokens;
            const one: replay.Part = .{ .slot = p.stream_id, .position = p.position + row, .ids = tape[offset..][0..1], .keep = 1, .anchor = tape[offset + 1], .out = &selected };
            try s.beginBlock();
            try s.round(0, &.{one});
            _ = try s.complete(1);
            last = try s.capture(p.stream_id, p.position + row + 1);
            if (selected[0] != want.outputs[i][row]) return error.ReferenceSelection;
            for (0..2) |rank| {
                if (!std.mem.eql(u8, &last[rank].taps[0], &want.shots[i][rank].taps[row])) return error.ReferenceTaps;
                if (!std.mem.eql(u8, &last[rank].logits[0], &want.shots[i][rank].logits[row])) return error.ReferenceLogits;
            }
        }
        for (0..2) |rank| try logical(last[rank], want.shots[i][rank]);
        _ = c;
    };
}

fn logical(a: replay.Snapshot, b: replay.Snapshot) !void {
    inline for (.{ "position", "device_position", "ring_from", "index_from", "has_latent", "conv", "recurrent", "latent_codes", "latent_scales", "index_keys", "index_gates", "pooled_keys", "ring" }) |field| {
        if (!std.meta.eql(@field(a, field), @field(b, field))) return error.ReferenceLogicalState;
    }
}

fn compare(a: replay.Snapshot, b: replay.Snapshot, kept: usize) !void {
    try logical(a, b);
    if (a.tap_rows != kept or b.tap_rows != kept) return error.TapExtent;
    for (0..kept) |i| {
        if (!std.mem.eql(u8, &a.taps[i], &b.taps[i])) return error.TimingChangedTaps;
        if (!std.mem.eql(u8, &a.logits[i], &b.logits[i])) return error.TimingChangedLogits;
    }
}
