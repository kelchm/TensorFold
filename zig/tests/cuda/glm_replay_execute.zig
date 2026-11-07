//! Preallocated fixed-input rounds through the production session.
const std = @import("std");
const glm = @import("glm_cuda");
const replay = glm.cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");

pub const Compiled = struct {
    parts: [4]replay.Part = undefined,
    outputs: [32]u32 = undefined,
    n: usize,
};

pub fn compile(a: std.mem.Allocator, f: fixture.Fixture, rounds: []const fixture.Round) ![]Compiled {
    const all = try a.alloc(Compiled, rounds.len);
    for (all, rounds) |*c, r| {
        c.n = r.parts.len;
        var row: usize = 0;
        for (r.parts, 0..) |p, i| {
            const tape = f.streams[p.stream_id].tape_tokens;
            c.parts[i] = .{ .slot = p.stream_id, .position = p.position, .ids = tape[p.tape_offset..][0..p.offered], .keep = p.kept, .anchor = tape[p.tape_offset + p.kept], .out = c.outputs[row..][0..p.offered] };
            row += p.offered;
        }
    }
    return all;
}

pub fn prepare(s: *replay.Session, f: fixture.Fixture, b: ctl.Budget, prep: []Compiled) !void {
    var streams: [4]replay.Stream = undefined;
    for (&streams, f.streams, 0..) |*out, st, i| out.* = .{ .slot = st.stream_id, .prompt = st.prompt_tokens, .continuation = b.reservation[i] * ctl.page_rows - st.prompt_tokens.len - 16, .owner = st.owner };
    try s.prepare(&streams);
    try s.beginBlock();
    for (prep, 0..) |*c, i| try s.round(i, c.parts[0..c.n]);
    _ = try s.complete(prep.len);
}

pub fn block(s: *replay.Session, compiled: []Compiled) !replay.Completion {
    try s.beginBlock();
    for (compiled, 0..) |*c, i| try s.round(i, c.parts[0..c.n]);
    return s.complete(compiled.len);
}

pub fn geometry(e: *glm.engine.Engine) !ctl.Geometry {
    var g: ctl.Geometry = .{ .layers = 0, .heads = 0, .latent = 0 };
    for (e.w.layers) |l| if (l.mixer == .mla) {
        const layer = l.mixer.mla;
        if (g.layers != 0 and (layer.heads != g.heads or layer.lw != g.latent)) return error.MixedOwnerGeometry;
        g.heads = layer.heads;
        g.latent = layer.lw;
        g.layers += 1;
    };
    if (g.layers == 0 or g.heads == 0 or g.latent == 0) return error.Geometry;
    return g;
}

pub fn checkCompletion(c: replay.Completion, rounds: []const fixture.Round, sharded: bool, g: ctl.Geometry) !void {
    if (c.records.len != 2 * rounds.len or c.elapsed_ns <= 0) return error.IncompleteReplay;
    for (0..2) |rank| for (rounds, c.records[rank * rounds.len ..][0..rounds.len], 0..) |r, rec, i| {
        var rows: u64 = 0;
        var kept: u64 = 0;
        for (r.parts) |p| {
            rows += p.offered;
            kept += p.kept;
        }
        if (rec.id != i or rec.rows != rows or rec.kept != kept) return error.IncompleteReplay;
        const after: ctl.Counters = .{ .owner_layers = rec.counts.owner_layers, .owner_rows = rec.counts.owner_rows, .owner_bytes = rec.counts.owner_bytes, .carries = rec.counts.carries };
        try ctl.checkCounters(.{}, after, rows, sharded, g);
    };
    for (c.ranks) |rank| if (rank.forwards != rounds.len or rank.groups != rounds.len or rank.calls != 0 or rank.lost != 0 or rank.carries != 0 or rank.twins != 0) return error.IncompleteReplay;
}
