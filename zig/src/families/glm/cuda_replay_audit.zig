//! Developer-only ingest accounting; production engines leave the audit pointer null.
const std = @import("std");
const cuda = @import("cuda");
const timing = @import("cuda_replay_timing.zig");
const kvs = @import("cuda_kv_shard.zig");
pub const max_parts = 4;
pub const Part = struct { slot: u32, pos: usize, rows: usize, kept: usize = 0 };
pub const Forward = struct { id: u64, rows: u64, kept: u64, counts: kvs.Counts };

pub const Audit = struct {
    parts: [max_parts]Part = undefined,
    n: usize = 0,
    pending: bool = false,
    forwards: u64 = 0,
    groups: u64 = 0,
    rows: u64 = 0,
    calls: u64 = 0,
    lost: u64 = 0,
    before: kvs.Counts = .{},
    trace: ?*timing.Trace = null,
    records: []Forward,
    used: usize = 0,

    pub fn start(a: *Audit, e: anytype, parts: []const Part) !void {
        if (a.pending) return error.ReplayIngestPending;
        if (parts.len == 0 or parts.len > max_parts or a.used == a.records.len) return error.ReplayCapacity;
        const dr = e.drafter orelse return error.NoDrafter;
        if (dr.calls != a.calls or dr.lost != a.lost) return error.ReplayDrafterChanged;
        if (e.pages.?.shard) |sh| {
            if (sh.parts.counts.carries != 0) return error.ReplayCarry;
            for (sh.parts.slots) |s| if (s.twin.items.len != 0) return error.ReplayTwin;
        }
        @memcpy(a.parts[0..parts.len], parts);
        a.n = parts.len;
        a.pending = true;
        a.before = counts(e);
        if (a.trace) |t| try t.begin(e.stream, a.forwards);
        a.forwards += 1;
    }

    pub fn keep(a: *Audit, slot: u32, n: usize) !void {
        if (!a.pending) return error.ReplayNoVerify;
        for (a.parts[0..a.n]) |*p| if (p.slot == slot) {
            if (p.kept != 0 or n == 0 or n > p.rows) return error.ReplayKeep;
            p.kept = n;
            return;
        };
        return error.ReplaySlot;
    }

    pub fn validate(a: *const Audit, e: anytype, reqs: anytype) !void {
        if (!a.pending or reqs.len != a.n) return error.ReplayIngestGroup;
        const dr = e.drafter orelse return error.NoDrafter;
        for (reqs, 0..) |q, i| {
            for (reqs[0..i]) |prev| if (prev.slot == q.slot) return error.ReplayDuplicateIngest;
            const p = a.parts[i];
            if (q.slot != p.slot or q.n != p.kept or p.kept == 0) return error.ReplayIngestGroup;
            const s = dr.slots.items[q.slot];
            if (s.v_step != dr.verifies or s.v_pos0 != p.pos or s.v_rows != p.rows or
                q.pos != p.pos + p.kept) return error.ReplayStaleIngest;
        }
        if (dr.calls != a.calls or dr.lost != a.lost) return error.ReplayDrafterChanged;
    }

    pub fn ingested(a: *Audit, e: anytype) !void {
        const dr = e.drafter.?;
        if (dr.calls != a.calls or dr.lost != a.lost) return error.ReplayDrafterChanged;
        var offered: u64 = 0;
        var kept: u64 = 0;
        for (a.parts[0..a.n]) |p| {
            offered += p.rows;
            kept += p.kept;
        }
        const after = counts(e);
        var delta: kvs.Counts = .{};
        inline for (@typeInfo(kvs.Counts).@"struct".field_names) |f| @field(delta, f) = @field(after, f) - @field(a.before, f);
        var layers: u64 = 0;
        var bytes: u64 = 0;
        for (e.w.layers) |layer| if (layer.mixer == .mla) {
            layers += 1;
            bytes += offered * @import("cuda_kv_shard_attn.zig").rowBytes(layer.mixer.mla);
        };
        if (e.settings.kv_shard.on) {
            if (delta.owner_layers != layers or delta.owner_rows != layers * offered or delta.owner_bytes != bytes)
                return error.ReplayOwnerCounters;
        } else if (delta.owner_layers != 0 or delta.owner_rows != 0 or delta.owner_bytes != 0)
            return error.ReplayOwnerCounters;
        a.records[a.used] = .{ .id = a.forwards - 1, .rows = offered, .kept = kept, .counts = delta };
        a.used += 1;
        a.groups += 1;
        a.rows += kept;
        a.pending = false;
        try timing.mark(a.trace, e.stream, .end);
    }
};

pub fn counts(e: anytype) kvs.Counts {
    return if (e.pages) |p| (if (p.shard) |sh| sh.parts.counts else .{}) else .{};
}

test "replay ingest rejects duplicates and stale generation before dispatch" {
    var records: [1]Forward = undefined;
    var audit: Audit = .{ .records = &records, .pending = true, .n = 2 };
    audit.parts[0] = .{ .slot = 0, .pos = 4, .rows = 2, .kept = 1 };
    audit.parts[1] = .{ .slot = 1, .pos = 8, .rows = 2, .kept = 2 };
    const S = struct { v_step: u64 = 1, v_pos0: usize, v_rows: usize = 2 };
    const D = struct { slots: struct { items: []const S }, verifies: u64 = 1, calls: u64 = 0, lost: u64 = 0 };
    const E = struct { drafter: ?*const D };
    const slots = [_]S{ .{ .v_pos0 = 4 }, .{ .v_pos0 = 8 } };
    const dr: D = .{ .slots = .{ .items = &slots } };
    const e: E = .{ .drafter = &dr };
    const Q = struct { slot: u32, n: usize, pos: usize };
    var reqs = [_]Q{ .{ .slot = 0, .n = 1, .pos = 5 }, .{ .slot = 1, .n = 2, .pos = 10 } };
    try audit.validate(e, &reqs);
    reqs[1].slot = 0;
    try std.testing.expectError(error.ReplayDuplicateIngest, audit.validate(e, &reqs));
    reqs[1].slot = 1;
    reqs[1].pos = 9;
    try std.testing.expectError(error.ReplayStaleIngest, audit.validate(e, &reqs));
}
