//! Bounded allgather protocol over immutable receipts; data stays on the rank that owns its NVMe.
const std = @import("std");
const f = @import("format.zig");
const fs = @import("files.zig");
const local = @import("local.zig");
pub const Phase = enum(u32) { prepared = 1, durable = 2, proposal = 3, recovered = 4 };

/// Requires reliable rank membership, timeout/cancel and identical allgather views; sends no token or row payload.
pub const Transport = struct {
    ctx: *anyopaque,
    world: u32,
    rank: u32,
    session: f.Name,
    scope: f.Name,
    round: u64 = 0,
    exchange: *const fn (*anyopaque, Phase, u64, []const u8, []u8) anyerror!void,

    fn gather(t: *Transport, comptime T: type, phase: Phase, value: T) ![f.epoch.max_ranks]T {
        if (t.world == 0 or t.world > f.epoch.max_ranks or t.rank >= t.world or std.mem.eql(u8, &t.session, &f.zero) or std.mem.eql(u8, &t.scope, &f.zero)) return error.BlockTransport;
        const Frame = struct { version: u32 = 1, session: f.Name, scope: f.Name, phase: u32, round: u64, world: u32, rank: u32, value: T };
        t.round = try std.math.add(u64, t.round, 1);
        const frame: Frame = .{ .session = t.session, .scope = t.scope, .phase = @backingInt(phase), .round = t.round, .world = t.world, .rank = t.rank, .value = value };
        var send: [f.wire.size(Frame)]u8 = undefined;
        const encoded = try f.wire.encode(frame, &send);
        var received: [f.epoch.max_ranks * f.wire.size(Frame)]u8 = undefined;
        const bytes = received[0 .. t.world * encoded.len];
        try t.exchange(t.ctx, phase, t.round, encoded, bytes);
        var out: [f.epoch.max_ranks]T = undefined;
        for (0..t.world) |i| {
            const r = try f.wire.decode(Frame, bytes[i * encoded.len ..][0..encoded.len]);
            if (r.version != 1 or !std.mem.eql(u8, &r.session, &t.session) or !std.mem.eql(u8, &r.scope, &t.scope) or r.phase != frame.phase or r.round != frame.round or r.world != t.world or r.rank != i) return error.BlockTransport;
            if (i == t.rank and !std.meta.eql(r, frame)) return error.BlockTransport;
            out[i] = r.value;
        }
        return out;
    }
};

const Prepared = struct { ok: u32, agreement: f.Name, record: f.epoch.Record = .{} };
const Durable = struct { ok: u32, manifest: f.Name };
const Proposal = struct { status: u32, manifest: f.epoch.Manifest };

pub const Epoch = struct {
    files: *fs.Files,
    manifest: f.epoch.Manifest,
    rank: u32,
    pub fn reader(e: *const Epoch, slab: []u8) !local.Reader {
        return local.Reader.open(e.files, &e.manifest, e.rank, slab);
    }
};

/// Even a failed local capture participates in the prepared vote. Never acknowledge from a local receipt alone.
pub fn commit(s: *fs.Files, a: f.epoch.Agreement, receipt: anyerror!?f.Receipt, t: *Transport, slab: []u8) !Epoch {
    try a.validate();
    if (a.world != t.world) return error.BlockTransport;
    var vote: Prepared = .{ .ok = 0, .agreement = a.digest() };
    const ready = receipt catch null;
    if (ready) |r| {
        if (std.meta.eql(r.agreement, a) and r.rank == t.rank) {
            var point = local.Point.open(s, a, t.rank, r.point, slab) catch null;
            if (point) |*p| {
                defer p.close();
                if (p.validateObjects(s, slab)) |_| {
                    vote.ok = 1;
                    vote.record = r.point;
                } else |_| {}
            }
        }
    }
    const votes = try t.gather(Prepared, .prepared, vote);
    var m: f.epoch.Manifest = .{ .agreement = a };
    for (votes[0..t.world], 0..) |v, i| {
        if (v.ok != 1 or !std.mem.eql(u8, &v.agreement, &vote.agreement)) return error.BlockCollectiveAbort;
        m.ranks[i] = v.record;
    }
    try local.validateManifest(&m);
    var durable: Durable = .{ .ok = 1, .manifest = f.wire.hash(m) };
    local.publish(s, &m, t.rank, slab) catch {
        durable.ok = 0;
    };
    const barriers = try t.gather(Durable, .durable, durable);
    for (barriers[0..t.world]) |v| if (v.ok != 1 or !std.mem.eql(u8, &v.manifest, &durable.manifest)) return error.BlockCollectiveAbort;
    return .{ .files = s, .manifest = m, .rank = t.rank };
}

/// Intersect locally valid histories within the same request scope; a bounded search may yield a collective cache miss.
pub fn recover(s: *fs.Files, selection: local.Selection, t: *Transport, slab: []u8, max_rounds: u32) !?Epoch {
    var ceiling: u64 = std.math.maxInt(u64);
    for (0..max_rounds) |_| {
        var proposal: Proposal = .{ .status = 0, .manifest = std.mem.zeroes(f.epoch.Manifest) };
        const candidate = local.latest(s, selection, t.world, t.rank, ceiling, slab) catch failed: {
            proposal.status = 2;
            break :failed null;
        };
        if (candidate) |m| proposal = .{ .status = 1, .manifest = m };
        const proposals = try t.gather(Proposal, .proposal, proposal);
        for (proposals[0..t.world]) |p| if (p.status > 1) return error.BlockCollectiveAbort;
        for (proposals[0..t.world]) |p| if (p.status == 0) return null;
        var lo: u64 = ceiling;
        var hi: u64 = 0;
        var identical = true;
        const common = f.wire.hash(proposals[0].manifest);
        for (proposals[0..t.world]) |p| {
            try local.validateManifest(&p.manifest);
            const a = p.manifest.agreement;
            if (a.world != t.world or a.epoch >= ceiling or !selection.accepts(selection.ctx, a)) return error.BlockCollectiveAbort;
            lo = @min(lo, a.epoch);
            hi = @max(hi, a.epoch);
            identical = identical and std.mem.eql(u8, &common, &f.wire.hash(p.manifest));
        }
        if (identical) {
            const m = proposals[0].manifest;
            var ready: Durable = .{ .ok = 1, .manifest = common };
            local.validateLocal(s, &m, t.rank, slab) catch {
                ready.ok = 0;
            };
            const barriers = try t.gather(Durable, .recovered, ready);
            for (barriers[0..t.world]) |v| if (v.ok != 1 or !std.mem.eql(u8, &v.manifest, &common)) return error.BlockCollectiveAbort;
            return .{ .files = s, .manifest = m, .rank = t.rank };
        }
        ceiling = if (lo == hi) lo else lo + 1;
    }
    return null;
}
