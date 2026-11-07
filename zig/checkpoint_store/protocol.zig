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
fn failed(err: anyerror) u32 {
    if (err == error.Cancelled) return 3;
    return if (fs.disk.invalid(err) or err == error.LocalWorkerFailed) 0 else 2;
}
fn failure(status: u32) anyerror {
    return switch (status) {
        3 => error.BlockCollectiveCancelled,
        2 => error.BlockCollectiveIO,
        else => error.BlockCollectiveAbort,
    };
}
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
    if (a.world != t.world) return error.BlockTransport;
    var vote: Prepared = .{ .ok = 0, .agreement = a.digest() };
    const ready = receipt catch |err| failed_receipt: {
        vote.ok = failed(err);
        break :failed_receipt null;
    };
    var owned = false;
    defer if (owned) {
        s.pending = null;
        s.transaction = false;
    };
    if (s.begin(ready)) |_| {
        owned = true;
    } else |err| {
        vote.ok = failed(err);
    }
    if (owned) if (ready) |r| {
        if (std.meta.eql(r.agreement, a) and r.rank == t.rank) {
            var point = local.Point.open(s, a, t.rank, r.point, slab) catch |err| no_point: {
                vote.ok = failed(err);
                break :no_point null;
            };
            if (point) |*p| {
                defer p.close();
                vote.ok = 1;
                vote.record = r.point;
            }
        }
    };
    const votes = try t.gather(Prepared, .prepared, vote);
    var m: f.epoch.Manifest = .{ .agreement = a };
    var rejected: ?u32 = null;
    for (votes[0..t.world], 0..) |v, i| {
        if (v.ok != 1) rejected = @max(rejected orelse 0, v.ok);
        if (!std.mem.eql(u8, &v.agreement, &vote.agreement)) rejected = rejected orelse 0;
        m.ranks[i] = v.record;
    }
    if (rejected) |status| return failure(status);
    try local.validateManifest(&m);
    var durable: Durable = .{ .ok = 1, .manifest = f.wire.hash(m) };
    const published_new = local.publish(s, &m, t.rank, slab) catch |err| unpublished: {
        durable.ok = failed(err);
        break :unpublished false;
    };
    const barriers = try t.gather(Durable, .durable, durable);
    rejected = null;
    for (barriers[0..t.world]) |v| {
        if (v.ok != 1) rejected = @max(rejected orelse 0, v.ok);
        if (!std.mem.eql(u8, &v.manifest, &durable.manifest)) rejected = rejected orelse 0;
    }
    if (rejected) |status| {
        // Reliable identical views establish that no rank acknowledged; uncertain transport loss retains publications.
        if (published_new) {
            s.remove(a.manifestName(), ".commit") catch {};
            _ = @import("sys").c.fsync(s.fd);
            s.recount() catch {
                s.used = s.quota;
            };
        }
        return failure(status);
    }
    return .{ .files = s, .manifest = m, .rank = t.rank };
}

/// Intersect locally valid histories within the same request scope; a bounded search may yield a collective cache miss.
pub fn recover(s: *fs.Files, selection: local.Selection, t: *Transport, slab: []u8, max_rounds: u32) !?Epoch {
    var guard_error: ?anyerror = null;
    if (s.begin(null)) |_| {} else |err| {
        guard_error = err;
    }
    defer if (guard_error == null) {
        s.transaction = false;
    };
    var bound: local.Bound = .{ .key = .{ .epoch = std.math.maxInt(u64), .name = f.zero } };
    for (0..max_rounds) |_| {
        var proposal: Proposal = .{ .status = 0, .manifest = std.mem.zeroes(f.epoch.Manifest) };
        const candidate = if (guard_error != null) no_guard: {
            proposal.status = 2;
            break :no_guard null;
        } else local.latestBefore(s, selection, t.world, t.rank, bound, slab) catch |err| no_candidate: {
            proposal.status = if (err == error.Cancelled) 3 else 2;
            break :no_candidate null;
        };
        if (candidate) |m| proposal = .{ .status = 1, .manifest = m };
        const proposals = try t.gather(Proposal, .proposal, proposal);
        var rejected: u32 = 0;
        for (proposals[0..t.world]) |p| if (p.status > 1) {
            rejected = @max(rejected, p.status);
        };
        if (rejected > 1) return failure(rejected);
        for (proposals[0..t.world]) |p| if (p.status == 0) return null;
        var lo = local.key(proposals[0].manifest);
        var hi = lo;
        var identical = true;
        const common = f.wire.hash(proposals[0].manifest);
        for (proposals[0..t.world]) |p| {
            try local.validateManifest(&p.manifest);
            const a = p.manifest.agreement;
            if (a.world != t.world or !local.within(p.manifest, bound) or !selection.accepts(selection.ctx, a)) return error.BlockCollectiveAbort;
            const k = local.key(p.manifest);
            if (local.less(k, lo)) lo = k;
            if (local.less(hi, k)) hi = k;
            identical = identical and std.mem.eql(u8, &common, &f.wire.hash(p.manifest));
        }
        if (identical) {
            const m = proposals[0].manifest;
            var ready: Durable = .{ .ok = 1, .manifest = common };
            s.control.check() catch |err| {
                ready.ok = failed(err);
            };
            const barriers = try t.gather(Durable, .recovered, ready);
            var aborted: ?u32 = null;
            for (barriers[0..t.world]) |v| {
                if (v.ok != 1) aborted = @max(aborted orelse 0, v.ok);
                if (!std.mem.eql(u8, &v.manifest, &common)) aborted = aborted orelse 0;
            }
            if (aborted) |status| return failure(status);
            return .{ .files = s, .manifest = m, .rank = t.rank };
        }
        bound = .{ .key = lo, .inclusive = !std.meta.eql(lo, hi) };
    }
    return error.BlockSearchBudget;
}
