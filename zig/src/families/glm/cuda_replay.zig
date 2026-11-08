//! Fixed-input developer sessions using the production loader, transport, verifier and DFlash2 ingest.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const rank = @import("cuda_rank_session.zig");
const ranks = @import("cuda_ranks.zig");
const engine = @import("cuda_engine.zig");
const follow_mod = @import("cuda_follow.zig");
const pages = @import("cuda_pages.zig");
const kvs = @import("cuda_kv_shard.zig");
const dflash = @import("dflash.zig");
const boundary = @import("cuda_replay_boundary.zig");
const device = @import("cuda_replay_device.zig");
pub const runtime = @import("cuda_replay_runtime.zig");
pub const timing = @import("cuda_replay_timing.zig");
pub const capture_mod = @import("cuda_replay_capture.zig");
pub const Snapshot = capture_mod.Snapshot;
pub const Receipt = boundary.Receipt;
pub const Forward = @import("cuda_replay_audit.zig").Forward;
pub const Options = struct {
    fixture_hash: [32]u8,
    config_hash: [32]u8,
    half_pages: usize,
    max_forwards: usize,
    observe: bool = false,
};
pub const Stream = struct { slot: u32, prompt: []const u32, continuation: usize, owner: u8 };
pub const Part = struct { slot: u32, position: usize, ids: []const u32, keep: usize, anchor: u32, out: []u32 };
pub const Completion = struct {
    ranks: [2]Receipt,
    elapsed_ns: i96,
    records: []const Forward,
    timings: []const timing.Record,

    // Session completion slices expire at the next complete; copy outside the block before another run.
    pub fn copy(c: Completion, a: std.mem.Allocator) !Completion {
        const records = try a.dupe(Forward, c.records);
        errdefer a.free(records);
        return .{ .ranks = c.ranks, .elapsed_ns = c.elapsed_ns, .records = records, .timings = try a.dupe(timing.Record, c.timings) };
    }

    pub fn deinitCopy(c: Completion, a: std.mem.Allocator) void {
        a.free(c.records);
        a.free(c.timings);
    }
};

const Preflight = struct {
    options: Options,
    native_options: native.Options,

    fn check(ptr: *anyopaque, group: *const cuda.tp_group.Group, ctx: *const cuda.Context) !void {
        const p: *Preflight = @ptrCast(@alignCast(ptr));
        if (group.world != 2) return error.ReplayNeedsTP2;
        var stream = try cuda.Stream.init(ctx.d, false);
        defer stream.deinit();
        const Mine = extern struct {
            fixture: [32]u8,
            config: [32]u8,
            half: u64,
            forwards: u64,
            context: u64,
            observe: u64,
            kv8: u64,
            drafts: u64,
            drafter_bits: u64,
        };
        const mine: Mine = .{ .fixture = p.options.fixture_hash, .config = p.options.config_hash, .half = p.options.half_pages, .forwards = p.options.max_forwards, .context = p.native_options.context, .observe = @intFromBool(p.options.observe), .kv8 = @intFromBool(p.native_options.kv8), .drafts = @intFromBool(p.native_options.drafts), .drafter_bits = p.native_options.drafter_bits };
        var all: [2]Mine = undefined;
        try ranks.gather(Mine, ctx.d, &group.lib, group.comm, stream, group.rank, mine, &all);
        if (!std.meta.eql(all[0], all[1])) return error.ReplayHashMismatch;
        if (!p.native_options.kv8 or !p.native_options.drafts or p.native_options.drafter == null or
            p.native_options.drafter_bits != 4 or p.native_options.vision != null or
            p.options.half_pages == 0 or p.options.max_forwards == 0) return error.ReplayConfiguration;
        var problem: []const u8 = "";
        const set = try ranks.Settings.read(&ranks.environment, &problem);
        if (set.dense_bits != 8 or set.head_bits != 8 or set.layers != 0 or set.numerics.bits() != 0)
            return error.ReplayConfiguration;
        if (!try @import("cuda_start.zig").paging(ranks.environment("TF_KV_PAGES"), 2, true))
            return error.ReplayNeedsPages;
        if (ranks.environment("TF_GLM_DFLASH2_PROF") != null) return error.ReplayIntrusiveProfiler;
        for ([_][:0]const u8{ "TF_GLM_PROF", "TF_GLM_DFLASH2_RING_DIGEST" }) |name| {
            if (ranks.environment(name)) |v| if (!std.mem.eql(u8, v, "0")) return error.ReplayIntrusiveProfiler;
        }
    }
};

pub const Session = struct {
    owned: *rank.Rank,
    state: boundary.Boundary,
    trace: ?timing.Trace = null,
    options: Options,
    runtime_sha256: [32]u8 = undefined,
    runtime_ranks: [2][32]u8 = undefined,
    session_nonce: [32]u8 = undefined,
    checkpoints: runtime.Checkpoints = undefined,
    device_uuids: [2][40]u8 = undefined,
    prepared: bool = false,
    ended: bool = false,
    failed: bool = false,
    captures: [2]Snapshot = undefined,
    capture_dev: cuda.DeviceBuffer,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: native.Options, r: Options) !*Session {
        var pre: Preflight = .{ .options = r, .native_options = o };
        const owned = try rank.open(gpa, io, ctx, dir, o, .{ .ctx = &pre, .run = Preflight.check });
        errdefer rank.release(owned);
        errdefer cuda.tp_watch.fail("replay session setup failed");
        const s = try gpa.create(Session);
        errdefer gpa.destroy(s);
        var b = try boundary.Boundary.init(gpa, ctx.d, r.max_forwards);
        errdefer b.deinit(gpa);
        var capture_dev = try cuda.DeviceBuffer.alloc(ctx.d, 2 * @sizeOf(Snapshot));
        errdefer capture_dev.free();
        s.* = .{ .owned = owned, .state = b, .options = r, .capture_dev = capture_dev };
        s.trace = try timing.Trace.init(gpa, ctx.d, r.max_forwards);
        errdefer if (s.trace) |*t| t.deinit();
        if (s.trace) |*t| {
            s.state.trace_store = t;
            s.state.audit.trace = if (r.observe) t else null;
        }
        const drafter = try ranks.artifact(gpa, io, o.drafter.?);
        const Loaded = extern struct { runtime: [32]u8, target: [32]u8, drafter: [32]u8, nonce: [32]u8, device_uuid: [40]u8 };
        const mine: Loaded = .{ .runtime = try runtime.loaded(owned.e, drafter), .target = owned.e.checkpoint_digest, .drafter = drafter, .nonce = try runtime.nonce(io), .device_uuid = try device.uuid(ctx) };
        var all: [2]Loaded = undefined;
        try ranks.gather(Loaded, ctx.d, &owned.group.lib, owned.group.comm, owned.e.stream, owned.group.rank, mine, &all);
        for (all, 0..) |value, i| {
            s.runtime_ranks[i] = value.runtime;
            s.checkpoints.target[i] = value.target;
            s.checkpoints.drafter[i] = value.drafter;
            s.device_uuids[i] = value.device_uuid;
        }
        s.session_nonce = all[0].nonce;
        s.runtime_sha256 = runtime.combine(s.runtime_ranks);
        owned.e.replay = &s.state;
        owned.e.replay_capture = .{ .ptr = s, .run = captureRank };
        return s;
    }

    pub fn leader(s: *Session) !follow_mod.Leader {
        if (s.owned.group.rank != 0) return error.ReplayNotLeader;
        return .{ .e = s.owned.e, .link = &s.owned.link };
    }

    pub fn agree(s: *Session) !void {
        var accepted: [2][32]u8 = undefined;
        const owned = s.owned;
        try ranks.gather([32]u8, owned.e.d, &owned.group.lib, owned.group.comm, owned.e.stream, owned.group.rank, s.runtime_sha256, &accepted);
        if (!std.mem.eql(u8, &accepted[0], &accepted[1])) return error.ReplayHashMismatch;
    }

    pub fn follow(s: *Session) !void {
        if (s.owned.group.rank == 0) return error.ReplayNotFollower;
        errdefer s.fail();
        try follow_mod.follow(s.owned.e, &s.owned.link);
        s.ended = true;
    }

    pub fn close(s: *Session) void {
        if (s.failed) std.c._exit(1);
        const gpa = s.owned.gpa;
        if (s.owned.group.rank == 0 and !s.ended) {
            cuda.tp_watch.ended();
            const x = s.leader() catch unreachable;
            x.end() catch {};
        }
        cuda.completion.drain(s.owned.e.d, s.owned.e.stream, "GLM replay cleanup did not drain");
        if (s.trace) |*t| t.deinit();
        s.state.deinit(gpa);
        s.capture_dev.free();
        s.owned.e.replay = null;
        s.owned.e.replay_capture = null;
        rank.release(s.owned);
        gpa.destroy(s);
    }

    pub fn prepare(s: *Session, streams: []const Stream) !void {
        if (s.state.block or streams.len != 4) return error.ReplayPreparation;
        const x = try s.leader();
        errdefer s.fail();
        const e = x.e;
        for (streams, 0..) |st, i| {
            if (st.slot != i or st.owner >= 2 or st.prompt.len == 0) return error.ReplayPreparation;
            for (st.prompt) |id| if (id >= e.c.vocab) return error.ReplayToken;
        }
        const layout = try pages.layout(e.gpa, e, dflash.ring_bytes);
        var main: [2]usize = .{ 0, 0 };
        var twins: [2]usize = .{ 0, 0 };
        for (streams) |st| {
            if (st.continuation > e.capacity -| st.prompt.len or st.prompt.len + 16 > e.capacity)
                return error.ReplayPreparation;
            main[st.owner] = try std.math.add(usize, main[st.owner], pages.need(layout, st.prompt.len, st.continuation));
            twins[1 - st.owner] = @max(twins[1 - st.owner], layout.pages(st.prompt.len));
        }
        if (e.settings.kv_shard.on) {
            for (main, twins) |m, t| if (try std.math.add(usize, m, t) > s.options.half_pages)
                return error.ReplayPoolReservation;
        } else if (try std.math.add(usize, main[0], main[1]) > try pageArgument(s.options.half_pages, false))
            return error.ReplayPoolReservation;
        if (e.pages == null) {
            const count = try pageArgument(s.options.half_pages, e.settings.kv_shard.on);
            const l = layout;
            try kvs.refusePool(e.settings.kv_shard, 4, s.options.half_pages, e.capacity);
            try x.pool(4, count, l.pageBytes());
            e.pages = try pages.Pages.init(e.gpa, e, l, 4, count, true);
            if (e.kvCtx() != null) e.kv_ctx.counts = &e.pages.?.shard.?.parts.counts;
        }
        if (s.prepared) for (streams) |st| try x.release(st.slot);
        for (streams) |st| {
            try x.developer(7, st.slot);
            try x.bind(st.slot);
            const ids = try e.pages.?.openFor(st.slot, st.prompt.len, st.continuation, st.owner);
            if (e.pages.?.shard) |sh| if (sh.parts.slots[st.slot].part != st.owner) return error.ReplayOwnerFallback;
            try x.beginOn(null, ids);
            _ = try x.prefill(st.prompt, null, null, null);
            try x.developer(5, st.slot);
        }
        s.prepared = true;
        try e.stream.synchronize();
    }

    pub fn enableTiming(s: *Session, enabled: bool) !void {
        const x = try s.leader();
        errdefer s.fail();
        try x.developer(if (enabled) 3 else 4, 0);
    }

    pub fn beginBlock(s: *Session) !void {
        if (!s.prepared) return error.ReplayPreparation;
        const x = try s.leader();
        errdefer s.fail();
        try x.developer(1, 0);
    }

    pub fn round(s: *Session, id: u64, parts: []const Part) !void {
        const x = try s.leader();
        if (!s.state.block or id != s.state.audit.forwards or parts.len == 0 or parts.len > 4)
            return error.ReplayForwardId;
        var verify_parts: [4]engine.Part = undefined;
        var reqs: [4]dflash.Req = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.slot >= 4 or p.ids.len == 0 or p.ids.len > 16 or p.keep == 0 or p.keep > p.ids.len or
                p.out.len != p.ids.len or p.anchor >= x.e.c.vocab) return error.ReplayPart;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.ReplaySlotTwice;
            const seq = x.e.seqs.items[p.slot] orelse return error.ReplaySlot;
            if (seq.st.pos != p.position or p.ids.len > seq.st.capacity -| p.position) return error.ReplayPosition;
            for (p.ids) |token| if (token >= x.e.c.vocab) return error.ReplayToken;
            rows += p.ids.len;
            verify_parts[i] = .{ .slot = p.slot, .ids = p.ids, .out = p.out };
            reqs[i] = .{ .slot = p.slot, .pos = @intCast(p.position + p.keep), .n = p.keep, .anchor = p.anchor };
        }
        if (rows > 32) return error.ReplayPart;
        errdefer s.fail();
        if (parts.len == 1) {
            try x.bind(parts[0].slot);
            try x.verify(parts[0].ids, parts[0].out, 0);
        } else try x.verifyShared(verify_parts[0..parts.len]);
        for (parts) |p| {
            try x.bind(p.slot);
            try x.keep(p.keep);
        }
        try s.state.audit.validate(x.e, reqs[0..parts.len]);
        try x.ingest(reqs[0..parts.len]);
    }

    pub fn complete(s: *Session, expected: u64) !Completion {
        const x = try s.leader();
        errdefer s.fail();
        try x.developer(0, expected);
        return .{ .ranks = s.state.receipts, .elapsed_ns = s.state.elapsed_ns, .records = s.state.all_records[0 .. 2 * s.state.audit.used], .timings = if (s.state.audit.trace != null) s.state.all_timings[0 .. 2 * s.state.audit.used] else &.{} };
    }

    pub fn capture(s: *Session, slot: u32, position: usize) ![2]Snapshot {
        return s.captureMode(slot, position, false);
    }

    // Only position, device_position, tap_rows, taps and logits are meaningful in a row capture.
    pub fn captureRows(s: *Session, slot: u32, position: usize) ![2]Snapshot {
        return s.captureMode(slot, position, true);
    }

    fn captureMode(s: *Session, slot: u32, position: usize, rows_only: bool) ![2]Snapshot {
        if (s.state.block) return error.ReplayCaptureWhileTimed;
        const x = try s.leader();
        errdefer s.fail();
        if (rows_only) try x.captureRows(slot, position) else try x.capture(slot, position);
        return s.captures;
    }

    fn fail(s: *Session) void {
        s.failed = true;
        cuda.tp_watch.fail("GLM replay session failed");
    }

    fn captureRank(ptr: *anyopaque, slot: u32, position: usize, rows_only: bool) !void {
        const s: *Session = @ptrCast(@alignCast(ptr));
        const e = s.owned.e;
        const shot = if (rows_only) try capture_mod.captureRows(e, slot, position) else try capture_mod.capture(e, slot, position);
        const size = @sizeOf(Snapshot);
        try s.capture_dev.uploadAsync(e.share.rank * size, std.mem.asBytes(&shot), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllGather(s.capture_dev.ptr + e.share.rank * size, s.capture_dev.ptr, size, .u8, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        try s.capture_dev.download(0, std.mem.asBytes(&s.captures));
    }
};

pub fn pageArgument(half: usize, sharded: bool) !usize {
    return std.math.add(usize, try std.math.mul(usize, half, 2), if (sharded) 4 else 0);
}

test "replay equal logical pages account for the sharded trash reservation and overflow" {
    try std.testing.expectEqual(@as(usize, 1024), try pageArgument(512, false));
    try std.testing.expectEqual(@as(usize, 1028), try pageArgument(512, true));
    try std.testing.expectError(error.Overflow, pageArgument(std.math.maxInt(usize), true));
}

test {
    _ = @import("cuda_replay_audit.zig");
    _ = boundary;
    _ = capture_mod;
    _ = timing;
    _ = runtime;
    _ = device;
}

test "replay completion copy survives the next block overwriting borrowed records and timings" {
    var records = [_]Forward{.{ .id = 0, .rows = 4, .kept = 2, .counts = .{} }};
    var timings = [_]timing.Record{.{ .id = 0, .ms = @splat(1) }};
    const receipt: Receipt = .{ .forwards = 1, .groups = 1, .rows = 2, .calls = 0, .lost = 0, .owner_layers = 0, .owner_rows = 0, .owner_bytes = 0, .carries = 0, .twins = 0 };
    const borrowed: Completion = .{ .ranks = .{ receipt, receipt }, .elapsed_ns = 10, .records = &records, .timings = &timings };
    const saved = try borrowed.copy(std.testing.allocator);
    defer saved.deinitCopy(std.testing.allocator);
    records[0].rows = 32;
    timings[0].ms[0] = 99;
    try std.testing.expectEqual(@as(u64, 4), saved.records[0].rows);
    try std.testing.expectEqual(@as(f32, 1), saved.timings[0].ms[0]);
    try std.testing.expectEqual(@as(u64, 32), borrowed.records[0].rows);
}
