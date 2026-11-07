//! Untimed logical-state digests; physical pages, inactive banks and discarded rows are excluded.
const std = @import("std");
const cuda = @import("cuda");
const dflash = @import("dflash.zig");
const state = @import("cuda_state.zig");
const Sha = std.crypto.hash.sha2.Sha256;
pub const Digest = [32]u8;
pub const Snapshot = extern struct {
    position: u64,
    ring_from: u64,
    index_from: u64,
    device_position: i32,
    reserved: u32,
    tap_rows: u32,
    has_latent: u32,
    conv: Digest,
    recurrent: Digest,
    latent_codes: Digest,
    latent_scales: Digest,
    index_keys: Digest,
    index_gates: Digest,
    pooled_keys: Digest,
    ring: Digest,
    taps: [16]Digest,
    logits: [16]Digest,
};

const Reader = struct {
    d: *const cuda.Driver,
    scratch: [65536]u8 = undefined,

    fn hash(r: *Reader, h: *Sha, ptr: u64, bytes: usize) !void {
        var at: usize = 0;
        while (at < bytes) {
            const n = @min(bytes - at, r.scratch.len);
            const view: cuda.DeviceBuffer = .{ .d = r.d, .ptr = ptr + at, .len = n };
            try view.download(0, r.scratch[0..n]);
            h.update(r.scratch[0..n]);
            at += n;
        }
    }

    fn paged(r: *Reader, h: *Sha, ptr: u64, rows: usize, width: usize, ids: []const u32, stride: u64, page_rows: usize) !void {
        var at: usize = 0;
        while (at < rows) {
            const n = @min(rows - at, page_rows);
            if (at / page_rows >= ids.len) return error.ReplayCapturePages;
            try r.hash(h, ptr + @as(u64, ids[at / page_rows]) * stride, n * width);
            at += n;
        }
    }
};

pub fn capture(e: anytype, slot: u32, position: usize) !Snapshot {
    return captureMode(e, slot, position, false);
}

pub fn captureRows(e: anytype, slot: u32, position: usize) !Snapshot {
    return captureMode(e, slot, position, true);
}

fn captureMode(e: anytype, slot: u32, position: usize, comptime rows_only: bool) !Snapshot {
    const audit = &(e.replay orelse return error.NotReplaySession).audit;
    if (e.replay.?.block or audit.pending or audit.groups != audit.forwards) return error.ReplayStaleCapture;
    if (slot >= e.seqs.items.len) return error.ReplaySlot;
    const seq = e.seqs.items[slot] orelse return error.ReplaySlot;
    const st = &seq.st;
    if (st.pos != position or st.mtp_len != 0) return error.ReplayCapturePosition;
    const dr = e.drafter orelse return error.NoDrafter;
    if (slot >= dr.slots.items.len) return error.ReplaySlot;
    const ds = dr.slots.items[slot];
    if (ds.v_step != dr.verifies or ds.v_pos0 < 0) return error.ReplayStaleCapture;
    const first: usize = @intCast(ds.v_pos0);
    const n = position -| first;
    if (n == 0 or n > ds.v_rows or n > 16) return error.ReplayStaleCapture;
    try e.stream.synchronize();
    if (e.side) |side| try side.s.synchronize();
    var r: Reader = .{ .d = e.d };
    var out: Snapshot = std.mem.zeroes(Snapshot);
    out.position = position;
    const pos_view: cuda.DeviceBuffer = .{ .d = e.d, .ptr = st.pos_dev, .len = 4 };
    try pos_view.download(0, std.mem.asBytes(&out.device_position));
    if (out.device_position != position) return error.ReplayCapturePosition;
    out.tap_rows = @intCast(n);
    const logits = if (audit.n == 1) e.bd.logits else e.bs.?.logits;
    for (0..n) |j| {
        var h = Sha.init(.{});
        try r.hash(&h, dr.taps_step + (ds.v_row0 + j) * dflash.TAP_ROW, dflash.TAP_ROW);
        h.final(&out.taps[j]);
        h = Sha.init(.{});
        try r.hash(&h, logits + (ds.v_row0 + j) * e.share.vocab.len() * 2, e.share.vocab.len() * 2);
        h.final(&out.logits[j]);
    }
    if (rows_only) return out;
    const owned = if (st.hold) |hold| hold.owner == e.share.rank else true;
    out.has_latent = @intFromBool(owned);
    var conv = Sha.init(.{});
    try r.hash(&conv, st.conv, st.kda_layers * 3 * st.conv_c * 2);
    conv.final(&out.conv);
    var rec = Sha.init(.{});
    for (st.cur, 0..) |bank, li| try r.hash(&rec, st.recAt(bank, li), st.rec_layer);
    rec.final(&out.recurrent);
    var codes = Sha.init(.{});
    var scales = Sha.init(.{});
    var ik = Sha.init(.{});
    var ig = Sha.init(.{});
    var pooled = Sha.init(.{});
    const pool = @import("cuda_sparse.zig").pool;
    const index_from = position / pool * pool;
    out.index_from = index_from;
    if (index_from < st.base) return error.ReplayCaptureIndex;
    for (st.caches) |c| {
        if (owned) {
            const ids = if (c.lpt != 0) st.lpages else st.pages;
            const stride = if (c.lpt != 0) c.lps else c.ps;
            try r.paged(&codes, c.lc, position, e.c.kv_lora, ids, stride, state.page_rows);
            try r.paged(&scales, c.ls, position, 4, ids, stride, state.page_rows);
        }
        try r.hash(&ik, c.ik + index_from * st.ring_row, (position - index_from) * st.ring_row);
        try r.hash(&ig, c.ig + index_from * st.ring_row, (position - index_from) * st.ring_row);
        try r.paged(&pooled, c.pk, position / pool, e.c.i_dim * 2, st.pages, c.ps, state.page_rows / pool);
    }
    codes.final(&out.latent_codes);
    scales.final(&out.latent_scales);
    ik.final(&out.index_keys);
    ig.final(&out.index_gates);
    pooled.final(&out.pooled_keys);
    var ring = Sha.init(.{});
    const ring_from = position -| @as(usize, @intCast(dflash.WIN));
    const live_from = @max(ring_from, @as(usize, @intCast(@max(ds.from, 0))));
    out.ring_from = live_from;
    const width = dflash.KVW * 2;
    for (0..dflash.LAYERS * 2) |j| for (live_from..position) |pos| {
        const row = pos % @as(usize, @intCast(dflash.WIN));
        try r.hash(&ring, ds.ring.?.ptr + (j * @as(usize, @intCast(dflash.WIN)) + row) * width, width);
    };
    ring.final(&out.ring);
    return out;
}

test "replay logical page digests ignore physical IDs and discard future bytes" {
    const F = struct {
        fn get(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
            const out: [*]u8 = @ptrCast(dst.?);
            const in: [*]const u8 = @ptrFromInt(src);
            @memcpy(out[0..n], in[0..n]);
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyDtoH_v2 = F.get;
    var r: Reader = .{ .d = &d };
    const first = [_]u8{ 3, 4, 1, 2 };
    const second = [_]u8{ 1, 2, 3, 255 };
    var h = Sha.init(.{});
    try r.paged(&h, @intFromPtr(&first), 3, 1, &.{ 1, 0 }, 2, 2);
    var a: Digest = undefined;
    h.final(&a);
    h = Sha.init(.{});
    try r.paged(&h, @intFromPtr(&second), 3, 1, &.{ 0, 1 }, 2, 2);
    var b: Digest = undefined;
    h.final(&b);
    try std.testing.expectEqual(a, b);
    h = Sha.init(.{});
    try std.testing.expectError(error.ReplayCapturePages, r.paged(&h, @intFromPtr(&second), 3, 1, &.{0}, 2, 2));
}

test "replay row capture reads positions taps logits only; full capture reads state and rejects stale ingest" {
    const F = struct {
        var bytes: usize = 0;
        fn get(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
            bytes += n;
            const out: [*]u8 = @ptrCast(dst.?);
            const in: [*]const u8 = @ptrFromInt(src);
            @memcpy(out[0..n], in[0..n]);
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        const Seq = struct { st: state.State };
        const Dr = struct { slots: struct { items: []dflash.Slot }, verifies: u64 = 1, taps_step: u64 };
        const Boundary = struct {
            block: bool = false,
            audit: struct { pending: bool = false, groups: u64 = 1, forwards: u64 = 1, n: usize = 1 } = .{},
        };
        const E = struct {
            d: *const cuda.Driver,
            seqs: struct { items: []?*Seq },
            drafter: ?*Dr,
            replay: ?*Boundary,
            stream: cuda.Stream,
            side: ?struct { s: cuda.Stream } = null,
            share: struct { rank: usize = 0, vocab: @import("cuda_shard.zig").Span = .{ .lo = 0, .hi = 2 } } = .{},
            c: struct { kv_lora: usize = 1, i_dim: usize = 1 } = .{},
            bd: struct { logits: u64 },
            bs: ?struct { logits: u64 } = null,
        };
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemcpyDtoH_v2 = F.get;
    d.api.cuStreamSynchronize = F.sync;
    var position: i32 = 1;
    const taps: [dflash.TAP_ROW]u8 = @splat(7);
    const logits = [_]u16{ 3, 4 };
    const conv = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const rec = [_]u32{ 7, 8 };
    var banks = [_]u1{0};
    var seq: F.Seq = .{ .st = undefined };
    seq.st.pos = 1;
    seq.st.mtp_len = 0;
    seq.st.pos_dev = @intFromPtr(&position);
    seq.st.hold = null;
    seq.st.kda_layers = 1;
    seq.st.conv_c = 1;
    seq.st.conv = @intFromPtr(&conv);
    seq.st.cur = &banks;
    seq.st.rec = @intFromPtr(&rec);
    seq.st.rec_layer = 4;
    seq.st.base = 0;
    seq.st.caches = &.{};
    var seqs = [_]?*F.Seq{&seq};
    var slots = [_]dflash.Slot{.{ .v_step = 1, .v_rows = 1, .v_pos0 = 0, .from = 1 }};
    var dr: F.Dr = .{ .slots = .{ .items = &slots }, .taps_step = @intFromPtr(&taps) };
    var b: F.Boundary = .{};
    var e: F.E = .{ .d = &d, .seqs = .{ .items = &seqs }, .drafter = &dr, .replay = &b, .stream = .{ .d = &d, .handle = null }, .bd = .{ .logits = @intFromPtr(&logits) } };
    F.bytes = 0;
    const light = try captureRows(&e, 0, 1);
    try std.testing.expectEqual(@as(usize, 4 + dflash.TAP_ROW + 4), F.bytes);
    F.bytes = 0;
    const full = try capture(&e, 0, 1);
    try std.testing.expectEqual(@as(usize, 4 + dflash.TAP_ROW + 4 + conv.len + 4), F.bytes);
    try std.testing.expectEqual(light.taps, full.taps);
    try std.testing.expectEqual(light.logits, full.logits);
    try std.testing.expect(!std.meta.eql(light.conv, full.conv));
    b.audit.pending = true;
    try std.testing.expectError(error.ReplayStaleCapture, captureRows(&e, 0, 1));
    b.audit.pending = false;
    dr.verifies += 1;
    try std.testing.expectError(error.ReplayStaleCapture, captureRows(&e, 0, 1));
    dr.verifies -= 1;
    position = 0;
    try std.testing.expectError(error.ReplayCapturePosition, captureRows(&e, 0, 1));
}
