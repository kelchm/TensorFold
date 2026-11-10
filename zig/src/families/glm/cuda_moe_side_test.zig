//! TF_GLM_SHARED_SIDE: its setting, the side partials' size, and a decode block's fork and join, without a device.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const b16 = @import("cuda_b16.zig");
const moe = @import("cuda_moe.zig");
const experts = @import("cuda_experts.zig");
const ranks = @import("cuda_ranks.zig");
const idOf = @import("cuda_glue_pdl_test.zig").idOf;
const t = std.testing;

test "TF_GLM_SHARED_SIDE: on by default, 0 or 1, refused otherwise, identical, and outside the ranks' fingerprint" {
    const Reader = struct {
        var raw: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_SHARED_SIDE")) raw else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "", "1" }) |raw| {
        Reader.raw = raw;
        try t.expect((try ranks.Settings.read(Reader.get, &problem)).shared_side);
    }
    Reader.raw = "0";
    const off = try ranks.Settings.read(Reader.get, &problem);
    try t.expect(!off.shared_side);
    for ([_][]const u8{ "on", "2", " 0 " }) |raw| {
        Reader.raw = raw;
        try t.expectError(error.BadSetting, ranks.Settings.read(Reader.get, &problem));
        try t.expectEqualStrings("TF_GLM_SHARED_SIDE: 0 or 1", problem);
    }
    const h: [32]u8 = @splat(0); // the same bytes either way, so one rank may fork while another does not
    var all = [_]ranks.Fingerprint{ ranks.Fingerprint.of(2, 0, off, 8192, true, h, h, h), ranks.Fingerprint.of(2, 1, .{}, 8192, true, h, h, h) };
    try t.expect(ranks.mismatch(&all) == null);
    try t.expectEqual(@import("profile").Class.identical, @import("profile").entry(.TF_GLM_SHARED_SIDE).class);
}

test "the side partials hold both shared matmuls' K slices, glm_mm's and b16's, at every window row count" {
    for ([_]usize{ 2048, 1024, 704, 256 }) |sh| for (1..b16.rows_from) |rows| {
        const have = @import("cuda_state.zig").sidePart(rows, sh, 4096);
        for ([_][2]usize{ .{ 2 * sh, 4096 }, .{ 4096, sh } }) |nk| {
            try t.expect(b16.partBytes(rows, nk[0], nk[1]) <= have);
            try t.expect(mm.partBytes(rows, .{ .w = 0, .n = nk[0], .k = nk[1] }) <= have);
        }
    };
}

/// Every launch, event record, wait and drain in order, with its stream; a matmul's or reduce's partials pointer.
const Log = struct {
    const Kind = enum { launch, record, wait, sync };
    const Op = struct { kind: Kind, stream: usize, ev: usize = 0, f: usize = 0, pdl: bool = false, part: u64 = 0 };
    var ops: [64]Op = undefined;
    var n: usize = 0;
    var refuse: usize = 0; // a launch of this kernel, or a wait on this event, is refused (not ready: no error log)
    const by_rows = 201;
    const reduce = 205;
    const put_slot = 300;
    const route = 700;

    fn add(op: Op) cuda.abi.Result {
        ops[n] = op;
        n += 1;
        const no = (op.kind == .launch and op.f == refuse) or (op.kind == .wait and op.ev == refuse);
        return if (no) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn word(p: ?[*]?*anyopaque, i: usize) u64 {
        return @as(*align(1) const u64, @ptrCast(p.?[i].?)).*;
    }
    fn launch(f: usize, s: cuda.abi.Stream, pdl: bool, p: ?[*]?*anyopaque) cuda.abi.Result {
        const part: u64 = if (f >= by_rows and f < reduce) word(p, 5) else if (f == reduce) word(p, 0) else 0;
        return add(.{ .kind = .launch, .stream = @intFromPtr(s), .f = f, .pdl = pdl, .part = part });
    }
    fn plain(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, s: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return launch(@intFromPtr(f), s, false, p);
    }
    fn extended(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        var pdl = false;
        for (c.attrs.?[0..c.num_attrs]) |a| pdl = pdl or (a.id == .programmatic_stream_serialization and a.value.int == 1);
        return launch(@intFromPtr(f), c.stream, pdl, p);
    }
    fn record(e: cuda.abi.Event, s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return add(.{ .kind = .record, .stream = @intFromPtr(s), .ev = @intFromPtr(e) });
    }
    fn wait(s: cuda.abi.Stream, e: cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        return add(.{ .kind = .wait, .stream = @intFromPtr(s), .ev = @intFromPtr(e) });
    }
    fn sync(s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return add(.{ .kind = .sync, .stream = @intFromPtr(s) });
    }
};

const main_s = 1;
const side_s = 2;
const fork_ev = 11;
const join_ev = 12;
const mm_part: u64 = 0x7000_0000;
const side_part: u64 = 0x7800_0000;

const Rig = struct {
    d: cuda.Driver = undefined,
    fns: glue.Kernels = undefined,
    mk: mm.Kernels = undefined,
    ek: experts.Kernels = undefined,

    fn fake(r: *Rig, h: usize) cuda.Function {
        return .{ .d = &r.d, .handle = @ptrFromInt(h) };
    }

    /// glm_mm with its dependents and TF_GLM_PDL_CHAIN's reduce, glue ids as cuda_glue_pdl_test's, routes refusable.
    fn init(r: *Rig) void {
        r.d.api.cuLaunchKernel = Log.plain;
        r.d.api.cuLaunchKernelEx = Log.extended;
        r.d.api.cuEventRecord = Log.record;
        r.d.api.cuStreamWaitEvent = Log.wait;
        r.d.api.cuStreamSynchronize = Log.sync;
        const info = @typeInfo(glue.Kernels).@"struct";
        inline for (info.field_names, info.field_types, 1..) |name, T, i| {
            if (T == cuda.Function) @field(r.fns, name) = r.fake(i);
        }
        for (&r.mk.by_rows, 0..) |*f, i| f.* = r.fake(Log.by_rows + i);
        r.mk.reduce = r.fake(Log.reduce);
        r.mk.pdl = true;
        r.mk.chain = mm.Chain.all;
        r.mk.prompt_fixed_k = false;
        inline for (@typeInfo(cuda.grouped.Router).@"struct".field_names) |name| {
            if (@TypeOf(@field(r.ek.router, name)) == cuda.Function) @field(r.ek.router, name) = r.fake(Log.route);
        }
        Log.n = 0;
        Log.refuse = 0;
    }

    fn kernels(r: *Rig, side: bool) moe.Kernels {
        const sd: moe.Side = .{ .s = .{ .d = &r.d, .handle = @ptrFromInt(side_s) }, .fork = .{ .d = &r.d, .handle = @ptrFromInt(fork_ev) }, .join = .{ .d = &r.d, .handle = @ptrFromInt(join_ev) } };
        return .{ .set = .{ .glue = &r.fns, .mm = &r.mk, .pdl_hc = true, .pdl_moe = true }, .experts = &r.ek, .exl3 = undefined, .put_slot = r.fake(Log.put_slot), .side = if (side) sd else null };
    }

    fn main(r: *Rig) cuda.Stream {
        return .{ .d = &r.d, .handle = @ptrFromInt(main_s) };
    }
};

fn layer() moe.Layer {
    var l: moe.Layer = undefined;
    l.routed.dims = 4096;
    l.routed.limit = 0;
    l.exl3 = null;
    l.shared_gu = .{ .bf16 = 0x1000_0000 };
    l.shared_down = .{ .bf16 = 0x2000_0000 };
    l.shared = 2048; // [gate | up] and down: four K slices each, so both reduce
    l.top_k = 8;
    l.count = 256;
    l.router = 0x3000_0000;
    l.bias = 0x3100_0000;
    l.scale = 2.5;
    l.norm = true;
    return l;
}

fn scratch() moe.Scratch {
    var s: moe.Scratch = undefined;
    inline for (.{ "mlog", "router_part", "pick", "wts", "eact", "ey", "sgu", "sact", "sxs", "sy" }, 1..) |name, i| @field(s, name) = 0x4000_0000 + i * 0x100_0000;
    s.mm_part = mm_part;
    s.side_part = side_part;
    return s;
}

/// The shared expert's six launches from `at`: matmul, reduce, SwiGLU, matmul, reduce, put_slot, all on `stream`.
fn expectShared(at: usize, stream: usize, part: u64, reduce_pdl: bool) !void {
    const want = [_]usize{ Log.by_rows, Log.reduce, idOf("swiglu_512"), Log.by_rows, Log.reduce, Log.put_slot };
    for (want, Log.ops[at..][0..6]) |f, op| {
        try t.expectEqual(Log.Kind.launch, op.kind);
        try t.expectEqual(f, op.f);
        try t.expectEqual(stream, op.stream);
        if (f == Log.by_rows) try t.expect(op.pdl); // glm_mm's own dependents everywhere
        if (f == Log.reduce) try t.expectEqual(reduce_pdl, op.pdl);
        if (f == Log.by_rows or f == Log.reduce) try t.expectEqual(part, op.part);
    }
}

test "TF_GLM_SHARED_SIDE: a decode block forks its shared expert on the side partials and joins it before the combine" {
    var r: Rig = .{};
    r.init();
    const k = r.kernels(true);
    const l = layer();
    const s = scratch();
    try t.expect(moe.forks(k, s, false) and !moe.forks(k, s, true));
    try moe.fork(k, r.main(), l, s, 0x5000_0000, 4096, 5);
    try moe.fill(k, r.main(), l, s, 0x5000_0000, 4096, 0, 5, false, 0x6000_0000);
    try t.expectEqual(@as(usize, 11), Log.n);
    try t.expectEqual(Log.Op{ .kind = .record, .stream = main_s, .ev = fork_ev }, Log.ops[0]);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = side_s, .ev = fork_ev }, Log.ops[1]);
    try expectShared(2, side_s, side_part, false); // no TF_GLM_PDL_CHAIN reduce on the side stream
    try t.expectEqual(Log.Op{ .kind = .record, .stream = side_s, .ev = join_ev }, Log.ops[8]);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = main_s, .ev = join_ev }, Log.ops[9]);
    try t.expectEqual(idOf("combine"), Log.ops[10].f);
    try t.expectEqual(@as(usize, main_s), Log.ops[10].stream);
}

test "TF_GLM_SHARED_SIDE: without a side stream, without side partials, or in a prompt chunk the shared expert stays in line" {
    var r: Rig = .{};
    for ([_][2]bool{ .{ false, false }, .{ true, true }, .{ true, false } }) |c| {
        r.init();
        const k = r.kernels(c[0]);
        var s = scratch();
        if (c[0] and !c[1]) s.side_part = 0;
        try t.expect(!moe.forks(k, s, c[1]));
        try moe.fill(k, r.main(), layer(), s, 0x5000_0000, 4096, 0, 5, c[1], 0x6000_0000);
        try t.expectEqual(@as(usize, 7), Log.n);
        try expectShared(0, main_s, mm_part, true);
        try t.expectEqual(idOf("combine"), Log.ops[6].f);
    }
}

test "TF_GLM_SHARED_SIDE: a block that fails after its fork, or a join refused, drains the side stream before returning" {
    var r: Rig = .{};
    r.init();
    Log.refuse = Log.route; // the routed experts' plan, after the router and the pick
    try t.expectError(error.NotReady, moe.body(r.kernels(true), r.main(), layer(), scratch(), 0x5000_0000, 4096, 5, false));
    try t.expectEqual(Log.Op{ .kind = .record, .stream = main_s, .ev = fork_ev }, Log.ops[0]);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[Log.n - 1]);
    r.init();
    Log.refuse = join_ev;
    try t.expectError(error.NotReady, moe.fill(r.kernels(true), r.main(), layer(), scratch(), 0x5000_0000, 4096, 0, 5, false, 0x6000_0000));
    try t.expectEqual(@as(usize, 2), Log.n);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[1]);
    r.init();
    Log.refuse = Log.route; // no side stream: nothing forked, nothing to drain
    try t.expectError(error.NotReady, moe.body(r.kernels(false), r.main(), layer(), scratch(), 0x5000_0000, 4096, 5, false));
    for (Log.ops[0..Log.n]) |op| try t.expect(op.kind == .launch and op.stream == main_s);
}
