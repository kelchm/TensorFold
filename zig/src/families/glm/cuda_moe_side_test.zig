//! TF_GLM_SHARED_SIDE: setting, side partials sizing, fork/join ordering, DIRECT/L2PF/ROCE interactions, error drain.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const b16 = @import("cuda_b16.zig");
const moe = @import("cuda_moe.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;

fn idOf(comptime name: []const u8) usize {
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |f, i| {
        if (comptime std.mem.eql(u8, f, name)) return i;
    }
    unreachable;
}

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
    const h: [32]u8 = @splat(0); // same bytes either way, so one rank may fork while another runs inline
    var all = [_]ranks.Fingerprint{ ranks.Fingerprint.of(2, 0, off, 8192, true, h, h, h), ranks.Fingerprint.of(2, 1, .{}, 8192, true, h, h, h) };
    try t.expect(ranks.mismatch(&all) == null);
    try t.expectEqual(@import("profile").Class.identical, @import("profile").entry(.TF_GLM_SHARED_SIDE).class);
}

test "the side partials hold both shared matmuls' K slices, glm_mm's and b16's, at every window row count" {
    for ([_]usize{ 2048, 1024, 704, 256 }) |sh| for (1..b16.rows_from) |rows| {
        const have = moe.sidePart(rows, sh, 4096);
        for ([_][2]usize{ .{ 2 * sh, 4096 }, .{ 4096, sh } }) |nk| {
            try t.expect(b16.partBytes(rows, nk[0], nk[1]) <= have);
            try t.expect(mm.partBytes(rows, .{ .w = 0, .n = nk[0], .k = nk[1] }) <= have);
        }
    };
}

/// Every launch, event record, wait and drain in order, with its stream; a matmul's or reduce's partials pointer.
const Log = struct {
    const Kind = enum { launch, record, wait, sync };
    const Op = struct { kind: Kind, stream: usize, ev: usize = 0, f: usize = 0, part: u64 = 0 };
    var ops: [64]Op = undefined;
    var n: usize = 0;
    var refuse_launch: usize = 0;
    var refuse_wait: usize = 0;
    const by_rows = 201;
    const reduce = 205;
    const put_slot = 300;
    const route = 700;

    fn add(op: Op) cuda.abi.Result {
        ops[n] = op;
        n += 1;
        const no = (op.kind == .launch and op.f == refuse_launch) or (op.kind == .wait and op.ev == refuse_wait);
        return if (no) cuda.abi.error_not_ready else cuda.abi.success;
    }
    fn word(p: ?[*]?*anyopaque, i: usize) u64 {
        return @as(*align(1) const u64, @ptrCast(p.?[i].?)).*;
    }
    fn launch(f: usize, s: cuda.abi.Stream, p: ?[*]?*anyopaque) cuda.abi.Result {
        const part: u64 = if (f >= by_rows and f < reduce) word(p, 5) else if (f == reduce) word(p, 0) else 0;
        return add(.{ .kind = .launch, .stream = @intFromPtr(s), .f = f, .part = part });
    }
    fn plain(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, s: cuda.abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return launch(@intFromPtr(f), s, p);
    }
    fn extended(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return launch(@intFromPtr(f), c.stream, p);
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
    xk: exl3.Kernels = undefined,

    fn fake(r: *Rig, h: usize) cuda.Function {
        return .{ .d = &r.d, .handle = @ptrFromInt(h) };
    }

    fn init(r: *Rig) void {
        r.d.api.cuLaunchKernel = Log.plain;
        r.d.api.cuLaunchKernelEx = Log.extended;
        r.d.api.cuEventRecord = Log.record;
        r.d.api.cuStreamWaitEvent = Log.wait;
        r.d.api.cuStreamSynchronize = Log.sync;
        inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |name, i| {
            const T = @FieldType(glue.Kernels, name);
            if (T == cuda.Function or T == ?cuda.Function) @field(r.fns, name) = r.fake(i) else if (T == bool) @field(r.fns, name) = true;
        }
        for (&r.mk.by_rows, 0..) |*f, i| f.* = r.fake(Log.by_rows + i);
        r.mk.reduce = r.fake(Log.reduce);
        r.mk.pdl = true;
        r.mk.prompt_fixed_k = false;
        inline for (@typeInfo(cuda.grouped.Router).@"struct".field_names) |name| {
            if (@FieldType(cuda.grouped.Router, name) == cuda.Function) @field(r.ek.router, name) = r.fake(Log.route);
        }
        Log.n = 0;
        Log.refuse_launch = 0;
        Log.refuse_wait = 0;
    }

    fn kernels(r: *Rig, side: bool) moe.Kernels {
        const sd: moe.Side = .{ .s = .{ .d = &r.d, .handle = @ptrFromInt(side_s) }, .fork = .{ .d = &r.d, .handle = @ptrFromInt(fork_ev) }, .join = .{ .d = &r.d, .handle = @ptrFromInt(join_ev) } };
        return .{ .set = .{ .glue = &r.fns, .mm = &r.mk }, .experts = &r.ek, .exl3 = &r.xk, .put_slot = r.fake(Log.put_slot), .side = if (side) sd else null };
    }

    fn main(r: *Rig) cuda.Stream {
        return .{ .d = &r.d, .handle = @ptrFromInt(main_s) };
    }
};

fn layer(with_exl3: bool) moe.Layer {
    var l: moe.Layer = undefined;
    l.routed.dims = 4096;
    l.routed.width = 1024;
    l.routed.limit = 0;
    l.routed.format = .nvfp4;
    l.tfq3 = null;
    l.exl3 = if (with_exl3) .{ .gt = 1, .ut = 2, .dt = 3, .suh_g = 4, .suh_u = 5, .svh_g = 6, .svh_u = 7, .suh_d = 8, .svh_d = 9, .k2 = 10, .shx = true } else null;
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
    s.x3 = .{ .picks = 0x8000_0000, .xg = 0, .xu = 0, .z = 0 };
    return s;
}

/// The shared expert's launches from `at`: matmul, reduce, SwiGLU, matmul, reduce, plus optional put_slot.
fn expectShared(at: usize, stream: usize, part: u64, with_put: bool) !void {
    const want = [_]usize{ Log.by_rows, Log.reduce, idOf("swiglu_512"), Log.by_rows, Log.reduce, Log.put_slot };
    const len: usize = if (with_put) 6 else 5;
    for (want[0..len], Log.ops[at..][0..len]) |f, op| {
        try t.expectEqual(Log.Kind.launch, op.kind);
        try t.expectEqual(f, op.f);
        try t.expectEqual(stream, op.stream);
        if (f == Log.by_rows or f == Log.reduce) try t.expectEqual(part, op.part);
    }
}

test "TF_GLM_SHARED_SIDE: a decode block forks its shared expert on side_part and joins before combine" {
    var r: Rig = .{};
    r.init();
    const k = r.kernels(true);
    const l = layer(false);
    const s = scratch();
    try t.expect(moe.forks(k, l, s, false) and !moe.forks(k, l, s, true));
    try moe.fork(k, r.main(), l, s, 0x5000_0000, 4096, 5);
    try moe.fill(k, r.main(), l, s, 0x5000_0000, 4096, 0, 5, false, 0x6000_0000);
    try t.expectEqual(@as(usize, 11), Log.n);
    try t.expectEqual(Log.Op{ .kind = .record, .stream = main_s, .ev = fork_ev }, Log.ops[0]);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = side_s, .ev = fork_ev }, Log.ops[1]);
    try expectShared(2, side_s, side_part, true);
    try t.expectEqual(Log.Op{ .kind = .record, .stream = side_s, .ev = join_ev }, Log.ops[8]);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = main_s, .ev = join_ev }, Log.ops[9]);
    try t.expectEqual(idOf("combine"), Log.ops[10].f);
    try t.expectEqual(@as(usize, main_s), Log.ops[10].stream);
}

test "TF_GLM_SHARED_SIDE + TF_GLM_MOE_DIRECT=1: fork skips put_slot and fill joins into combineTail" {
    var r: Rig = .{};
    r.init();
    var k = r.kernels(true);
    k.direct = true;
    const l = layer(true);
    const s = scratch();
    try moe.fork(k, r.main(), l, s, 0x5000_0000, 4096, 5);
    try moe.fill(k, r.main(), l, s, 0x5000_0000, 4096, 0, 5, false, 0x6000_0000);
    try t.expectEqual(@as(usize, 10), Log.n);
    try t.expectEqual(Log.Op{ .kind = .record, .stream = main_s, .ev = fork_ev }, Log.ops[0]);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = side_s, .ev = fork_ev }, Log.ops[1]);
    try expectShared(2, side_s, side_part, false); // no put_slot on side stream under direct
    try t.expectEqual(Log.Op{ .kind = .record, .stream = side_s, .ev = join_ev }, Log.ops[7]);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = main_s, .ev = join_ev }, Log.ops[8]);
    try t.expectEqual(idOf("combine_tail"), Log.ops[9].f);
    try t.expectEqual(@as(usize, main_s), Log.ops[9].stream);
    // partial decode window on a forked block is refused and drains side stream
    r.init();
    try t.expectError(error.BadWindow, moe.fill(k, r.main(), l, s, 0x5000_0000, 4096, 1, 5, false, 0x6000_0000));
    try t.expectEqual(@as(usize, 1), Log.n);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[0]);
}

test "TF_GLM_SHARED_SIDE + TF_GLM_L2PF_SHARED_FIRST=1: forked body never runs sharedRows twice" {
    var r: Rig = .{};
    r.init();
    var k = r.kernels(true);
    k.early_shared = true;
    Log.refuse_launch = Log.route; // stop right after prepare + any inline sharedRows
    try t.expectError(error.NotReady, moe.body(k, r.main(), layer(false), scratch(), 0x5000_0000, 4096, 5, false));
    var main_mm: usize = 0;
    var side_mm: usize = 0;
    for (Log.ops[0..Log.n]) |op| if (op.kind == .launch and op.f == Log.by_rows) {
        if (op.stream == main_s) main_mm += 1 else if (op.stream == side_s) side_mm += 1;
    };
    try t.expectEqual(@as(usize, 0), main_mm);
    try t.expectEqual(@as(usize, 2), side_mm);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[Log.n - 1]);
    // With TF_GLM_SHARED_SIDE=0, early_shared still runs sharedRows inline on main_s using mm_part.
    r.init();
    var k_off = r.kernels(false);
    k_off.early_shared = true;
    Log.refuse_launch = Log.route;
    try t.expectError(error.NotReady, moe.body(k_off, r.main(), layer(false), scratch(), 0x5000_0000, 4096, 5, false));
    main_mm = 0;
    for (Log.ops[0..Log.n]) |op| if (op.kind == .launch and op.f == Log.by_rows) {
        try t.expectEqual(@as(usize, main_s), op.stream);
        try t.expectEqual(mm_part, op.part);
        main_mm += 1;
    };
    try t.expectEqual(@as(usize, 2), main_mm);
}

test "TF_GLM_SHARED_SIDE: without side, without side_part, or in a prompt chunk shared stays inline" {
    var r: Rig = .{};
    for ([_][2]bool{ .{ false, false }, .{ true, true }, .{ true, false } }) |c| {
        r.init();
        const k = r.kernels(c[0]);
        var s = scratch();
        if (c[0] and !c[1]) s.side_part = 0;
        try t.expect(!moe.forks(k, layer(false), s, c[1]));
        try moe.fill(k, r.main(), layer(false), s, 0x5000_0000, 4096, 0, 5, c[1], 0x6000_0000);
        try t.expectEqual(@as(usize, 7), Log.n);
        try expectShared(0, main_s, mm_part, true);
        try t.expectEqual(idOf("combine"), Log.ops[6].f);
    }
}

test "TF_GLM_SHARED_SIDE: TFQ3 routed or shared layers stay inline to avoid Kernels.xrot races" {
    var r: Rig = .{};
    const s = scratch();
    const dw: cuda.tfq3.DenseWeight = undefined;
    // 1. TFQ3 routed layer with bf16 shared projections: never forks; body & fill run inline on main_s.
    r.init();
    const k = r.kernels(true);
    var l_routed = layer(false);
    l_routed.tfq3 = @as(cuda.tfq3.MoeLayer, undefined);
    try t.expect(!moe.forks(k, l_routed, s, false));
    try t.expectError(error.FormatNotLoaded, moe.body(k, r.main(), l_routed, s, 0x5000_0000, 4096, 5, false));
    for (Log.ops[0..Log.n]) |op| try t.expect(op.kind == .launch and op.stream == main_s);
    r.init();
    try moe.fill(k, r.main(), l_routed, s, 0x5000_0000, 4096, 0, 5, false, 0x6000_0000);
    try t.expectEqual(@as(usize, 7), Log.n);
    try expectShared(0, main_s, mm_part, true);
    try t.expectEqual(idOf("combine"), Log.ops[6].f);
    // 2. TFQ3 shared_gu or shared_down (dense or pair): never forks.
    var l_gu = layer(false);
    l_gu.shared_gu = .{ .tfq3 = dw };
    try t.expect(!moe.forks(k, l_gu, s, false));
    l_gu.shared_gu = .{ .tfq3_pair = .{ dw, dw } };
    try t.expect(!moe.forks(k, l_gu, s, false));
    var l_down = layer(false);
    l_down.shared_down = .{ .tfq3 = dw };
    try t.expect(!moe.forks(k, l_down, s, false));
    l_down.shared_down = .{ .tfq3_pair = .{ dw, dw } };
    try t.expect(!moe.forks(k, l_down, s, false));
}

test "TF_GLM_SHARED_SIDE: failures in fork, body, join wait, or combine drain the side stream" {
    var r: Rig = .{};
    // 1. Failure inside fork after side stream wait drains once.
    r.init();
    Log.refuse_launch = Log.put_slot;
    try t.expectError(error.NotReady, moe.body(r.kernels(true), r.main(), layer(false), scratch(), 0x5000_0000, 4096, 5, false));
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[Log.n - 1]);
    var syncs: usize = 0;
    for (Log.ops[0..Log.n]) |op| if (op.kind == .sync) {
        syncs += 1;
    };
    try t.expectEqual(@as(usize, 1), syncs);
    // 2. Failure in body after fork drains once.
    r.init();
    Log.refuse_launch = Log.route;
    try t.expectError(error.NotReady, moe.body(r.kernels(true), r.main(), layer(false), scratch(), 0x5000_0000, 4096, 5, false));
    try t.expectEqual(Log.Op{ .kind = .record, .stream = main_s, .ev = fork_ev }, Log.ops[0]);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[Log.n - 1]);
    // 3. Failure waiting on join in fill drains.
    r.init();
    Log.refuse_wait = join_ev;
    try t.expectError(error.NotReady, moe.fill(r.kernels(true), r.main(), layer(false), scratch(), 0x5000_0000, 4096, 0, 5, false, 0x6000_0000));
    try t.expectEqual(@as(usize, 2), Log.n);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[1]);
    // 4. Failure launching combine after join wait also drains (side stream may still run asynchronously).
    r.init();
    Log.refuse_launch = idOf("combine");
    try t.expectError(error.NotReady, moe.fill(r.kernels(true), r.main(), layer(false), scratch(), 0x5000_0000, 4096, 0, 5, false, 0x6000_0000));
    try t.expectEqual(@as(usize, 3), Log.n);
    try t.expectEqual(Log.Op{ .kind = .sync, .stream = side_s }, Log.ops[2]);
    // 5. No side stream: nothing forked, nothing drained.
    r.init();
    Log.refuse_launch = Log.route;
    try t.expectError(error.NotReady, moe.body(r.kernels(false), r.main(), layer(false), scratch(), 0x5000_0000, 4096, 5, false));
    for (Log.ops[0..Log.n]) |op| try t.expect(op.kind == .launch and op.stream == main_s);
}

test "TF_GLM_SHARED_SIDE + TF_GLM_ROCE_FUSE=1: both coexist and ffn joins before returning to resolve" {
    const Reader = struct {
        fn get(key: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_ROCE_FUSE")) return "1";
            if (std.mem.eql(u8, key, "TF_GLM_SHARED_SIDE")) return "1";
            return null;
        }
    };
    var problem: []const u8 = "";
    const both = try ranks.Settings.read(Reader.get, &problem);
    try t.expect(both.roce_fuse and both.shared_side);
    // ffn with out == null never forks, even on a decode buffer with side_part.
    var r: Rig = .{};
    r.init();
    var k = r.kernels(true);
    var mods: @import("cuda_forward.zig").Modules = undefined;
    mods.glue = r.fns;
    mods.mm = r.mk;
    mods.tiled = true;
    mods.packs = null;
    mods.experts = r.ek;
    mods.exl3 = r.xk;
    mods.tfq3 = null;
    mods.put_slot = r.fake(Log.put_slot);
    mods.moe_direct = false;
    mods.index = null;
    mods.copy_rows = r.fake(301);
    mods.dsel = false;
    mods.dscore = null;
    mods.mla = null;
    mods.lane = undefined;
    var w: @import("cuda_weights.zig").Weights = undefined;
    w.config.hidden = 4096;
    var b: @import("cuda_state.zig").Buffers = undefined;
    b.prefill = false;
    b.normed = 0x5000_0000;
    b.moe = scratch();
    var one = [_]@import("cuda_forward.zig").Rank{.{ .w = &w, .st = undefined, .b = &b }};
    const fw = @import("cuda_forward.zig").Forward{ .d = &r.d, .m = &mods, .stream = r.main(), .ranks = &one, .exchange = .local, .side = k.side };
    var lay: @import("cuda_weights.zig").Layer = undefined;
    lay.ffn = .{ .moe = layer(false) };
    Log.refuse_launch = Log.route;
    try t.expectError(error.NotReady, fw.ffn(one[0], lay, 5, null));
    for (Log.ops[0..Log.n]) |op| try t.expect(op.stream == main_s and op.kind == .launch); // out == null cleared side
    // With out != null, moe.block (body + fill) joins side_s onto main_s and runs combine before returning to resolve.
    r.init();
    k = r.kernels(true);
    try moe.fork(k, r.main(), layer(false), scratch(), 0x5000_0000, 4096, 5);
    try moe.fill(k, r.main(), layer(false), scratch(), 0x5000_0000, 4096, 0, 5, false, 0x6000_0000);
    try t.expectEqual(Log.Op{ .kind = .wait, .stream = main_s, .ev = join_ev }, Log.ops[Log.n - 2]);
    try t.expectEqual(idOf("combine"), Log.ops[Log.n - 1].f);
    try t.expectEqual(@as(usize, main_s), Log.ops[Log.n - 1].stream);
}
