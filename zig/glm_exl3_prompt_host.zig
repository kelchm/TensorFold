//! The real GLM EXL3 caller and prompt launcher at the driver boundary, without a CUDA device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;
const source = @embedFile("kernels/cuda/exl3_experts.cu");
const decode_source = @embedFile("kernels/cuda/exl3_glm_decode.cuh");

const Record = struct {
    var gate_layout: usize = 0;
    var down_layout: usize = 0;
    var result: u64 = 0;
    var partials: u64 = 0;
    var copies: usize = 0;
    var up: usize = 0;
    var down: usize = 0;
    var decode: usize = 0;
    var pairs: [16]i32 = undefined;
    var experts: [16]i32 = undefined;
    var divisor: [16]i32 = undefined;
    var outputs: [16]u64 = undefined;
    var skip: [16]i32 = undefined;
    var routes: usize = 0;
    var item_calls: usize = 0;
    var item_outputs: [16]u64 = undefined;
    var item_picks: [16]u64 = undefined;
    var item_pairs: [16]i32 = undefined;

    fn reset() void {
        copies = 0;
        gate_layout = 0;
        down_layout = 0;
        result = 0;
        partials = 0;
        up = 0;
        down = 0;
        decode = 0;
        routes = 0;
        item_calls = 0;
    }
    fn arg(comptime T: type, args: ?[*]?*anyopaque, index: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[index].?))).*;
    }
    fn copy(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) c_int {
        copies += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        switch (@intFromPtr(f)) {
            1, 2 => {
                pairs[routes] = arg(i32, args, 1);
                experts[routes] = arg(i32, args, 2);
                routes += 1;
            },
            7 => {
                divisor[up] = arg(i32, args, 13);
                up += 1;
            },
            8 => {
                outputs[down] = arg(u64, args, 6);
                skip[down] = arg(i32, args, 9);
                down += 1;
            },
            10 => {
                t.expectEqual(@as(i32, 16), arg(i32, args, 14)) catch return 1;
                t.expectEqual(@as(i32, 1), arg(i32, args, 19)) catch return 1;
                t.expectEqual(@as(i32, 288), arg(i32, args, 20)) catch return 1;
            },
            14 => {
                item_outputs[item_calls] = arg(u64, args, 3);
                item_picks[item_calls] = arg(u64, args, 1);
                item_pairs[item_calls] = arg(i32, args, 6);
                item_calls += 1;
            },
            30, 31, 33, 34 => {
                gate_layout = @intFromPtr(f);
                partials = arg(u64, args, 6);
            },
            32, 35 => {
                down_layout = @intFromPtr(f);
                result = arg(u64, args, 8);
            },
            9 => decode += 1,
            else => {},
        }
        return 0;
    }
    fn function(d: *cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
};

fn run(rows: usize, prompt: bool, shared: bool, count: usize) !void {
    Record.reset();
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Record.launch;
    d.api.cuLaunchKernelEx = glm.pdl.forwardEx(Record.launch);
    d.api.cuMemcpy2DAsync_v2 = Record.copy;
    var k: glm.exl3.Kernels = undefined;
    inline for (.{ "group", "rot_in", "gateup", "down_combine" }, .{ 9, 5, 13, 14 }) |field, id| @field(k, field) = Record.function(&d, id);
    k.grouped = .{ Record.function(&d, 10), Record.function(&d, 11), Record.function(&d, 12) };
    k.prompt = .{ .rotate = Record.function(&d, 6), .gateup = Record.function(&d, 7), .down = Record.function(&d, 8), .gateup_k6 = Record.function(&d, 21), .down_k6 = Record.function(&d, 22), .gateup_m = @splat(Record.function(&d, 23)), .down_m = @splat(Record.function(&d, 24)), .gateup_glm = Record.function(&d, 7), .down_glm = Record.function(&d, 8) };
    var ek: glm.experts.Kernels = undefined;
    inline for (.{ "plan_small", "plan_rows", "plan_rank", "plan_offsets", "plan_scatter", "plan_rank_rows", "plan_scatter_rows" }, .{ 1, 18, 2, 3, 4, 19, 20 }) |field, id| @field(ek.router, field) = Record.function(&d, id);
    const w = glm.exl3.Experts{ .gt = 101, .ut = 102, .dt = 103, .k2 = 104, .suh_g = 105, .suh_u = 106, .svh_g = 107, .svh_u = 108, .suh_d = 109, .svh_d = 110, .shx = shared };
    const sc = glm.exl3.Scratch{ .picks = 0x100000000, .xg = 0, .xu = 0, .z = 0 };
    const plan = cuda.grouped.Plan{ .members = 201, .items = 202, .counts = 203, .rank = 204, .hist = 205 };
    try glm.exl3.run(&k, &ek, .{ .d = &d, .handle = null }, w, sc, plan, 0x200000000, 512, 0x300000000, 9, rows, count, 512, 256, 7, prompt, 0x400000000, 0);
}

test "GLM EXL3 prompt: a whole pass makes one expert call and no 2D copies" {
    const cap = glm.exl3.prompt_rows;
    for ([_]bool{ false, true }) |shared| for ([_]usize{ 2048, 1, 17, 1024, 4096, cap, cap + 6 }) |rows| {
        try run(rows, true, shared, 288);
        try t.expectEqual(@as(usize, 0), Record.copies);
        const calls = (rows + cap - 1) / cap;
        try t.expectEqual(calls, Record.up);
        try t.expectEqual(calls, Record.down);
        try t.expectEqual(calls, Record.routes);
        try t.expectEqual(@as(usize, 0), Record.decode);
        for (0..calls) |j| {
            const n: usize = @min(rows - j * cap, cap);
            try t.expectEqual(@as(i32, @intCast(n * 9)), Record.pairs[j]);
            try t.expectEqual(@as(i32, 289), Record.experts[j]);
            try t.expectEqual(@as(i32, if (shared) 9 else 1), Record.divisor[j]);
            try t.expectEqual(@as(i32, 288), Record.skip[j]);
            try t.expectEqual(@as(u64, 0x400000000 + j * cap * 9 * 512 * 4), Record.outputs[j]);
        }
    };
}

test "GLM EXL3 prompt: decode item calls keep the shared slot and chunk offsets without copies" {
    for ([_]usize{ 4, 1030 }) |rows| {
        try run(rows, false, false, 288);
        try t.expectEqual(@as(usize, 0), Record.copies);
        try t.expectEqual(@as(usize, 0), Record.decode);
        try t.expectEqual(@as(usize, 0), Record.up);
        try t.expectEqual(@as(usize, 0), Record.down);
        const calls = (rows + 1023) / 1024;
        try t.expectEqual(calls, Record.routes);
        try t.expectEqual(calls, Record.item_calls);
        for (0..calls) |j| {
            const at = j * 1024;
            const n: usize = @min(rows - at, 1024);
            try t.expectEqual(@as(i32, @intCast(n * 9)), Record.item_pairs[j]);
            try t.expectEqual(@as(u64, 0x300000000 + at * 9 * 4), Record.item_picks[j]);
            try t.expectEqual(@as(u64, 0x400000000 + at * 9 * 512 * 4), Record.item_outputs[j]);
        }
    }
}

test "GLM EXL3 prompt: the direct shared-id route refuses an overflowing expert table before launch" {
    try t.expectError(error.Exl3Shape, run(2048, true, true, 1024));
    try t.expectEqual(@as(usize, 0), Record.routes);
    try t.expectEqual(@as(usize, 0), Record.up);
    try t.expectEqual(@as(usize, 0), Record.copies);
}

test "GLM EXL3 prompt: scratch names the rank width and rotation layout" {
    const MiB = 1024 * 1024;
    const D = 4096;
    const W_tp2 = 1024;
    const W_tp1 = 2048;
    for ([_]usize{ 2048, 4096, 8192 }, [_]usize{ 1, 2, 4 }) |rows, scale| {
        try t.expectEqual(@as(usize, 52 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp2, true, true)[0]);
        try t.expectEqual(@as(usize, 88 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp1, true, true)[0]);
        try t.expectEqual(@as(usize, 324 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp2, true, false)[0]);
        try t.expectEqual(@as(usize, 360 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp1, true, false)[0]);
    }
    const cap = glm.exl3.prompt_rows;
    try t.expectEqual(glm.exl3.scratchBytes(cap, D, W_tp2, true, true), glm.exl3.scratchBytes(cap + 6, D, W_tp2, true, true));
    const l: cuda.exl3_experts.Layer = .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = cuda.grouped.max_experts, .dims = D, .width = W_tp2, .codebook = .mcg };
    for ([_]usize{ 1, 4, 64, 1024, 1030 }) |rows| {
        const n: usize = @min(rows, 1024);
        const picks = std.mem.alignForward(usize, n * 8 * 4, 256);
        const need = @max(try cuda.exl3_experts.scratchBytes(l, n, 8), try cuda.exl3_experts.fusedBytes(l, n, 9));
        try t.expectEqual(picks + need, glm.exl3.scratchBytes(rows, D, W_tp2, false, false)[0]);
    }
}

fn guarded(text: []const u8, name: []const u8, payload: []const u8) bool {
    const at = std.mem.indexOf(u8, text, name) orelse return false;
    const tail = text[at..];
    const end = std.mem.indexOf(u8, tail, "\n}") orelse return false;
    const body = tail[0..end];
    const guard = std.mem.indexOf(u8, body, "if (e == skip) return;") orelse return false;
    const load = std.mem.indexOf(u8, body, payload) orelse return false;
    const member = std.mem.indexOf(u8, body, "members[") orelse body.len;
    return guard < @min(load, member);
}

/// Whether kernel `name`'s whole body is the call `call` (the guarded template at one width).
fn forwards(text: []const u8, name: []const u8, call: []const u8) bool {
    const at = std.mem.indexOf(u8, text, name) orelse return false;
    const tail = text[at..];
    const open = std.mem.indexOf(u8, tail, "{\n") orelse return false;
    const end = std.mem.indexOf(u8, tail, "\n}") orelse return false;
    return open < end and std.mem.startsWith(u8, std.mem.trim(u8, tail[open + 2 .. end], " "), call) and std.mem.count(u8, tail[open + 2 .. end], ";") == 1;
}

test "GLM EXL3 prompt: shared-id guards precede payload loads and late-guard controls are caught" {
    try t.expect(guarded(source, "void prompt_gu(", "tgateup<"));
    try t.expect(guarded(source, "void prompt_down(", "members[first"));
    for ([_][2][]const u8{ .{ "tf_exl3_prompt_gu(", "prompt_gu<8>(" }, .{ "tf_exl3_prompt_gu_k6(", "prompt_gu<6>(" }, .{ "tf_exl3_prompt_down(", "prompt_down<8>(" }, .{ "tf_exl3_prompt_down_k6(", "prompt_down<6>(" } }) |k| try t.expect(forwards(source, k[0], k[1]));
    try t.expect(!forwards(source, "tf_exl3_prompt_gu(", "prompt_gu<6>("));
    const bad = try std.mem.replaceOwned(u8, t.allocator, source, "if (e == skip) return;", "const volatile int control_member = members[items[3 * item + 1]];\n    if (e == skip) return;");
    defer t.allocator.free(bad);
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, bad, "if (e == skip) return;"));
    try t.expect(!guarded(bad, "void prompt_gu(", "tgateup<"));
    try t.expect(!guarded(bad, "void prompt_down(", "members[first"));
    const item_guard = std.mem.indexOf(u8, decode_source, "if (item >= total) return;") orelse return error.MissingGuard;
    const item_load = std.mem.indexOf(u8, decode_source, "items[3 * item]") orelse return error.MissingLoad;
    const exp_guard = std.mem.indexOf(u8, decode_source, "if (e >= E) return;") orelse return error.MissingGuard;
    const tbase_load = std.mem.indexOf(u8, decode_source, "const uint32_t* tbase") orelse return error.MissingLoad;
    try t.expect(item_guard < item_load and item_load < exp_guard and exp_guard < tbase_load);
}

test "GLM EXL3 prompt: specialized verification uses native slot stride and optional N-major layout" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Record.launch;
    var router: cuda.grouped.Router = undefined;
    inline for (.{ "plan_small", "plan_rows", "plan_rank", "plan_offsets", "plan_scatter", "plan_rank_rows", "plan_scatter_rows" }, .{ 1, 18, 2, 3, 4, 19, 20 }) |field, id| @field(router, field) = Record.function(&d, id);
    var k: glm.exl3.Kernels = undefined;
    k.prompt.rotate = Record.function(&d, 6);
    k.glm_gateup = Record.function(&d, 30);
    k.glm_gateup_flat = Record.function(&d, 31);
    k.glm_down = Record.function(&d, 32);
    k.glm_gateup_nmajor = Record.function(&d, 33);
    k.glm_gateup_flat_nmajor = Record.function(&d, 34);
    k.glm_down_nmajor = Record.function(&d, 35);
    k.gateup = Record.function(&d, 13);
    const w: glm.exl3.Experts = .{ .banks = .{ 101, 102, 103 }, .gt = 104, .ut = 105, .dt = 106, .k2 = 107, .suh_g = 108, .suh_u = 109, .svh_g = 110, .svh_u = 111, .suh_d = 112, .svh_d = 113, .shx = true };
    const x3: glm.exl3.Scratch = .{ .picks = 0x100000000, .xg = 0, .xu = 0, .z = 0 };
    const plan: cuda.grouped.Plan = .{ .members = 201, .items = 202, .counts = 203, .rank = 204, .hist = 205 };
    for (1..33) |rows| for ([_]usize{ 8, 9 }) |slots| for ([_]bool{ false, true }) |enabled| {
        k.glm_nmajor = enabled;
        Record.reset();
        try t.expect(glm.exl3.eligibleDecode(w, rows, 288, 4096, 1024, false));
        const flat = slots == 8;
        const arena = if (flat) x3.picks + glm.exl3.picksBytes(rows) else x3.picks;
        const out: u64 = if (flat) try glm.exl3.directOut(w, x3, rows, 288, 4096, 1024, 7) else 0x700000000;
        try glm.exl3.runDecode(&k, router, .{ .d = &d, .handle = null }, w, arena, plan, 0x200000000, 4096, if (flat) x3.picks else 0x300000000, slots, rows, 288, 7, out, 0x400000000);
        const reordered = enabled and (rows == 4 or rows >= 8);
        const expected_gate: usize = if (flat) 31 else 30;
        try t.expectEqual(expected_gate + @as(usize, if (reordered) 3 else 0), Record.gate_layout);
        try t.expectEqual(@as(usize, if (reordered) 35 else 32), Record.down_layout);
        try t.expectEqual(rows * slots, @as(usize, @intCast(Record.pairs[0])));
        try t.expectEqual(arena + rows * 4096 * 2, Record.partials);
        const end = Record.partials + 2 * 4 * rows * slots * 1024 * 4;
        try t.expect(end <= x3.picks + glm.exl3.scratchFor(rows, 4096, 1024, false, w)[0]);
        if (flat) try t.expect(end <= out);
        try t.expectEqual(out, Record.result);
        try t.expectEqual(@as(usize, 0), Record.copies);
    };
    try t.expect(!glm.exl3.eligibleDecode(w, 0, 288, 4096, 1024, false));
    try t.expect(!glm.exl3.eligibleDecode(w, 33, 288, 4096, 1024, false));
    try t.expect(!glm.exl3.eligibleDecode(w, 8, 288, 4096, 1024, true));
}

test "GLM EXL3 prompt: prepareGlmDecode resolves decode symbols lazily and runDecode refuses unprepared kernels" {
    const Fake = struct {
        var allow_base = false;
        var allow_nmajor = false;
        fn getFn(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            const s = std.mem.span(name);
            if (std.mem.endsWith(u8, s, "_nmajor")) {
                if (!allow_nmajor) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(33);
                return 0;
            }
            if (!allow_base) return cuda.abi.error_not_found;
            out.* = @ptrFromInt(30);
            return 0;
        }
        fn errStr(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
            out.* = null;
            return 1;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Fake.getFn;
    d.api.cuGetErrorName = Fake.errStr;
    d.api.cuGetErrorString = Fake.errStr;
    d.api.cuLaunchKernel = Record.launch;
    var k: glm.exl3.Kernels = .{ .mod = .{ .d = &d, .handle = null }, .group = undefined, .rot_in = undefined, .grouped = undefined, .gateup = undefined, .down_combine = undefined };
    const router: cuda.grouped.Router = undefined;
    const w: glm.exl3.Experts = .{ .banks = .{ 101, 102, 103 }, .gt = 104, .ut = 105, .dt = 106, .k2 = 107, .suh_g = 108, .suh_u = 109, .svh_g = 110, .svh_u = 111, .suh_d = 112, .svh_d = 113, .shx = true };
    const plan: cuda.grouped.Plan = .{ .members = 201, .items = 202, .counts = 203, .rank = 204, .hist = 205 };
    Record.reset();
    try t.expectError(error.MissingExl3DecodeKernel, glm.exl3.runDecode(&k, router, .{ .d = &d, .handle = null }, w, 0x100000000, plan, 0x200000000, 4096, 0x300000000, 9, 4, 288, 7, 0x700000000, 0x400000000));
    try t.expectEqual(@as(usize, 0), Record.routes);
    try t.expectError(error.MissingExl3DecodeKernel, k.prepareGlmDecode(false));
    try t.expect(!k.glm_decode and k.glm_gateup == null);
    Fake.allow_base = true;
    try k.prepareGlmDecode(false);
    try t.expect(k.glm_decode and !k.glm_nmajor and k.glm_gateup != null and k.glm_gateup_nmajor == null);
    try t.expectError(error.MissingExl3DecodeKernel, k.prepareGlmDecode(true));
    try t.expect(!k.glm_nmajor and k.glm_gateup_nmajor == null);
    Fake.allow_nmajor = true;
    try k.prepareGlmDecode(true);
    try t.expect(k.glm_decode and k.glm_nmajor and k.glm_gateup_nmajor != null and k.glm_down_nmajor != null);
}
