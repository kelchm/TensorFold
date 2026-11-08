//! Prompt MLA selection and arithmetic agreement with recorded CUDA launches, without a device.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const ranks = @import("cuda_ranks.zig");
const t = std.testing;
const Record = struct {
    id: usize,
    grid: [3]u32,
    threads: u32,
    shared: u32,
    args: [8]u64,
    var last: Record = undefined;
    fn launch(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, _: u32, _: u32, shared: u32, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        const id = @intFromPtr(f);
        last = .{ .id = id, .grid = .{ gx, gy, gz }, .threads = bx, .shared = shared, .args = @splat(0) };
        const count: usize = if (id <= 2) 6 else 8;
        for (0..count) |i| last.args[i] = if (id > 2 and i >= 6) @as(*const u32, @ptrCast(@alignCast(args.?[i].?))).* else @as(*const u64, @ptrCast(@alignCast(args.?[i].?))).*;
        return 0;
    }
};

test "GLM MLA: default and explicit off launch identical SIMT arguments; the opt-in keeps its fallbacks" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Record.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var fns: glue.Kernels = undefined;
    fns.absorb_q = .{ .d = &d, .handle = @ptrFromInt(1) };
    fns.expand_v = .{ .d = &d, .handle = @ptrFromInt(2) };
    fns.latent_mma_kn = .{ .d = &d, .handle = @ptrFromInt(3) };
    fns.latent_mma_nk = .{ .d = &d, .handle = @ptrFromInt(4) };
    var set: glue.Set = .{ .glue = &fns, .prompt = true };
    try latent.absorbQ(set, stream, 16, 32, 48, 64, 32, 256, 512);
    const initial = Record.last;
    set.absorb_mma = false;
    try latent.absorbQ(set, stream, 16, 32, 48, 64, 32, 256, 512);
    try t.expectEqualDeep(initial, Record.last);
    try t.expectEqual(@as(usize, 1), initial.id);
    set.absorb_mma = true;
    for ([_]usize{ 1, 8, 63 }) |rows| {
        try latent.absorbQ(set, stream, 16, 32, 48, rows, 32, 256, 512);
        try t.expectEqual(@as(usize, 1), Record.last.id);
    }
    for ([_][3]u64{ .{ 18, 32, 48 }, .{ 16, 34, 48 }, .{ 16, 32, 50 } }) |p| {
        try latent.absorbQ(set, stream, p[0], p[1], p[2], 64, 32, 256, 512);
        try t.expectEqual(@as(usize, 1), Record.last.id);
    }
    set.prompt = false;
    try latent.absorbQ(set, stream, 16, 32, 48, 128, 32, 256, 512);
    try t.expectEqual(@as(usize, 1), Record.last.id);
    set.prompt = true;
    try latent.absorbQ(set, stream, 16, 32, 48, 64, 32, 256, 512);
    try t.expectEqual(@as(usize, 3), Record.last.id);
    try t.expectEqual([3]u32{ 4, 1, 32 }, Record.last.grid);
    try t.expectEqual(@as(u32, 256), Record.last.threads);
    try t.expectEqual(glue.latent_mma_smem[0], Record.last.shared);
    try t.expectEqual([8]u64{ 16, 32 * 256, 32, 48, 32 * 512, 64, 512, 256 }, Record.last.args);
    try latent.expandV(set, stream, 16, 32, 48, 129, 32, 256, 512);
    try t.expectEqual(@as(usize, 4), Record.last.id);
    try t.expectEqual([3]u32{ 2, 2, 32 }, Record.last.grid);
    try t.expectEqual(glue.latent_mma_smem[1], Record.last.shared);
    try t.expectEqual([8]u64{ 16, 32 * 512, 32, 48, 32 * 256, 129, 256, 512 }, Record.last.args);
    try t.expectError(error.LatentShape, latent.absorbQ(set, stream, 16, 32, 48, 64, 32, 128, 512));
}

const Reader = struct {
    var raw: ?[]const u8 = null;
    fn get(name: [:0]const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, "TF_GLM_ABSORB_MMA")) raw else null;
    }
};

test "GLM MLA: strict opt-in parsing, arithmetic metadata and rank agreement" {
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "", "0" }) |raw| {
        Reader.raw = raw;
        try t.expectEqual(@as(u32, 0), (try ranks.Settings.read(Reader.get, &problem)).numerics.bits());
    }
    Reader.raw = "1";
    const on = try ranks.Settings.read(Reader.get, &problem);
    try t.expect(on.numerics.absorb_mma);
    const zero: [32]u8 = @splat(0);
    var all = [_]ranks.Fingerprint{ .of(2, 0, .{}, 8192, true, zero, zero, zero), .of(2, 1, on, 8192, true, zero, zero, zero) };
    try t.expectEqual(ranks.Fingerprint.Field.numerics, ranks.mismatch(&all).?.field);
    all[0] = .of(2, 0, on, 8192, true, zero, zero, zero);
    try t.expect(ranks.mismatch(&all) == null);
    for ([_][]const u8{ "on", "2", "01", " 1 " }) |raw| {
        Reader.raw = raw;
        try t.expectError(error.BadSetting, ranks.Settings.read(Reader.get, &problem));
        try t.expectEqualStrings("TF_GLM_ABSORB_MMA: 0 or 1", problem);
    }
    const entry = @import("profile").entry(.TF_GLM_ABSORB_MMA);
    try t.expectEqual(@import("profile").Class.arithmetic, entry.class);
    try t.expect(entry.stable == null and entry.differs("1") and !entry.differs("0"));
}
