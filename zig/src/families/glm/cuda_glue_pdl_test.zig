//! A decode window's glue launches: which kernel, and whether a dependent (TF_GLM_PDL_CHAIN), without a device.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const t = std.testing;

/// Each launch's kernel id, grid, block and whether it carried the programmatic-serialization attribute.
pub const Record = struct {
    id: usize,
    grid: [3]u32,
    block: u32,
    pdl: bool,
    pub var calls: [16]Record = undefined;
    pub var count: usize = 0;
    fn add(f: cuda.abi.Function, grid: [3]u32, block: u32, pdl: bool) cuda.abi.Result {
        calls[count] = .{ .id = @intFromPtr(f), .grid = grid, .block = block, .pdl = pdl };
        count += 1;
        return cuda.abi.success;
    }
    fn plain(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return add(f, .{ gx, gy, gz }, bx, false);
    }
    fn extended(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        var pdl = false;
        for (c.attrs.?[0..c.num_attrs]) |a| pdl = pdl or (a.id == .programmatic_stream_serialization and a.value.int == 1);
        return add(f, .{ c.grid_x, c.grid_y, c.grid_z }, c.block_x, pdl);
    }
    /// A driver that records; `fns` gets every glue function as its field index + 1, the split schedules off.
    pub fn driver(d: *cuda.Driver, fns: *glue.Kernels) void {
        d.api.cuLaunchKernel = plain;
        d.api.cuLaunchKernelEx = extended;
        const info = @typeInfo(glue.Kernels).@"struct";
        inline for (info.field_names, info.field_types, 1..) |name, T, i| {
            if (T == cuda.Function) @field(fns, name) = .{ .d = d, .handle = @ptrFromInt(i) };
        }
        fns.hc_split = false;
        fns.hc_part = false;
        count = 0;
    }
};

pub fn idOf(comptime name: []const u8) usize {
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    unreachable;
}

fn hcPre(set: glue.Set, s: cuda.Stream, rows: usize) !void {
    return hcPreBy(set, s, .{ .bf16 = 5 << 20 }, rows);
}

fn hcPreBy(set: glue.Set, s: cuda.Stream, norm: glue.Norm, rows: usize) !void {
    try glue.hcPre(set, s, 1 << 20, 2 << 20, 3 << 20, 4 << 20, norm, 6 << 20, 7 << 20, 8 << 20, 9 << 20, 10 << 20, 1e-6, 1e-6, 20, rows, 4096, 4);
}

test "GLM decode glue: TF_GLM_PDL_CHAIN's words are the startup check's, each naming its parts" {
    const words = @import("sys").environment.rule("TF_GLM_PDL_CHAIN").words;
    try t.expectEqual(mm.Chain.words.len, words.len);
    for (words, mm.Chain.words, 0..) |w, ours, i| {
        try t.expectEqualStrings(ours, w);
        const c = mm.Chain.parse(w).?;
        try t.expectEqual(mm.Chain.of(i), c);
        if (i < 8) try t.expectEqual(i, @as(usize, @intFromBool(c.reduce)) | @as(usize, @intFromBool(c.hc)) << 1 | @as(usize, @intFromBool(c.moe)) << 2);
    }
    try t.expectEqual(mm.Chain.all, mm.Chain.parse(words[words.len - 1]).?);
}

test "GLM decode glue: hc_pre and hc_post launch as dependents only with pdl_hc and only for decode rows" {
    var d: cuda.Driver = undefined;
    var fns: glue.Kernels = undefined;
    Record.driver(&d, &fns);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |on| for ([_]usize{ 1, 5, 63, 64, 70 }) |rows| {
        Record.count = 0;
        try hcPre(.{ .glue = &fns, .pdl_hc = on }, s, rows);
        try t.expectEqual(@as(usize, 2), Record.count);
        const decode = rows < 64;
        try t.expectEqual(if (decode) idOf("hc_partial8") else idOf("hc_partial"), Record.calls[0].id);
        try t.expectEqual(idOf("hc_finish"), Record.calls[1].id);
        for (Record.calls[0..2]) |c| try t.expectEqual(on and decode, c.pdl);
        Record.count = 0;
        try glue.hcPost(.{ .glue = &fns, .pdl_hc = on }, s, 1 << 20, 1 << 20, 2 << 20, 3 << 20, 4 << 20, rows, 4096, 4, 2);
        try t.expectEqual(Record{ .id = idOf("hc_post"), .grid = .{ @intCast(rows), 4, 1 }, .block = 128, .pdl = on }, Record.calls[0]);
    };
}

test "GLM decode glue: TF_GLM_HC_PART and TF_GLM_HC_SPLIT pick decode windows' kernels, dependents as the plain ones" {
    var d: cuda.Driver = undefined;
    var fns: glue.Kernels = undefined;
    Record.driver(&d, &fns);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |part| for ([_]bool{ false, true }) |split| for ([_]bool{ false, true }) |f32w| for ([_]usize{ 1, 16, 17, 63, 64 }) |rows| {
        fns.hc_part = part;
        fns.hc_split = split;
        Record.count = 0;
        try hcPreBy(.{ .glue = &fns, .pdl_hc = true }, s, if (f32w) .{ .f32 = 5 << 20 } else .{ .bf16 = 5 << 20 }, rows);
        const o2 = part and rows <= glue.hc_part_rows;
        const p = Record.calls[0];
        try t.expectEqual(if (o2) idOf("hc_partial8_o2") else if (rows < 64) idOf("hc_partial8") else idOf("hc_partial"), p.id);
        if (o2) try t.expectEqual(Record{ .id = p.id, .grid = .{ @intCast(rows), glue.hc_blocks, 13 }, .block = 256, .pdl = true }, p);
        const cut = split and rows < 64;
        const want = if (f32w) (if (cut) idOf("hc_finish_f32w_split") else idOf("hc_finish_f32w")) else if (cut) idOf("hc_finish_split") else idOf("hc_finish");
        try t.expectEqual(Record{ .id = want, .grid = .{ @intCast(rows), 1, 1 }, .block = 256, .pdl = rows < 64 }, Record.calls[1]);
    };
}

test "GLM decode glue: the router's sums and the pick launch as dependents only with pdl_moe; router_tile never" {
    var d: cuda.Driver = undefined;
    var fns: glue.Kernels = undefined;
    Record.driver(&d, &fns);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]bool{ false, true }) |on| for ([_]usize{ 1, 9, 32 }) |rows| {
        Record.count = 0;
        const set: glue.Set = .{ .glue = &fns, .pdl_moe = on };
        try glue.router(set, s, 1 << 20, 4096, 2 << 20, 3 << 20, 4 << 20, rows, 4096, 256);
        try glue.select(set, s, 3 << 20, 5 << 20, 6 << 20, 7 << 20, rows, 8, 256, 2.5, true);
        try t.expectEqual(@as(usize, 3), Record.count);
        try t.expectEqual(Record{ .id = idOf("router_tile"), .grid = .{ 1, 8, glue.router_ks }, .block = 128, .pdl = false }, Record.calls[0]);
        try t.expectEqual(idOf("slice_sum"), Record.calls[1].id);
        try t.expectEqual(idOf("topk"), Record.calls[2].id);
        for (Record.calls[1..3]) |c| try t.expectEqual(on, c.pdl);
    };
}
