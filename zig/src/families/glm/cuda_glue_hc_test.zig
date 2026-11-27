//! Host checks for TF_GLM_HC_PART and TF_GLM_HC_SPLIT kernel selection, grids, legacy fallback and PDL.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const cuda_pdl = @import("cuda_pdl.zig");
const t = std.testing;

const Rec = struct {
    id: usize,
    grid: [3]u32,
    block: u32,
    pdl: bool,

    var calls: [8]Rec = undefined;
    var count: usize = 0;

    fn add(f: cuda.abi.Function, grid: [3]u32, block: u32, pdl: bool) cuda.abi.Result {
        calls[count] = .{ .id = @intFromPtr(f), .grid = grid, .block = block, .pdl = pdl };
        count += 1;
        return 0;
    }
    fn plain(f: cuda.abi.Function, gx: u32, gy: u32, gz: u32, bx: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        return add(f, .{ gx, gy, gz }, bx, false);
    }
    fn ex(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        var pdl = false;
        for (c.attrs.?[0..c.num_attrs]) |a| pdl = pdl or (a.id == .programmatic_stream_serialization and a.value.int == 1);
        return add(f, .{ c.grid_x, c.grid_y, c.grid_z }, c.block_x, pdl);
    }
    fn setup(d: *cuda.Driver, fns: *glue.Kernels) void {
        d.* = undefined;
        d.api.cuLaunchKernel = plain;
        d.api.cuLaunchKernelEx = ex;
        inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |name, i| {
            const T = @FieldType(glue.Kernels, name);
            if (T == cuda.Function or T == ?cuda.Function) @field(fns, name) = .{ .d = d, .handle = @ptrFromInt(i) } else if (T == bool) @field(fns, name) = false;
        }
        count = 0;
    }
};

fn idOf(comptime name: []const u8) usize {
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |f, i| {
        if (comptime std.mem.eql(u8, f, name)) return i;
    }
    unreachable;
}

test "GLM decode glue: TF_GLM_HC_PART and TF_GLM_HC_SPLIT pick decode windows' kernels and fallback when optional symbols are null" {
    var d: cuda.Driver = undefined;
    var fns: glue.Kernels = undefined;
    Rec.setup(&d, &fns);
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const want_pdl = cuda_pdl.on();
    for ([_]bool{ false, true }) |part| for ([_]bool{ false, true }) |split| for ([_]bool{ false, true }) |f32w| for ([_]usize{ 1, 16, 17, 63, 64 }) |rows| {
        fns.hc_part = part;
        fns.hc_split = split;
        Rec.count = 0;
        try glue.hcPre(.{ .glue = &fns }, s, 1 << 20, 2 << 20, 3 << 20, 4 << 20, if (f32w) .{ .f32 = 5 << 20 } else .{ .bf16 = 5 << 20 }, 6 << 20, 7 << 20, 8 << 20, 9 << 20, 10 << 20, 1e-6, 1e-6, 20, rows, 4096, 4);
        try t.expectEqual(@as(usize, 2), Rec.count);
        const o2 = part and rows <= glue.hc_part_rows;
        const p = Rec.calls[0];
        try t.expectEqual(if (o2) idOf("hc_partial8_o2") else if (rows < glue.hc_prompt_rows) idOf("hc_partial8") else idOf("hc_partial"), p.id);
        if (o2) try t.expectEqual(Rec{ .id = p.id, .grid = .{ @intCast(rows), glue.hc_blocks, 13 }, .block = 256, .pdl = want_pdl }, p);
        const cut = split and rows < glue.hc_prompt_rows;
        const want = if (f32w) (if (cut) idOf("hc_finish_f32w_split") else idOf("hc_finish_f32w")) else if (cut) idOf("hc_finish_split") else idOf("hc_finish");
        try t.expectEqual(Rec{ .id = want, .grid = .{ @intCast(rows), 1, 1 }, .block = 256, .pdl = want_pdl }, Rec.calls[1]);
    };
    // Legacy fatbins without the new symbols fall back cleanly to hc_partial8 and hc_finish(_f32w).
    fns.hc_part = true;
    fns.hc_split = true;
    fns.hc_partial8_o2 = null;
    fns.hc_finish_split = null;
    fns.hc_finish_f32w_split = null;
    Rec.count = 0;
    try glue.hcPre(.{ .glue = &fns }, s, 1 << 20, 2 << 20, 3 << 20, 4 << 20, .{ .bf16 = 5 << 20 }, 6 << 20, 7 << 20, 8 << 20, 9 << 20, 10 << 20, 1e-6, 1e-6, 20, 4, 4096, 4);
    try t.expectEqual(idOf("hc_partial8"), Rec.calls[0].id);
    try t.expectEqual(idOf("hc_finish"), Rec.calls[1].id);
    Rec.count = 0;
    try glue.hcPre(.{ .glue = &fns }, s, 1 << 20, 2 << 20, 3 << 20, 4 << 20, .{ .f32 = 5 << 20 }, 6 << 20, 7 << 20, 8 << 20, 9 << 20, 10 << 20, 1e-6, 1e-6, 20, 4, 4096, 4);
    try t.expectEqual(idOf("hc_finish_f32w"), Rec.calls[1].id);
}
