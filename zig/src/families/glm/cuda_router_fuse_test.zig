//! TF_GLM_ROUTER_FUSE: which router calls take tf_glm_router_fused, and its launch, on a recording driver.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const cuda_pdl = @import("cuda_pdl.zig");
const t = std.testing;

const x = 1 << 20;
const w = 2 << 20;
const out = 3 << 20;
const part = 4 << 20;

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

fn call(fns: *glue.Kernels, fuse: bool, prompt: bool, xp: u64, m: usize, ne: usize) !void {
    fns.router_fuse = fuse;
    Rec.count = 0;
    const s: cuda.Stream = .{ .d = fns.router_tile.d, .handle = null };
    try glue.router(.{ .glue = fns, .prompt = prompt }, s, xp, 4096, w, out, part, m, 4096, ne);
}

test "TF_GLM_ROUTER_FUSE: a prompt call of a wave of tiles is one launch, experts along x; others keep the slices" {
    var d: cuda.Driver = undefined;
    var fns: glue.Kernels = undefined;
    Rec.setup(&d, &fns);
    const want_pdl = cuda_pdl.on();
    for ([_][3]usize{ .{ 4096, 288, 9 }, .{ 4096, 128, 4 }, .{ 1024, 288, 9 }, .{ 1000, 288, 9 }, .{ 2304, 128, 4 } }) |c| {
        try call(&fns, true, true, x, c[0], c[1]);
        try t.expectEqual(@as(usize, 1), Rec.count);
        const rows: u32 = @intCast((c[0] + 63) / 64);
        try t.expectEqual(Rec{ .id = idOf("router_fused"), .grid = .{ @intCast(c[2]), rows, 1 }, .block = 128, .pdl = want_pdl }, Rec.calls[0]);
    }
    // under a wave (15 x 9, 35 x 4 tiles), decode, the switch off: router_tile then slice_sum
    const kept = [_]struct { fuse: bool, prompt: bool, m: usize, ne: usize }{
        .{ .fuse = true, .prompt = true, .m = 960, .ne = 288 },
        .{ .fuse = true, .prompt = true, .m = 2240, .ne = 128 },
        .{ .fuse = true, .prompt = false, .m = 4096, .ne = 288 },
        .{ .fuse = false, .prompt = true, .m = 4096, .ne = 288 },
    };
    for (kept) |k| {
        try call(&fns, k.fuse, k.prompt, x, k.m, k.ne);
        try t.expectEqual(@as(usize, 2), Rec.count);
        try t.expectEqual(idOf("router_tile"), Rec.calls[0].id);
        try t.expectEqual(glue.router_ks, Rec.calls[0].grid[2]);
        try t.expectEqual(idOf("slice_sum"), Rec.calls[1].id);
    }
    try t.expectError(error.RowShape, call(&fns, true, true, x + 2, 4096, 288));
}
