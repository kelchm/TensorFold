//! TF_GLM_DENSE_PRUNE's dense rows and their launches, on a recording driver: no model or GPU execution.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const dsa = @import("cuda_dsa_block.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const t = std.testing;

const Launch = struct {
    var n: usize = 0;
    var ids: [64]usize = undefined;
    var grid: [64][3]c_uint = undefined;
    fn run(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        ids[n] = @intFromPtr(f);
        grid[n] = .{ gx, gy, gz };
        n += 1;
        return 0;
    }
    fn function(out: *cuda.abi.Function, _: cuda.abi.Module, _: [*:0]const u8) callconv(.c) c_int {
        out.* = @ptrFromInt(14); // tf_glm_add_i32
        return 0;
    }
};

fn layer(limit: usize) dsa.Layer {
    var l: dsa.Layer = undefined;
    l.dense_limit = limit;
    l.heads = 32;
    l.lw = 512;
    return l;
}

test "TF_GLM_DENSE_PRUNE: dense attention keeps the rows before the dense limit, all of them without it" {
    const l = layer(2051);
    try t.expectEqual(sparse.width, l.dense_limit); // the first position whose sparse count is nonzero
    for ([_][3]usize{ .{ 0, 2048, 2048 }, .{ 0, 2050, 2050 }, .{ 0, 2051, 2051 }, .{ 0, 2052, 2051 }, .{ 0, 4096, 2051 }, .{ 2048, 8, 3 }, .{ 1000, 2048, 1051 }, .{ 2050, 1, 1 } }) |c| {
        try t.expectEqual(c[2], dsa.denseRows(l, c[0], c[1], true));
        try t.expectEqual(c[1], dsa.denseRows(l, c[0], c[1], false));
    }
    for ([_]bool{ false, true }) |prune| try t.expectEqual(@as(usize, 0), dsa.denseRows(l, 2051, 4096, prune));
    try t.expectEqual(@as(usize, 4096), dsa.denseRows(layer(2048), 0, 4096, true)); // under sparse.width: counts of 0 past it
    try t.expectEqual(@as(usize, 2100), dsa.denseRows(layer(2100), 0, 4096, true));
}

test "TF_GLM_DENSE_PRUNE: the kept rows launch as before, the whole chunk's head block and chunks, 512-row blocks" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Launch.run;
    d.api.cuModuleGetFunction = Launch.function;
    var ks: glue.Kernels = undefined;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names) |name| if (@FieldType(glue.Kernels, name) == cuda.Function) {
        @field(ks, name) = .{ .d = &d, .handle = @ptrFromInt(10) };
    };
    ks.chunks_16.handle = @ptrFromInt(3);
    ks.chunks_32.handle = @ptrFromInt(5);
    ks.merge.handle = @ptrFromInt(7);
    const k: dsa.Kernels = .{ .set = .{ .glue = &ks }, .ops = .{ .d = &d, .handle = null } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var s = std.mem.zeroes(dsa.Scratch);
    s.partials = .{ .po = 1000, .pm = 2000, .pl = 3000 };
    const c: dsa.Caches = .{ .lc = 200, .ik = 800, .ig = 900, .pk = 1000, .pk_rows = 2048 };
    const l = layer(2051);
    for ([_][2]usize{ .{ 0, 2048 }, .{ 0, 2050 }, .{ 0, 2051 }, .{ 0, 2052 }, .{ 0, 4096 }, .{ 2048, 8 } }) |at| for ([_]bool{ false, true }) |prune| {
        const host_pos, const rows = at;
        const kept = dsa.denseRows(l, host_pos, rows, prune);
        const nch = latent.chunksFor(host_pos + rows);
        Launch.n = 0;
        try dsa.denseAttention(k, stream, l, c, s, 300, rows, kept, nch, dsa.prompt_att_rows, 0.125);
        var covered: usize = 0;
        var blocks: usize = 0;
        for (Launch.ids[0..Launch.n], Launch.grid[0..Launch.n]) |id, g| switch (id) {
            3, 5 => {
                try t.expectEqual(@as(usize, if (latent.headBlock(rows) == 32) 5 else 3), id);
                try t.expectEqual(@as(c_uint, @intCast(l.heads / latent.headBlock(rows))), g[1]);
                try t.expectEqual(@as(c_uint, @intCast(nch)), g[2]);
                try t.expectEqual(@as(c_uint, @intCast(@min(dsa.prompt_att_rows, kept - covered))), g[0]);
                covered += g[0];
                blocks += 1;
            },
            7, 14 => {},
            else => return error.UnexpectedLaunch,
        };
        try t.expectEqual(kept, covered);
        try t.expectEqual((kept + dsa.prompt_att_rows - 1) / dsa.prompt_att_rows, blocks);
        try t.expectEqual(3 * blocks - 1, Launch.n); // each block's chunks and merge, and past the first its position
    };
}
