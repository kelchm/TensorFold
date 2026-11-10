//! TF_GLM_ROUTER_FUSE: which router calls take tf_glm_router_fused, and its launch, on a recording driver.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const pdl = @import("cuda_glue_pdl_test.zig");
const Record = pdl.Record;
const idOf = pdl.idOf;
const t = std.testing;

const x = 1 << 20;
const w = 2 << 20;
const out = 3 << 20;
const part = 4 << 20;

fn call(fns: *glue.Kernels, fuse: bool, prompt: bool, xp: u64, m: usize, ne: usize) !void {
    fns.router_fuse = fuse;
    Record.count = 0;
    const s: cuda.Stream = .{ .d = fns.router_tile.d, .handle = null };
    try glue.router(.{ .glue = fns, .prompt = prompt }, s, xp, 4096, w, out, part, m, 4096, ne);
}

test "TF_GLM_ROUTER_FUSE: a prompt call of a wave of tiles is one launch, experts along x; others keep the slices" {
    var d: cuda.Driver = undefined;
    var fns: glue.Kernels = undefined;
    Record.driver(&d, &fns);
    for ([_][3]usize{ .{ 4096, 288, 9 }, .{ 4096, 128, 4 }, .{ 1024, 288, 9 }, .{ 1000, 288, 9 }, .{ 2304, 128, 4 } }) |c| {
        try call(&fns, true, true, x, c[0], c[1]);
        try t.expectEqual(@as(usize, 1), Record.count);
        const rows: u32 = @intCast((c[0] + 63) / 64);
        try t.expectEqual(Record{ .id = idOf("router_fused"), .grid = .{ @intCast(c[2]), rows, 1 }, .block = 128, .pdl = false }, Record.calls[0]);
    }
    // under a wave (15 x 9, 35 x 4 tiles), decode, the switch off, unaligned rows: router_tile or _part, slice_sum
    const kept = [_]struct { fuse: bool, prompt: bool, xp: u64, m: usize, ne: usize, f: []const u8 }{
        .{ .fuse = true, .prompt = true, .xp = x, .m = 960, .ne = 288, .f = "router_tile" },
        .{ .fuse = true, .prompt = true, .xp = x, .m = 2240, .ne = 128, .f = "router_tile" },
        .{ .fuse = true, .prompt = false, .xp = x, .m = 4096, .ne = 288, .f = "router_tile" },
        .{ .fuse = false, .prompt = true, .xp = x, .m = 4096, .ne = 288, .f = "router_tile" },
        .{ .fuse = true, .prompt = true, .xp = x + 2, .m = 4096, .ne = 288, .f = "router_part" },
    };
    for (kept) |k| {
        try call(&fns, k.fuse, k.prompt, k.xp, k.m, k.ne);
        try t.expectEqual(@as(usize, 2), Record.count);
        inline for (.{ "router_tile", "router_part" }) |name| if (std.mem.eql(u8, name, k.f)) try t.expectEqual(idOf(name), Record.calls[0].id);
        try t.expectEqual(glue.router_ks, Record.calls[0].grid[2]);
        try t.expectEqual(idOf("slice_sum"), Record.calls[1].id);
    }
}
