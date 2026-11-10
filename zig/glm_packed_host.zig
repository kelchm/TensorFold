//! Actual GLM load, transport-cap and kernel-catalog boundaries, without loading a GPU.
const std = @import("std");
const glm = @import("glm");
const registry = @import("core").contract.registry;
const t = std.testing;

test "glm packed: invalid RoCE cap text and overflow are refused" {
    for ([_][]const u8{ "", "abc", "-1", "0", "4194304", "18446744073709551615" }) |v|
        try t.expectError(error.BadRoceCap, glm.native.roceBytes(v));
    try t.expectEqual(@as(usize, 4096 << 10), try glm.native.roceBytes(null));
    try t.expectEqual(@as(usize, 1024), try glm.native.roceBytes("1"));
    try t.expectEqual(@as(usize, 4194303) * 1024, try glm.native.roceBytes("4194303"));
}

fn record(symbol: []const u8) !registry.CudaRecord {
    for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) return r;
    return error.MissingGlmLaunch;
}

test "glm packed: every loaded matmul bucket and tiler has a measured resource record" {
    inline for (.{ 16, 4, 6, 8 }) |bits| {
        inline for (.{ 16, 32, 64, 128 }) |bm| {
            const r = try record(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ bits, bm }));
            try t.expectEqualStrings("glm_mm", r.launch.image);
            try t.expectEqual(@as(u32, if (bm <= 32) 128 else 256), r.launch.threads);
            try t.expectEqual(glm.mm.sharedBytes(bits, bm), r.launch.dynamic);
            for (r.usage) |u| try t.expect(u.regs > 0);
        }
        _ = try record(std.fmt.comptimePrint("glm_mm_tile{d}", .{bits}));
    }
    _ = try record("glm_mm16_rows");
    _ = try record("glm_mm_reduce");
}

test "glm packed: two-stage prompt geometry is registered without claiming unmeasured resources" {
    inline for (.{ "glm_mm4_128s2", "glm_mm6_128s2", "glm_mm8_128s2", "glm_mm4_128s2_ld", "glm_mm6_128s2_ld", "glm_mm8_128s2_ld" }, 0..) |symbol, i| {
        const bits: u8 = @intCast(4 + 2 * (i % 3));
        const r = try record(symbol);
        try t.expectEqualStrings("glm_mm", r.launch.image);
        try t.expectEqual(@as(u32, 256), r.launch.threads);
        try t.expectEqual(glm.mm.sharedBytesStages(bits, 128, 2), r.launch.dynamic);
        try t.expect(r.launch.opt_in and !r.geometry_pending);
        try t.expectEqual(@as(usize, 5), try registry.checkCuda(t.allocator, &.{r}, true));
        try t.expectEqual(registry.none, registry.usageFor(r, .{ .major = 13, .minor = 3 }));
    }
    const ld = try record("glm_mm16_rows_ld"); // TF_GLM_SHARED_SLOT: glm_mm16_rows' launch, its usage pending
    const parent = try record("glm_mm16_rows");
    try t.expectEqual(parent.launch.dynamic, ld.launch.dynamic);
    try t.expectEqual(parent.launch.threads, ld.launch.threads);
    try t.expectEqual(registry.none, registry.usageFor(ld, .{ .major = 13, .minor = 3 }));
}
