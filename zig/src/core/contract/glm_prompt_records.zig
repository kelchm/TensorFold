//! Packed GLM prompt launches; compile resources remain pending until nvcc receipts are collected.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = blk: {
    var out: [3]registry.CudaRecord = undefined;
    for (.{ 4, 6, 8 }, 0..) |bits, i| {
        const symbol = std.fmt.comptimePrint("glm_mm{d}_128s2", .{bits});
        out[i] = .{ .launch = .{
            .name = symbol,
            .symbol = symbol,
            .image = "glm_mm",
            .threads = 256,
            .dynamic = 2 * (128 * 128 + 8 * 32 * (bits / 2) * 4 + 128),
            .opt_in = true,
            .min_sm = 80,
        }, .usage = registry.none, .later_pending = true };
    }
    break :blk out;
};
