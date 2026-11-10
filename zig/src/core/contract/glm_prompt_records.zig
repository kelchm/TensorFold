//! Packed GLM prompt launches; compile resources remain pending until nvcc receipts are collected.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = blk: {
    var out: [7]registry.CudaRecord = undefined;
    for (.{ 4, 6, 8 }, 0..) |bits, i| for (.{ "", "_ld" }, 0..) |suffix, j| { // _ld: TF_GLM_SHARED_SLOT's strided rows
        const symbol = std.fmt.comptimePrint("glm_mm{d}_128s2{s}", .{ bits, suffix });
        out[2 * i + j] = .{ .launch = .{
            .name = symbol,
            .symbol = symbol,
            .image = "glm_mm",
            .threads = 256,
            .dynamic = 2 * (128 * 128 + 8 * 32 * (bits / 2) * 4 + 128),
            .opt_in = true,
            .min_sm = 80,
        }, .usage = registry.none, .later_pending = true };
    };
    out[6] = .{ .launch = .{ .name = "glm_mm16_rows_ld", .symbol = "glm_mm16_rows_ld", .image = "glm_mm", .threads = 256, .dynamic = 74112, .opt_in = true, .min_sm = 80 }, .usage = registry.none, .later_pending = true };
    break :blk out;
};
