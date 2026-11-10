//! Packed GLM prompt launches; compile resources remain pending until nvcc receipts are collected.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = blk: {
    var out: [16]registry.CudaRecord = undefined;
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
    for (.{ "glm_mm4_128s2f", "glm_mm4_128s2f_ld", "tf_ctl_glm_mm4_128s2f_swap" }, 7..) |symbol, i| { // TF_GLM_MM_FAST4, its control
        const image = if (i == 9) "contract_controls" else "glm_mm";
        out[i] = .{ .launch = .{ .name = if (i == 9) image ++ " " ++ symbol else symbol, .symbol = symbol, .image = image, .threads = 256, .dynamic = out[0].launch.dynamic, .opt_in = true, .min_sm = 80 }, .usage = registry.none, .later_pending = true };
    }
    for (.{ "glm_mm4_128w", "glm_mm4_128w_ld", "glm_mm4_128wb", "glm_mm4_128wb_ld", "tf_ctl_glm_mm4_128w_swap", "tf_ctl_glm_mm4_128wb_swap" }, 10..) |symbol, i| { // TF_GLM_MM_WIDE, its controls
        const image = if (i >= 14) "contract_controls" else "glm_mm";
        const dynamic = out[0].launch.dynamic + if (i == 12 or i == 13 or i == 15) 8 * 4 * 32 * 8 else 0; // _128wb's decoded pairs
        out[i] = .{ .launch = .{ .name = if (i >= 14) image ++ " " ++ symbol else symbol, .symbol = symbol, .image = image, .threads = 256, .dynamic = dynamic, .opt_in = true, .min_sm = 80 }, .usage = registry.none, .later_pending = true };
    }
    break :blk out;
};
