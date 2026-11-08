//! MLA launch geometry; every compile-resource receipt is pending until measured with nvcc.
const registry = @import("registry.zig");
pub const records = blk: {
    var out: [4]registry.CudaRecord = undefined;
    for ([_]u32{ 16, 32 }, 0..) |hb, i| {
        for ([_][]const u8{ "old", "bad" }, 0..) |suffix, j| {
            const symbol = @import("std").fmt.comptimePrint("tf_ctl_glm_chunks_{d}_fp8_{s}", .{ hb, suffix });
            out[i * 2 + j] = .{ .launch = .{ .name = "contract_controls " ++ symbol, .symbol = symbol, .image = "contract_controls", .threads = 256, .dynamic = if (hb == 32) 101120 else 83328, .opt_in = true, .min_sm = 80 }, .usage = registry.none };
        }
    }
    break :blk out;
} ++ chunks ++ mma;

const mma = blk: {
    var out: [4]registry.CudaRecord = undefined;
    for ([_][]const u8{ "kn", "nk" }, 0..) |suffix, i| {
        const dynamic = if (i == 0) 56832 else 61440;
        for ([_]bool{ false, true }, 0..) |control, j| {
            const symbol = (if (control) "tf_ctl_glm_mma_" else "tf_glm_latent_mma_") ++ suffix;
            const image = if (control) "contract_controls" else "glm_glue";
            out[i * 2 + j] = .{ .launch = .{ .name = image ++ " " ++ symbol, .symbol = symbol, .image = image, .threads = 256, .dynamic = dynamic, .opt_in = true, .min_sm = 80 }, .usage = registry.none };
        }
    }
    break :blk out;
};

const chunks = blk: {
    var out: [2]registry.CudaRecord = undefined;
    for ([_]u32{ 16, 32 }, 0..) |hb, i| {
        const symbol = @import("std").fmt.comptimePrint("tf_glm_chunks_{d}", .{hb});
        out[i] = .{ .launch = .{ .name = "glm_glue " ++ symbol, .symbol = symbol, .image = "glm_glue", .threads = 256, .dynamic = if (hb == 32) 101120 else 83328, .opt_in = true, .min_sm = 80 }, .usage = registry.none };
    }
    break :blk out;
};
