//! GLM MLA fused sparse attention and FP8 tile-load control records: pending until measured with nvcc.
const std = @import("std");
const registry = @import("registry.zig");

fn smem(hb: u32) u32 {
    return if (hb == 32) 101120 else 83328;
}

pub const records = blk: {
    var out: [14]registry.CudaRecord = undefined;
    var n: usize = 0;
    for (.{ 16, 32 }) |hb| for (.{ "", "_fp8", "_fp8_paged" }) |kind| {
        const symbol = std.fmt.comptimePrint("tf_glm_chunks_{d}{s}_fused", .{ hb, kind });
        out[n] = .{
            .launch = .{ .name = "glm_glue " ++ symbol, .symbol = symbol, .image = "glm_glue", .threads = 256, .dynamic = smem(hb), .opt_in = true, .min_sm = 80 },
            .usage = registry.none,
            .later_pending = true,
        };
        n += 1;
    };
    for (.{ 16, 32 }) |hb| for (.{ "old", "bad" }) |tag| {
        const symbol = std.fmt.comptimePrint("tf_ctl_glm_chunks_{d}_fp8_{s}", .{ hb, tag });
        out[n] = .{
            .launch = .{ .name = "contract_controls " ++ symbol, .symbol = symbol, .image = "contract_controls", .threads = 256, .dynamic = smem(hb), .opt_in = true, .min_sm = 80 },
            .usage = registry.none,
            .later_pending = true,
        };
        n += 1;
    };
    for ([_][]const u8{ "kn", "nk" }, 0..) |suffix, i| {
        const dynamic: u32 = if (i == 0) 56832 else 61440;
        for ([_]bool{ false, true }) |control| {
            const symbol = (if (control) "tf_ctl_glm_mma_" else "tf_glm_latent_mma_") ++ suffix;
            const image = if (control) "contract_controls" else "glm_glue";
            out[n] = .{
                .launch = .{ .name = image ++ " " ++ symbol, .symbol = symbol, .image = image, .threads = 256, .dynamic = dynamic, .opt_in = true, .min_sm = 80 },
                .usage = registry.none,
                .later_pending = true,
            };
            n += 1;
        }
    }
    break :blk out;
};
