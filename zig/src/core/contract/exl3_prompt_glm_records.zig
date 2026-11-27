//! GLM-only EXL3 prompt kernels: nvcc 13.0 pending; 13.3 receipts recorded in measurements/pending_133.zig.
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu_glm", .symbol = "tf_exl3_prompt_gu_glm", .image = "exl3_experts", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm", .symbol = "tf_exl3_prompt_down_glm", .image = "exl3_experts", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm_b16", .symbol = "tf_exl3_prompt_down_glm_b16", .image = "exl3_experts", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    // TF_EXL3_PROMPT_GU=pc: 640 threads, 96,832 bytes opt-in dynamic shared (exl3_prompt.zig pc)
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu_glm_pc", .symbol = "tf_exl3_prompt_gu_glm_pc", .image = "exl3_experts", .threads = 640, .dynamic = 96832, .opt_in = true, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm_h", .symbol = "tf_exl3_prompt_down_glm_h", .image = "exl3_experts", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm_h_b16", .symbol = "tf_exl3_prompt_down_glm_h_b16", .image = "exl3_experts", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};

test "GLM EXL3 prompt records: unmeasured compiler usage stays pending until measured" {
    const std = @import("std");
    for (records) |r| {
        try std.testing.expectEqual(registry.none, r.usage);
        try std.testing.expect(r.later_pending);
    }
}
