//! GLM prompt slot output: compile receipts must be re-derived for the port's symbols.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_slots", .symbol = "tf_exl3_prompt_down_slots", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu_glm", .symbol = "tf_exl3_prompt_gu_glm", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm", .symbol = "tf_exl3_prompt_down_glm", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm_b16", .symbol = "tf_exl3_prompt_down_glm_b16", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    // TF_EXL3_PROMPT_GU=pc: 640 threads, opt-in shared (exl3_prompt.zig pc); nvcc 13.3: 96 registers, no spills
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu_glm_pc", .symbol = "tf_exl3_prompt_gu_glm_pc", .image = "exl3_experts", .threads = 640, .dynamic = 96832, .opt_in = true }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm_h", .symbol = "tf_exl3_prompt_down_glm_h", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm_h_b16", .symbol = "tf_exl3_prompt_down_glm_h_b16", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
};

test "GLM prompt port: source kit counts never stand in for missing compile receipts" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(registry.none, r.usage);
        try t.expectEqual(@as(usize, 5), try registry.checkCuda(t.allocator, &.{r}, true));
        for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version|
            try t.expectEqual(registry.none, registry.usageFor(r, version));
    }
}
