//! GLM prompt slot output: compile receipts must be re-derived for the port's symbols.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_slots", .symbol = "tf_exl3_prompt_down_slots", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_gu_glm", .symbol = "tf_exl3_prompt_gu_glm", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_prompt_down_glm", .symbol = "tf_exl3_prompt_down_glm", .image = "exl3_experts", .threads = 256 }, .usage = registry.none },
};
