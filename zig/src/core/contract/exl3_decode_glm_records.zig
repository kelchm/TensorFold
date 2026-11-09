//! GLM's EXL3 decode-window grouped kernels (contract_exl3_decode.zig): compile receipts pending until measured.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_grouped_1_nt4", .symbol = "tf_exl3_grouped_1_nt4", .image = "exl3_experts", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_grouped_1_k8r8", .symbol = "tf_exl3_grouped_1_k8r8", .image = "exl3_experts", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_grouped_1_k8r8s", .symbol = "tf_exl3_grouped_1_k8r8s", .image = "exl3_experts", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_grouped_1_k8r16s", .symbol = "tf_exl3_grouped_1_k8r16s", .image = "exl3_experts", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_grouped_1_k8r8p", .symbol = "tf_exl3_grouped_1_k8r8p", .image = "exl3_experts", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_grouped_1_k8r16p", .symbol = "tf_exl3_grouped_1_k8r16p", .image = "exl3_experts", .threads = 128 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_experts_rot_in_p", .symbol = "tf_exl3_experts_rot_in_p", .image = "exl3_experts", .threads = 32 }, .usage = registry.none },
    .{ .launch = .{ .name = "exl3_experts tf_exl3_gateup_epilogue_p", .symbol = "tf_exl3_gateup_epilogue_p", .image = "exl3_experts", .threads = 32 }, .usage = registry.none },
};

test "GLM decode port: source kit counts never stand in for missing compile receipts" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(registry.none, r.usage);
        try t.expectEqual(@as(usize, 5), try registry.checkCuda(t.allocator, &.{r}, true));
        for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version|
            try t.expectEqual(registry.none, registry.usageFor(r, version));
    }
}
