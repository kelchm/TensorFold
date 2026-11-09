//! A shared window's KDA segments in one launch a kernel; compile resources remain pending until measured with nvcc.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_kda tf_glm_kda_prep_segs", .symbol = "tf_glm_kda_prep_segs", .image = "glm_kda", .threads = 512 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_kda tf_glm_kda_step_segs", .symbol = "tf_glm_kda_step_segs", .image = "glm_kda", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "glm_kda tf_glm_kda_out_segs", .symbol = "tf_glm_kda_out_segs", .image = "glm_kda", .threads = 128 }, .usage = registry.none },
};

test "GLM KDA segs: the per-segment kernels' compiles never stand in for the batched kernels' receipts" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(registry.none, r.usage);
        try t.expectEqual(@as(usize, 5), try registry.checkCuda(t.allocator, &.{r}, true));
        for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version|
            try t.expectEqual(registry.none, registry.usageFor(r, version));
    }
}
