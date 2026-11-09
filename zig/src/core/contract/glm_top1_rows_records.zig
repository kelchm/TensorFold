//! A window's greedy picks in one launch (tf_glm_top1_rows) and its planted control: compile receipts are pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_top1_rows", .symbol = "tf_glm_top1_rows", .image = "glm_glue", .threads = 1024 }, .usage = registry.none },
    .{ .launch = .{ .name = "contract_controls tf_ctl_glm_top1_rows_words", .symbol = "tf_ctl_glm_top1_rows_words", .image = "contract_controls", .threads = 1024 }, .usage = registry.none },
};

test "GLM top-1 rows: tf_glm_top1's compile receipts never stand in for the rows launch's missing ones" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(registry.none, r.usage);
        try t.expectEqual(@as(usize, 5), try registry.checkCuda(t.allocator, &.{r}, true));
        for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version|
            try t.expectEqual(registry.none, registry.usageFor(r, version));
    }
}
