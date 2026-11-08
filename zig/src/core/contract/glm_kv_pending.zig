//! GLM E4M3 latent cache resources from nvcc 13.0; numerical device gates remain separate.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_lat_write_fp8", .symbol = "tf_glm_lat_write_fp8", .image = "glm_glue", .threads = 512, .min_sm = 80 }, .usage = .{ .{ .regs = 26, .static = 64, .bound = 512 }, .{ .regs = 28, .static = 64, .bound = 512 }, .{ .regs = 28, .static = 64, .bound = 512 }, .{ .regs = 25, .static = 64, .bound = 512 }, .{ .regs = 25, .static = 64, .bound = 512 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_16_fp8", .symbol = "tf_glm_chunks_16_fp8", .image = "glm_glue", .threads = 256, .dynamic = 83328, .opt_in = true, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_32_fp8", .symbol = "tf_glm_chunks_32_fp8", .image = "glm_glue", .threads = 256, .dynamic = 101120, .opt_in = true, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
test "GLM FP8 KV: the writer is measured and changed chunk readers await receipts" {
    const t = @import("std").testing;
    for (records) |r| {
        if (r.launch.threads == 512) for (r.usage) |u| {
            try t.expect(u.regs > 0);
        };
        try t.expectEqual(@as(usize, if (r.launch.threads == 512) 0 else 5), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
