//! GLM's decode hyper-connection kernels over more threads; compile resources remain pending until nvcc receipts.
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_finish_split", .symbol = "tf_glm_hc_finish_split", .image = "glm_glue", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_finish_f32w_split", .symbol = "tf_glm_hc_finish_f32w_split", .image = "glm_glue", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_partial8_o2", .symbol = "tf_glm_hc_partial8_o2", .image = "glm_glue", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
