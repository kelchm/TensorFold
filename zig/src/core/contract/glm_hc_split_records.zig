//! GLM decode hyper-connection split kernels: nvcc 13.0 pending, 13.3 in measurements/pending_133.zig.
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_finish_split", .symbol = "tf_glm_hc_finish_split", .image = "glm_glue", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_finish_f32w_split", .symbol = "tf_glm_hc_finish_f32w_split", .image = "glm_glue", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_partial8_o2", .symbol = "tf_glm_hc_partial8_o2", .image = "glm_glue", .threads = 256, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
