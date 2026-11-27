//! GLM KDA step-width and segmented verify kernels: nvcc 13.0 pending, 13.3 in measurements/pending_133.zig.
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_kda tf_glm_kda_step4", .symbol = "tf_glm_kda_step4", .image = "glm_kda", .threads = 128, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "glm_kda tf_glm_kda_step2", .symbol = "tf_glm_kda_step2", .image = "glm_kda", .threads = 64, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
