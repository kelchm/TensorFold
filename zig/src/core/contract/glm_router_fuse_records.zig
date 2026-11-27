//! TF_GLM_ROUTER_FUSE's launch and its one-chain control (27 KiB static shared): pending until nvcc 13.3 measures them.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_router_fused", .symbol = "tf_glm_router_fused", .image = "glm_glue", .threads = 128, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
    .{ .launch = .{ .name = "contract_controls tf_ctl_glm_router_chain", .symbol = "tf_ctl_glm_router_chain", .image = "contract_controls", .threads = 128, .min_sm = 80 }, .usage = registry.none, .later_pending = true },
};
