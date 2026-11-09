//! TF_GLM_HC_FUSE's launch and its order control (32 KiB of static shared); compile usage pending nvcc receipts.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_post_partial", .symbol = "tf_glm_hc_post_partial", .image = "glm_glue", .threads = 512, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "contract_controls tf_ctl_glm_hc_post_partial_order", .symbol = "tf_ctl_glm_hc_post_partial_order", .image = "contract_controls", .threads = 512, .min_sm = 80 }, .usage = registry.none },
};
