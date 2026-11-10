//! TF_GLM_ROUTER_FUSE's launch and its one-chain control (27 KiB of static shared); usage from nvcc 13.3's compiles.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_router_fused", .symbol = "tf_glm_router_fused", .image = "glm_glue", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 86, .static = 27648, .bound = 128 }, .{ .regs = 86, .static = 27648, .bound = 128 }, .{ .regs = 86, .static = 27648, .bound = 128 }, .{ .regs = 89, .static = 27648, .bound = 128 }, .{ .regs = 89, .static = 27648, .bound = 128 } } },
    .{ .launch = .{ .name = "contract_controls tf_ctl_glm_router_chain", .symbol = "tf_ctl_glm_router_chain", .image = "contract_controls", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 72, .static = 27648, .bound = 128 }, .{ .regs = 74, .static = 27648, .bound = 128 }, .{ .regs = 74, .static = 27648, .bound = 128 }, .{ .regs = 82, .static = 27648, .bound = 128 }, .{ .regs = 82, .static = 27648, .bound = 128 } } },
};
