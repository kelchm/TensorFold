//! Compile all device contract hooks even when the CUDA runner uses inert kernel images.
const std = @import("std");
const fp8 = @import("contract_glm_fp8_attn.zig");
const mma = @import("contract_glm_mma.zig");
const fused = @import("contract_glm_fused_attn.zig");
test "GLM MLA: device contract hooks compile without CUDA images" {
    inline for (.{ fp8.Case, mma.Case, fused.Case }) |T| {
        inline for (.{ "prepare", "run", "reference", "count", "kernel" }) |name| std.mem.doNotOptimizeAway(&@field(T, name));
    }
    inline for (.{ "replaced", "check", "bounds", "simulate" }) |name| std.mem.doNotOptimizeAway(&@field(fused.Case, name));
    std.mem.doNotOptimizeAway(&fp8.Group.create);
    std.mem.doNotOptimizeAway(&mma.Group.create);
    std.mem.doNotOptimizeAway(&fused.Group.create);
}
test {
    _ = @import("glm_mla_ref.zig");
}
