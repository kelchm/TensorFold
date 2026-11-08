//! Compile all device contract hooks even when the CUDA runner uses inert kernel images.
const std = @import("std");
const fp8 = @import("contract_glm_fp8_attn.zig");
const mma = @import("contract_glm_mma.zig");
test "GLM MLA: device contract hooks compile without CUDA images" {
    inline for (.{ fp8.Case, mma.Case }) |T| {
        inline for (.{ "prepare", "run", "reference", "count", "kernel" }) |name| std.mem.doNotOptimizeAway(&@field(T, name));
    }
    std.mem.doNotOptimizeAway(&fp8.Group.create);
    std.mem.doNotOptimizeAway(&mma.Group.create);
}
test {
    _ = @import("glm_mla_ref.zig");
}
