//! GLM-only startup settings are checked before rank placement, NCCL or weights.
const environment = @import("sys").environment;
pub fn startup() !void {
    try environment.checkScope(.glm, environment.Process{});
}
