//! Isolated build graph. Existing build files and the server graph remain untouched.
const std = @import("std");
pub fn build(b: *std.Build) void {
    const host = b.graph.host.result;
    const defaults: std.Target.Query = if (host.cpu.arch == .aarch64 and host.os.tag == .macos)
        .{ .cpu_arch = .aarch64, .os_tag = .macos, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 } }
    else
        .{};
    const target = b.standardTargetOptions(.{ .default_target = defaults });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .debug, .link_libc = true });
    core.addImport("gpu_memory", @import("zig/build/gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .debug, .link_libc = true }));
    core.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = .debug, .link_libc = true }));
    const cuda = @import("zig/build/cuda_runtime.zig").runtime(b, target, .debug, &.{});
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .debug });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = .debug });
    const glm = @import("zig/build/cuda_engines.zig").glmCuda(b, target, .debug, cuda, core, lanes, heat);
    const actual = b.createModule(.{ .root_source_file = b.path("zig/checkpoint_store_host.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const tests = b.addTest(.{ .root_module = actual });
    b.step("test", "Rank-local block store, direct inactive restore, separate-directory epochs").dependOn(&b.addRunArtifact(tests).step);
    b.step("compile", "Compile the complete store test body for the selected target").dependOn(&tests.step);
    @import("zig/build/sys.zig").wire(b);
}
