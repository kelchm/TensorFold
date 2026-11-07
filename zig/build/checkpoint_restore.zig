//! Dedicated host modules: all new checkpoint tests use apple_m1 code generation on arm64 macOS.
const std = @import("std");
pub fn tests(b: *std.Build) void {
    const host = b.graph.host;
    const target = if (host.result.cpu.arch == .aarch64 and host.result.os.tag == .macos)
        b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos, .cpu_model = .{ .explicit = &std.Target.aarch64.cpu.apple_m1 } })
    else
        host;
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = .debug, .link_libc = true });
    core.addImport("gpu_memory", @import("gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .debug, .link_libc = true }));
    core.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = .debug, .link_libc = true }));
    const cuda = @import("cuda_runtime.zig").runtime(b, target, .debug, &.{});
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .debug, .link_libc = true });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = .debug, .link_libc = true });
    const glm = @import("cuda_engines.zig").glmCuda(b, target, .debug, cuda, core, lanes, heat);
    const step = b.step("test-checkpoint-restore-host", "Native inactive GLM restore and all-rank filesystem recovery");
    const actual = b.createModule(.{ .root_source_file = b.path("zig/glm_checkpoint_restore_host.zig"), .target = target, .optimize = .debug, .link_libc = true, .imports = &.{ .{ .name = "glm", .module = glm }, .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = actual })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/checkpoint_epoch_test.zig"), .target = target, .optimize = .debug, .link_libc = true }), .filters = &.{"checkpoint restore:"} })).step);
}
