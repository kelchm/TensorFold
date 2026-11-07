//! Driver-free checks of native replay fixtures and controls.
const std = @import("std");

pub fn host(b: *std.Build, all: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_replay_host_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    const step = b.step("test-glm-replay-host", "GLM fixed replay hashes, malformed fixtures, pool and ingest controls");
    step.dependOn(&run.step);
    all.dependOn(step);
    const cli = b.addExecutable(.{ .name = "tf-glm-replay-fixture", .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_replay_fixture.zig"), .target = b.graph.host, .optimize = .debug }) });
    b.step("glm-replay-host", "Build the driver-free replay fixture preflight").dependOn(&b.addInstallArtifact(cli, .{}).step);
}

pub fn compileCommand(b: *std.Build, target: std.Build.ResolvedTarget, cuda: *std.Build.Module, glm: *std.Build.Module) void {
    const obj = b.addObject(.{ .name = "glm-replay-command", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_replay_compile.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = glm } },
    }) });
    _ = obj.getEmittedBin();
    const correctness_obj = b.addObject(.{ .name = "glm-replay-correctness", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_replay_correctness_compile.zig"),
        .target = target,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = glm } },
    }) });
    _ = correctness_obj.getEmittedBin();
    const step = b.step("compile-glm-replay", "Compile the complete production replay command without executing CUDA");
    step.dependOn(&obj.step);
    step.dependOn(&correctness_obj.step);
}

pub fn correctness(b: *std.Build, cuda: *std.Build.Module, glm: *std.Build.Module, all: *std.Build.Step) void {
    const root = b.createModule(.{
        .root_source_file = b.path("zig/tests/cuda/glm_replay_correctness_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = glm } },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
    const native = b.addRunArtifact(b.addTest(.{ .root_module = glm, .filters = &.{"replay "} }));
    const step = b.step("test-glm-replay-correctness-host", "Adversarial lookahead, full-block evidence and light capture controls without devices");
    step.dependOn(&run.step);
    step.dependOn(&native.step);
    all.dependOn(step);
    b.top_level_steps.get("test-glm-replay-host").?.step.dependOn(step);
}
