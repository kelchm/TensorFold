//! Analyze every packed prompt replacement hook without CUDA images or device execution.
const std = @import("std");
const cuda = @import("cuda");
const g = @import("core").contract.gates;
const gate = @import("gate");

export fn compileGlmMmContract(gpu: *const gate.Gpu) void {
    compile(gpu.*) catch {};
}

fn compile(gpu: gate.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prompts = try gate.Prompts.init(gpu, try cuda.Module.load(gpu.d, &.{})); // analyzed only: no images
    defer prompts.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try prompts.append(gpu, stream, &kernels);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 1, &tally);
}
