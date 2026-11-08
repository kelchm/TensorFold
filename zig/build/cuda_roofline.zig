//! Benchmark-only images belong to tf-cuda-test, outside the engine's kernel catalog and contract records.
const std = @import("std");
const build = @import("cuda_fatbin.zig");

pub fn images(b: *std.Build, runner: *std.Build.Module, install: *std.Build.Step, prebuilt: ?[]const u8, nvcc: ?[]const u8, version: ?std.Build.LazyPath, sms: []const u8) void {
    const options = b.addOptions();
    options.addOption(bool, "roofline_images", prebuilt != null or nvcc != null);
    runner.addOptions("roofline_options", options);
    for ([_][]const u8{ "roofline", "roofline_low" }) |name| {
        const low = std.mem.eql(u8, name, "roofline_low");
        const image = if (prebuilt) |dir|
            b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{name}) }))
        else if (nvcc) |tool|
            build.fatbin(b, tool, version.?, .{ .name = name, .flags = &.{"-O3"}, .min_sm = if (low) 120 else 80, .arch_specific = low }, sms)
        else
            continue;
        runner.addAnonymousImport(b.fmt("{s}_image", .{name}), .{ .root_source_file = image });
        install.dependOn(&b.addInstallFile(image, b.fmt("fatbin/{s}.fatbin", .{name})).step);
    }
}
