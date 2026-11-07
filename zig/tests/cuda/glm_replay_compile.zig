//! Compile the complete command including validation without a driver or kernel image.
const std = @import("std");

export fn compileReplayCommand(init: *std.process.Init, args: [*]const [:0]const u8, count: usize) void {
    _ = @import("glm_replay.zig").command(init.*, "glm-replay", args[0..count]) catch {};
}
