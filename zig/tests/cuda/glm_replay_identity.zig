//! CPU-only local checkpoint identity producer; directory names do not qualify upstream revisions.
const std = @import("std");
const identity = @import("checkpoint_identity");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.ReplayIdentityArguments;
    const target = try identity.fullIdentity(init.gpa, init.io, args[1]);
    const drafter = try identity.fullIdentity(init.gpa, init.io, args[2]);
    const raw = try std.json.Stringify.valueAlloc(a, .{
        .schema = "glm-replay-native-local-identity-v1",
        .scheme = "tensorfold-full-identity-v1",
        .target = std.fmt.bytesToHex(target, .lower),
        .drafter = std.fmt.bytesToHex(drafter, .lower),
        .revision_qualification = "unverified",
    }, .{});
    try std.Io.File.stdout().writeStreamingAll(init.io, raw);
    try std.Io.File.stdout().writeStreamingAll(init.io, "\n");
}
