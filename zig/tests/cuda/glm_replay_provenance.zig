//! Actual checkpoint identities and per-Session receipt provenance, separate from stable runtime binding.
const std = @import("std");
const fixture = @import("glm_replay_fixture.zig");

pub const Checkpoints = struct {
    scheme: []const u8,
    target: [2][]const u8,
    drafter: [2][]const u8,

    pub fn check(c: Checkpoints) !void {
        if (!std.mem.eql(u8, c.scheme, "tensorfold-full-identity-v1")) return error.CheckpointScheme;
        for (c.target ++ c.drafter) |digest| try fixture.hex(digest, 64);
    }

    pub fn require(expected: Checkpoints, actual: Checkpoints) !void {
        try expected.check();
        try actual.check();
        for (expected.target ++ expected.drafter, actual.target ++ actual.drafter) |want, got|
            if (!std.mem.eql(u8, want, got)) return error.CheckpointMismatch;
    }
};

pub const Provenance = struct {
    session_nonce: []const u8,
    campaign_sha256: []const u8,
    checkpoints: Checkpoints,

    pub fn fromSession(a: std.mem.Allocator, s: anytype) !Provenance {
        var p: Provenance = .{
            .session_nonce = try hex(a, s.session_nonce),
            .campaign_sha256 = try hex(a, s.options.config_hash),
            .checkpoints = undefined,
        };
        p.checkpoints.scheme = "tensorfold-full-identity-v1";
        for (0..2) |rank| {
            p.checkpoints.target[rank] = try hex(a, s.checkpoints.target[rank]);
            p.checkpoints.drafter[rank] = try hex(a, s.checkpoints.drafter[rank]);
        }
        return p;
    }
};

fn hex(a: std.mem.Allocator, digest: [32]u8) ![]const u8 {
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
}

test "replay checkpoint pins reject changed target or drafter on either rank and invalid schemes" {
    const zero = "0000000000000000000000000000000000000000000000000000000000000000";
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const want: Checkpoints = .{ .scheme = "tensorfold-full-identity-v1", .target = .{ zero, zero }, .drafter = .{ one, one } };
    try want.require(want);
    inline for (.{ "target", "drafter" }) |field| for (0..2) |rank| {
        var changed = want;
        @field(changed, field)[rank] = "2222222222222222222222222222222222222222222222222222222222222222";
        try std.testing.expectError(error.CheckpointMismatch, want.require(changed));
    };
    var bad = want;
    bad.scheme = "file-manifest-v1";
    try std.testing.expectError(error.CheckpointScheme, bad.check());
    bad = want;
    bad.target[1] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    try std.testing.expectError(error.InvalidDigest, bad.check());
}
