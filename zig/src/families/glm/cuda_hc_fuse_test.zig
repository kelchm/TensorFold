//! TF_GLM_HC_FUSE: its setting, and which glue calls write the next hc_pre's partials in hc_post.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const mismatch = ranks.mismatch;

test "TF_GLM_HC_FUSE: off by default, 0 or 1, refused otherwise, identical, and outside the ranks' fingerprint" {
    const Reader = struct {
        var raw: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_HC_FUSE")) raw else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "", "0" }) |raw| {
        Reader.raw = raw;
        try std.testing.expect(!(try Settings.read(Reader.get, &problem)).hc_fuse);
    }
    Reader.raw = "1";
    const on = try Settings.read(Reader.get, &problem);
    try std.testing.expect(on.hc_fuse and on.numerics.bits() == 0);
    for ([_][]const u8{ "on", "2", " 1 " }) |raw| {
        Reader.raw = raw;
        try std.testing.expectError(error.BadSetting, Settings.read(Reader.get, &problem));
        try std.testing.expectEqualStrings("TF_GLM_HC_FUSE: 0 or 1", problem);
    }
    // byte-identical to the unfused launches, so a rank may fuse while another does not
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, on, 8192, true, h, h, h), Fingerprint.of(2, 1, .{}, 8192, true, h, h, h) };
    try std.testing.expect(mismatch(&all) == null);
    try std.testing.expectEqual(@import("profile").Class.identical, @import("profile").entry(.TF_GLM_HC_FUSE).class);
}

test "TF_GLM_HC_FUSE fuses a prompt buffer's calls of 64 rows or more, never a decode window's" {
    var b: fw.Buffers = undefined;
    b.prefill = true;
    const one = [1]fw.Rank{.{ .w = undefined, .st = undefined, .b = &b }};
    var f: fw.Forward = .{ .d = undefined, .m = undefined, .stream = undefined, .ranks = &one, .exchange = .local, .hc_fuse = true };
    try std.testing.expect(f.fuses(64) and f.fuses(4096) and !f.fuses(63));
    b.prefill = false;
    try std.testing.expect(!f.fuses(64));
    b.prefill = true;
    f.hc_fuse = false;
    try std.testing.expect(!f.fuses(4096));
}
