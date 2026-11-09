//! TF_EXL3_PROMPT_TILE: its setting, the ranks' agreement and the profile's class.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;

test "TF_EXL3_PROMPT_TILE: 64 by default, 64 or 128, refused otherwise, identical, and outside the ranks' fingerprint" {
    const Reader = struct {
        var raw: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_EXL3_PROMPT_TILE")) raw else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "", "64" }) |raw| {
        Reader.raw = raw;
        try std.testing.expectEqual(@as(u32, 64), (try Settings.read(Reader.get, &problem)).exl3_tile);
    }
    Reader.raw = "128";
    const wide = try Settings.read(Reader.get, &problem);
    try std.testing.expect(wide.exl3_tile == 128 and wide.numerics.bits() == 0);
    for ([_][]const u8{ "96", "256", "0", "1", "wide" }) |raw| {
        Reader.raw = raw;
        try std.testing.expectError(error.BadSetting, Settings.read(Reader.get, &problem));
        try std.testing.expectEqualStrings("TF_EXL3_PROMPT_TILE: 64 or 128", problem);
    }
    // byte-identical to 64-member items, so one rank may run 128 while another runs 64
    const h: [32]u8 = @splat(0);
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, wide, 8192, true, h, h, h), Fingerprint.of(2, 1, .{}, 8192, true, h, h, h) };
    try std.testing.expect(ranks.mismatch(&all) == null);
    const e = @import("profile").entry(.TF_EXL3_PROMPT_TILE);
    try std.testing.expectEqual(@import("profile").Class.identical, e.class);
    try std.testing.expect(e.applies("glm5_next", "cuda") and !e.applies("qwen3_5", "cuda"));
    try std.testing.expect(e.differs("128") and !e.differs("64") and !e.differs(""));
}
