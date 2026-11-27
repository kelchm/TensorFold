//! Additional cuda_ranks checks: pair_experts, wire pieces, and bit-exact P2 switches.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;

test "TF_GLM_PAIR_EXPERTS: off by default, 0 or 1, and every rank must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_EXPERTS")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_EXPERTS")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_EXPERTS")) "2" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).pair_experts);
    const on = try Settings.read(Reader.on, &problem);
    try std.testing.expect(on.pair_experts);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).pair_experts);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PAIR_EXPERTS: 0 or 1", problem);
    const all = [_]Fingerprint{
        Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0)),
        Fingerprint.of(2, 1, .{}, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expectEqual(Mismatch{ .field = .pair_experts, .rank = 1 }, mismatch(&all).?);
}

test "wire pieces: a blank override is unset; the tools' form takes its world from TF_GLM_WORLD" {
    const Reader = struct {
        fn at(key: [:0]const u8, world: ?[]const u8, pieces: ?[]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_CHUNK_SOLO")) return "4096";
            if (std.mem.eql(u8, key, "TF_GLM_ROCE_MULTI")) return "1";
            if (std.mem.eql(u8, key, "TF_GLM_WORLD")) return world;
            if (std.mem.eql(u8, key, "TF_GLM_OVERLAP_PIECES")) return pieces;
            return null;
        }
        fn blank(key: [:0]const u8) ?[]const u8 {
            return at(key, null, " ");
        }
        fn three(key: [:0]const u8) ?[]const u8 {
            return at(key, "3", null);
        }
        fn none(key: [:0]const u8) ?[]const u8 {
            return at(key, null, null);
        }
    };
    var s: Settings = .{ .solo = 4096, .roce_multi = true, .pieces = 3 };
    ranks.wirePieces(&s, 3, Reader.blank);
    try std.testing.expectEqual(@as(u32, 8), s.pieces);
    s.pieces = 3;
    ranks.toolPieces(&s, Reader.three);
    try std.testing.expectEqual(@as(u32, 8), s.pieces);
    s.pieces = 3;
    ranks.toolPieces(&s, Reader.none);
    try std.testing.expectEqual(@as(u32, 3), s.pieces);
}

test "TF_GLM_DFLASH_ONLY: off by default, 0 or 1, and every rank must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_ONLY")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_ONLY")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_ONLY")) "yes" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).dflash_only);
    const on = try Settings.read(Reader.on, &problem);
    try std.testing.expect(on.dflash_only);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).dflash_only);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_ONLY: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{
        Fingerprint.of(2, 0, on, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    };
    try std.testing.expectEqual(Mismatch{ .field = .mtp, .rank = 1 }, mismatch(&all).?);
}

test "TF_GLM_DFLASH_Q4_DOWN: off by default, 0 or 1, and every rank must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_Q4_DOWN")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_Q4_DOWN")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_Q4_DOWN")) "4" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).dflash_q4_down);
    const on = try Settings.read(Reader.on, &problem);
    try std.testing.expect(on.dflash_q4_down);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).dflash_q4_down);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_Q4_DOWN: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    const all = [_]Fingerprint{
        Fingerprint.of(2, 0, on, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    };
    try std.testing.expectEqual(Mismatch{ .field = .mtp, .rank = 1 }, mismatch(&all).?);
}

test "startup switches: verify_tiles and TF_GLM_L2PF_SHARED_FIRST agree across ranks" {
    const Reader = struct {
        fn on(key: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_DECODE_MLA_WIDE")) return "1";
            if (std.mem.eql(u8, key, "TF_GLM_DECODE_TILE_RANGES")) return "1";
            if (std.mem.eql(u8, key, "TF_GLM_L2PF_SHARED_FIRST")) return "1";
            return null;
        }
        fn badMla(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DECODE_MLA_WIDE")) "2" else null;
        }
        fn badRanges(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DECODE_TILE_RANGES")) "yes" else null;
        }
    };
    var problem: []const u8 = "";
    const s = try Settings.read(Reader.on, &problem);
    try std.testing.expect(s.verify_tiles.mla and s.verify_tiles.ranges and s.l2pf.shared_first);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.badMla, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DECODE_MLA_WIDE: 0 or 1", problem);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.badRanges, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DECODE_TILE_RANGES: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    try std.testing.expectEqual(Mismatch{ .field = .tiled, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, .{ .verify_tiles = .{ .mla = true } }, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
    try std.testing.expectEqual(Mismatch{ .field = .tiled, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, .{ .verify_tiles = .{ .ranges = true } }, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
    try std.testing.expectEqual(Mismatch{ .field = .l2pf, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, .{ .l2pf = .{ .shared_first = true } }, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
}

test "startup switches: TF_GLM_EXL3_DECODE and TF_GLM_EXL3_N_MAJOR agree across ranks" {
    const Reader = struct {
        fn on(key: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_EXL3_DECODE")) return "1";
            if (std.mem.eql(u8, key, "TF_GLM_EXL3_N_MAJOR")) return "1";
            return null;
        }
        fn badDecode(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_EXL3_DECODE")) "2" else null;
        }
        fn badNmajor(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_EXL3_N_MAJOR")) "yes" else null;
        }
    };
    var problem: []const u8 = "";
    const s = try Settings.read(Reader.on, &problem);
    try std.testing.expect(s.exl3_decode and s.exl3_nmajor);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.badDecode, &problem));
    try std.testing.expectEqualStrings("TF_GLM_EXL3_DECODE: 0 or 1", problem);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.badNmajor, &problem));
    try std.testing.expectEqualStrings("TF_GLM_EXL3_N_MAJOR: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    try std.testing.expectEqual(Mismatch{ .field = .pair_experts, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, .{ .exl3_decode = true }, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
    try std.testing.expectEqual(Mismatch{ .field = .pair_experts, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, .{ .exl3_decode = true, .exl3_nmajor = true }, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{ .exl3_decode = true }, 8192, true, h, h, h),
    }).?);
}

test "startup switches: TF_GLM_PACKED_PROMPT_S2 agrees across ranks" {
    const Reader = struct {
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PACKED_PROMPT_S2")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PACKED_PROMPT_S2")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PACKED_PROMPT_S2")) "2" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect((try Settings.read(Reader.on, &problem)).prompt_s2);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).prompt_s2);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PACKED_PROMPT_S2: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    try std.testing.expectEqual(Mismatch{ .field = .bits, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, .{ .prompt_s2 = true }, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
}

test "startup switches: TF_GLM_DFLASH_SNAP_CONTEXT is off by default, 0 or 1, and agrees across ranks" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_SNAP_CONTEXT")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_SNAP_CONTEXT")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_SNAP_CONTEXT")) "2" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).dflash_snap_context);
    const on = try Settings.read(Reader.on, &problem);
    try std.testing.expect(on.dflash_snap_context);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).dflash_snap_context);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_DFLASH_SNAP_CONTEXT: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    try std.testing.expectEqual(Mismatch{ .field = .mtp, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, on, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
}

test "startup switches: TF_GLM_PAIR_DFLASH is off by default, 0 or 1, and agrees across ranks" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn on(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_DFLASH")) "1" else null;
        }
        fn off(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_DFLASH")) "0" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PAIR_DFLASH")) "2" else null;
        }
    };
    var problem: []const u8 = "";
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).pair_dflash);
    const on = try Settings.read(Reader.on, &problem);
    try std.testing.expect(on.pair_dflash);
    try std.testing.expect(!(try Settings.read(Reader.off, &problem)).pair_dflash);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    try std.testing.expectEqualStrings("TF_GLM_PAIR_DFLASH: 0 or 1", problem);
    const h: [32]u8 = @splat(0);
    try std.testing.expectEqual(Mismatch{ .field = .pair, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, on, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    }).?);
}

test "startup switches: external drafter precision and checkpoint digest agree across ranks" {
    const h: [32]u8 = @splat(0);
    const d0 = ranks.hashBytes("drafter-a");
    const d1 = ranks.hashBytes("drafter-b");
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{}, 8192, true, h, h, h),
        Fingerprint.of(2, 1, .{}, 8192, true, h, h, h),
    };
    ranks.drafterFingerprint(&all[0], 4, d0);
    ranks.drafterFingerprint(&all[1], 4, d0);
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
    ranks.drafterFingerprint(&all[1], 0, d0);
    try std.testing.expectEqual(Mismatch{ .field = .drafter_bits, .rank = 1 }, mismatch(&all).?);
    ranks.drafterFingerprint(&all[1], 4, d1);
    try std.testing.expectEqual(Mismatch{ .field = .drafter_checkpoint, .rank = 1 }, mismatch(&all).?);
}

test "startup switches: rank-local P2 validation failures notify waiting peers through cuda_start.open" {
    const startup = @import("cuda_start.zig");
    const env = @import("cuda_env.zig");
    const Place = struct { rank: usize, world: usize = 2 };
    const World = struct {
        connected: [2]bool = @splat(false),
        refused: [2]?[]const u8 = @splat(null),
    };
    const Watch = struct {
        world: *World,
        rank: usize,
        pub fn start(w: @This(), _: std.Io, rank: usize, _: usize) !void {
            w.world.connected[rank] = true;
        }
        pub fn fail(w: @This(), name: []const u8) void {
            const peer = 1 - w.rank;
            if (w.world.connected[w.rank] and w.world.connected[peer]) w.world.refused[peer] = name;
        }
    };
    const Args = struct { share: usize = 0, drafts: bool = true, external: bool = true, bits: u8 = 4 };
    const Hook = struct {
        fn after(a: Args, set: Settings) anyerror!void {
            try env.validateRankOptions(set, a.share, a.drafts, a.external, a.bits);
        }
        fn report(_: Args, _: startup.Report) void {}
    };
    const cases = [_]struct { set: Settings, args: Args, err: anyerror, name: []const u8 }{
        .{ .set = .{ .dflash_q4_down = true }, .args = .{ .bits = 0 }, .err = error.BadSetting, .name = "BadSetting" },
        .{ .set = .{ .dflash_only = true }, .args = .{ .external = false }, .err = error.BadSetting, .name = "BadSetting" },
        .{ .set = .{ .dflash_snap_context = true }, .args = .{ .share = 1 }, .err = error.GlmDraftSnapshotWithSharedKv, .name = "GlmDraftSnapshotWithSharedKv" },
    };
    for (cases) |c| {
        var w: World = .{};
        try startup.open(std.testing.io, Place{ .rank = 0 }, "auto", Settings{}, Watch{ .world = &w, .rank = 0 }, Args{}, Hook.after, Hook.report);
        try std.testing.expectError(c.err, startup.open(std.testing.io, Place{ .rank = 1 }, "auto", c.set, Watch{ .world = &w, .rank = 1 }, c.args, Hook.after, Hook.report));
        try std.testing.expectEqualStrings(c.name, w.refused[0] orelse "missing notification");
    }
}

test "startup switches: TF_GLM_SHARED_KDA_WIDE inherits TF_GLM_KDA_DECODE_WIDE, overrides independently, and agrees across ranks" {
    const Reader = struct {
        var kda: ?[]const u8 = null;
        var shared: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_KDA_DECODE_WIDE")) return kda;
            if (std.mem.eql(u8, key, "TF_GLM_SHARED_KDA_WIDE")) return shared;
            return null;
        }
    };
    var problem: []const u8 = "";
    // Unset inherits TF_GLM_KDA_DECODE_WIDE (default true, or false when decode wide is 0).
    Reader.kda = null;
    Reader.shared = null;
    const def = try Settings.read(Reader.get, &problem);
    try std.testing.expect(def.kda_wide and def.shared_kda_wide);
    Reader.kda = "0";
    Reader.shared = null;
    const both_off = try Settings.read(Reader.get, &problem);
    try std.testing.expect(!both_off.kda_wide and !both_off.shared_kda_wide);
    // Independent override in either direction.
    Reader.kda = "1";
    Reader.shared = "0";
    const shared_off = try Settings.read(Reader.get, &problem);
    try std.testing.expect(shared_off.kda_wide and !shared_off.shared_kda_wide);
    Reader.kda = "0";
    Reader.shared = "1";
    const shared_on = try Settings.read(Reader.get, &problem);
    try std.testing.expect(!shared_on.kda_wide and shared_on.shared_kda_wide);
    // Malformed values are refused by name.
    for ([_][]const u8{ "2", "yes", " 0" }) |bad_val| {
        Reader.kda = "1";
        Reader.shared = bad_val;
        try std.testing.expectError(error.BadSetting, Settings.read(Reader.get, &problem));
        try std.testing.expectEqualStrings("TF_GLM_SHARED_KDA_WIDE: 0 or 1", problem);
    }
    // Ranks with differing TF_GLM_SHARED_KDA_WIDE disagree on .kda_wide.
    const h: [32]u8 = @splat(0);
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&.{
        Fingerprint.of(2, 0, shared_off, 8192, true, h, h, h),
        Fingerprint.of(2, 1, shared_off, 8192, true, h, h, h),
    }));
    try std.testing.expectEqual(Mismatch{ .field = .kda_wide, .rank = 1 }, mismatch(&.{
        Fingerprint.of(2, 0, def, 8192, true, h, h, h),
        Fingerprint.of(2, 1, shared_off, 8192, true, h, h, h),
    }).?);
}

test "TF_GLM_PROMPT_PAIR=solo pairs TF_GLM_CHUNK_SOLO=4096 passes only, rings of both, and every rank must agree" {
    const Reader = struct {
        var solo: ?[]const u8 = null;
        var pair: ?[]const u8 = "solo";
        fn get(key: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, key, "TF_GLM_CHUNK_SOLO")) return solo;
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_PAIR")) pair else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "2048", "8192" }) |width| {
        Reader.solo = width;
        try std.testing.expectError(error.BadSetting, Settings.read(Reader.get, &problem));
        try std.testing.expectEqualStrings("TF_GLM_PROMPT_PAIR=solo requires TF_GLM_CHUNK_SOLO=4096", problem);
    }
    Reader.solo = "4096";
    const on = try Settings.read(Reader.get, &problem);
    try std.testing.expect(on.pair and on.pair_solo);
    try std.testing.expectEqual(@as(usize, 8192), on.ringWidth());
    Reader.pair = "1";
    const unpaired = try Settings.read(Reader.get, &problem);
    try std.testing.expect(unpaired.pair and !unpaired.pair_solo);
    try std.testing.expectEqual(@as(usize, 4096), unpaired.ringWidth());
    try std.testing.expectEqual(@as(usize, 2048), (Settings{}).ringWidth());
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, on, 8192, true, @splat(0), @splat(0), @splat(0)), Fingerprint.of(2, 1, on, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
    try std.testing.expectEqual(@as(u32, 5), all[0].pair); // bit 0 pair | bit 2 pair_solo (bit 1 reserved for pair_dflash)
    var with_dflash = on;
    with_dflash.pair_dflash = true;
    try std.testing.expectEqual(@as(u32, 7), Fingerprint.of(2, 0, with_dflash, 8192, true, @splat(0), @splat(0), @splat(0)).pair);
    all[1] = Fingerprint.of(2, 1, unpaired, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(Mismatch{ .field = .pair, .rank = 1 }, mismatch(&all).?);
}
