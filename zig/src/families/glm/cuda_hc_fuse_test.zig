//! TF_GLM_HC_FUSE: its setting, and which glue calls write the next hc_pre's partials in hc_post.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const glue = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
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
    try std.testing.expect(on.hc_fuse);
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

test "TF_GLM_HC_FUSE fuses 4-stream prompt calls of 64 rows or more, never decode windows or full GLM-5.3" {
    var w: weights.Weights = undefined;
    w.config = .{};
    var b: fw.Buffers = undefined;
    b.prefill = true;
    const one = [1]fw.Rank{.{ .w = &w, .st = undefined, .b = &b }};
    var f: fw.Forward = .{ .d = undefined, .m = undefined, .stream = undefined, .ranks = &one, .exchange = .local, .hc_fuse = true };
    try std.testing.expect(f.fuses(64) and f.fuses(4096) and !f.fuses(63));
    b.prefill = false;
    try std.testing.expect(!f.fuses(64));
    b.prefill = true;
    w.config.dsa = true;
    try std.testing.expect(!f.fuses(4096));
    w.config.dsa = false;
    f.hc_fuse = false;
    try std.testing.expect(!f.fuses(4096));
}

test "TF_GLM_HC_FUSE: glueRows launches hc_post_partial or hc_post_partial_own then hc_finish" {
    const Rec = struct {
        var ids: [8]usize = undefined;
        var n: usize = 0;
        fn run(f: cuda.abi.Function, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            ids[n] = @intFromPtr(f);
            n += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.run;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Rec.run);
    var m: fw.Modules = undefined;
    m.tiled = false;
    m.packs = null;
    m.index = null;
    m.dscore = null;
    m.mla = null;
    m.tfq3 = null;
    m.dsel = false;
    inline for (@typeInfo(glue.Kernels).@"struct".field_names, 1..) |name, i| {
        const T = @FieldType(glue.Kernels, name);
        if (T == cuda.Function or T == ?cuda.Function) @field(m.glue, name) = .{ .d = &d, .handle = @ptrFromInt(i) } else if (T == bool) @field(m.glue, name) = false;
    }
    var layer: weights.Layer = undefined;
    layer.ffn_hc = .{ .fn_w = 1 << 20, .base = 2 << 20, .scale = 3 << 20 };
    layer.post_norm = .{ .bf16 = 4 << 20 };
    var w: weights.Weights = undefined;
    w.config = .{};
    w.share.world = 2;
    w.share.rank = 1;
    var layers = [1]weights.Layer{layer};
    w.layers = &layers;
    var b: fw.Buffers = undefined;
    b.prefill = true;
    b.x = 5 << 20;
    b.post = 6 << 20;
    b.comb = 7 << 20;
    b.normed = 8 << 20;
    b.xs = 9 << 20;
    b.hcpart = 10 << 20;
    const one = [1]fw.Rank{.{ .w = &w, .st = undefined, .b = &b }};
    const f: fw.Forward = .{ .d = &d, .m = &m, .stream = .{ .d = &d, .handle = null }, .ranks = &one, .exchange = .local, .hc_fuse = true };
    Rec.n = 0;
    try f.glueRows(one[0], 0, .mixer, 0, 64, 11 << 20, 0);
    try std.testing.expectEqual(@as(usize, 2), Rec.n);
    try std.testing.expectEqual(@intFromPtr(m.glue.hc_post_partial.handle), Rec.ids[0]);
    try std.testing.expectEqual(@intFromPtr(m.glue.hc_finish.handle), Rec.ids[1]);
    Rec.n = 0;
    try f.glueRows(one[0], 0, .mixer, 0, 64, 11 << 20, 12 << 20);
    try std.testing.expectEqual(@as(usize, 2), Rec.n);
    try std.testing.expectEqual(@intFromPtr(m.glue.hc_post_partial_own.handle), Rec.ids[0]);
    try std.testing.expectEqual(@intFromPtr(m.glue.hc_finish.handle), Rec.ids[1]);
}
