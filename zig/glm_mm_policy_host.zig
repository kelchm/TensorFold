const std = @import("std");
const cuda = @import("cuda");
const mm = @import("mm");
const kernels: mm.Kernels = .{
    .by_rows = .{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } },
    .packed_rows = @splat(.{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } }),
    .packed_prompt = .{ .{ .id = 2048 }, .{ .id = 2049 }, .{ .id = 2050 } },
    .prompt = .{ .id = 512 },
    .reduce = .{ .id = 1024 },
    .tile = .{},
    .tile_packed = @splat(.{}),
    .pdl = false,
};
test "verify and draft windows keep shape-selected split K at every width" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 31, 32 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expectEqual(@as(u32, 4), cuda.configs[0].grid.z);
        try std.testing.expect(cuda.symbols[0] < 2048);
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
    };
}
test "prompt flag chooses one K slice even for one row, alone or many" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 3, 16, 64, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
        const wi: usize = switch (bits) {
            4 => 0,
            6 => 1,
            8 => 2,
            else => 0,
        };
        try std.testing.expectEqual(if (bits == 16) @as(u32, 512) else kernels.packed_prompt[wi].id, cuda.symbols[0]);
        try std.testing.expectEqual(mm.sharedBytesStages(bits, 128, if (bits == 16) 3 else 2), cuda.configs[0].shared);
    };
}

test "fixed K prompt experiment uses the same shape split across bucket, offset and tail boundaries" {
    var k = kernels;
    k.prompt_fixed_k = true;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096 + 7 * 2048 * 2, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 2 * ((rows + 63) / 64)), cuda.count);
        var i: usize = 0;
        while (i < cuda.count) : (i += 2) {
            try std.testing.expectEqual(@as(u32, 4), cuda.configs[i].grid.z);
            try std.testing.expect(cuda.symbols[i] < 2048);
            try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[i + 1]);
        }
    };
    try std.testing.expect(!kernels.prompt_fixed_k);
}

test "TF_GLM_PDL_CHAIN: a K-sliced call's reduce launches as the matmul's dependent only with chain.reduce" {
    for ([_]bool{ false, true }) |on| for ([_]usize{ 1, 8, 32 }) |rows| {
        var k = kernels;
        k.pdl = true;
        k.chain.reduce = on;
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = 8 }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expect(cuda.configs[0].pdl); // the matmul's own, as before
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
        try std.testing.expectEqual(on, cuda.configs[1].pdl);
    };
    try std.testing.expectEqual(mm.Chain.all, mm.Chain.parse("1").?);
    try std.testing.expectEqual(mm.Chain{}, mm.Chain.parse("0").?);
    try std.testing.expectEqual(mm.Chain{ .reduce = true, .moe = true }, mm.Chain.parse("reduce,moe").?);
    try std.testing.expectEqual(mm.Chain{ .hc = true }, mm.Chain.parse("hc").?);
    try std.testing.expectEqual(mm.Chain.all, mm.Chain.parse("reduce,hc,moe").?);
    for ([_][]const u8{ "reduce,hc,x", "moe,hc", "p2p", "", "2" }) |bad| try std.testing.expect(mm.Chain.parse(bad) == null);
}
