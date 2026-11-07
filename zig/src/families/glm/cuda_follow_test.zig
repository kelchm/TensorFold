const std = @import("std");
const cuda = @import("cuda");
const images_mod = @import("cuda_images.zig");
const follow = @import("cuda_follow.zig");
const Header = follow.Header;
const Kind = follow.Kind;
const Link = follow.Link;
const spanWords = follow.spanWords;
const promptRows = follow.promptRows;
const structuredStep = follow.structuredStep;
const indexSplit = follow.indexSplit;
test "a step's header is 64 bytes and keeps its fields" {
    const h: Header = .{ .kind = @backingInt(Kind.draft), .n = 3, .depth = 2, .flags = 1, .a = 7, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 20 };
    const bytes = std.mem.asBytes(&h);
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(h, back);
    try std.testing.expectEqual(Kind.draft, @as(Kind, @fromBackingInt(@intCast(back.kind))));
}

test "128 video pairs cross the span serializer and a follower's receiver; one word short is refused" {
    const a = std.testing.allocator;
    var spans: [128]images_mod.Span = undefined;
    for (&spans, 0..) |*sp, i| sp.* = .{ .at = @intCast(i * 2), .n = 1, .row = @intCast(i) };
    const words = try spanWords(a, &spans, 3 * spans.len);
    defer a.free(words);
    var v: images_mod.Images = undefined;
    v.gpa = a;
    v.spans = .empty;
    v.slot = 9;
    defer v.spans.deinit(a);
    v.limits = .{ .image_tokens = 64, .max_images = 8, .video_tokens = 128, .max_frames = 256 };
    v.video_ids = .{ 1, 2, 3 };
    try v.take(9, words);
    try std.testing.expectEqualSlices(images_mod.Span, &spans, v.spans.items);
    try std.testing.expectEqual(@as(u32, 9), v.slot);
    try std.testing.expectEqual(@as(usize, 128), v.rowCount());
    try std.testing.expectError(error.TooManyIds, spanWords(a, &spans, 3 * spans.len - 1));
}

test "a link whose host allocation fails frees its device buffer" {
    const Mock = struct {
        var frees: usize = 0;
        fn alloc(ptr: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            ptr.* = 4096;
            return 0;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            frees += 1;
            return 0;
        }
    };
    Mock.frees = 0;
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemFree_v2 = Mock.free;
    var lib: cuda.nccl.Library = undefined;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const before = cuda.usage(false).device;
    defer {
        if (Mock.frees == 0) {
            var leaked: cuda.DeviceBuffer = .{ .d = &d, .ptr = 4096, .len = 64 + 128 * 4 };
            leaked.free();
        }
    }
    try std.testing.expectError(error.OutOfMemory, Link.init(failing.allocator(), &d, &lib, null, .{ .d = &d, .handle = null }, 128));
    try std.testing.expectEqual(@as(usize, 1), Mock.frees);
    try std.testing.expectEqual(before, cuda.usage(false).device);
}
test "flag 1 asks for prompt rows on prefill and chunk steps only (begin's is sampling, draft's from the prompt)" {
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 2048, .top_k = 4000 }));
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.prefill), .flags = 1 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.chunk), .depth = 2048, .top_k = 4000 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.begin), .flags = 1, .seed = 7 }));
    try std.testing.expect(!promptRows(.{ .kind = @backingInt(Kind.draft), .flags = 1, .depth = 3 }));
}

test "flag 1 on verify and shared steps draws from whole rows; a prompt step's flag 1 does not" {
    try std.testing.expect(structuredStep(.{ .kind = @backingInt(Kind.verify), .flags = 1, .n = 4 }));
    try std.testing.expect(structuredStep(.{ .kind = @backingInt(Kind.shared), .flags = 1, .depth = 2 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.verify), .n = 4 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 2048 }));
    try std.testing.expect(!structuredStep(.{ .kind = @backingInt(Kind.begin), .flags = 1 }));
}

test "flag 2 on a chunk splits its indexer rows across the ranks, beside flag 1's prompt rows; elsewhere it does not" {
    try std.testing.expect(indexSplit(.{ .kind = @backingInt(Kind.chunk), .flags = 2, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(indexSplit(.{ .kind = @backingInt(Kind.chunk), .flags = 3, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(promptRows(.{ .kind = @backingInt(Kind.chunk), .flags = 3, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(!indexSplit(.{ .kind = @backingInt(Kind.chunk), .flags = 1, .depth = 65536, .top_k = 70000 }));
    try std.testing.expect(!indexSplit(.{ .kind = @backingInt(Kind.prefill), .flags = 2 }));
}
