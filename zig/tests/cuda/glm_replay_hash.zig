//! Canonical ASCII JSON hashes shared with the external fixture adapter.
const std = @import("std");
pub const Value = std.json.Value;

pub fn digest(a: std.mem.Allocator, v: Value) ![64]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try canonical(a, &out.writer, v);
    return bytes(out.written());
}

pub fn bytes(raw: []const u8) [64]u8 {
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &hash, .{});
    return std.fmt.bytesToHex(hash, .lower);
}

pub fn matches(expected: []const u8, actual: [64]u8) !void {
    if (expected.len != 64 or !std.mem.eql(u8, expected, &actual)) return error.HashMismatch;
}

pub fn canonical(a: std.mem.Allocator, w: *std.Io.Writer, v: Value) anyerror!void {
    switch (v) {
        .null => try w.writeAll("null"),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .integer => |i| try w.print("{d}", .{i}),
        .string => |s| try string(w, s),
        .array => |arr| {
            try w.writeByte('[');
            for (arr.items, 0..) |item, i| {
                if (i != 0) try w.writeByte(',');
                try canonical(a, w, item);
            }
            try w.writeByte(']');
        },
        .object => |obj| {
            const keys = try a.dupe([]const u8, obj.keys());
            defer a.free(keys);
            std.mem.sort([]const u8, keys, {}, less);
            try w.writeByte('{');
            for (keys, 0..) |key, i| {
                for (key) |c| if (c > 127) return error.NonAsciiKey;
                if (i != 0) try w.writeByte(',');
                try string(w, key);
                try w.writeByte(':');
                try canonical(a, w, obj.get(key).?);
            }
            try w.writeByte('}');
        },
        else => return error.NonIntegerJson,
    }
}

fn less(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

fn string(w: *std.Io.Writer, raw: []const u8) !void {
    try w.writeByte('"');
    var it = (try std.unicode.Utf8View.init(raw)).iterator();
    while (it.nextCodepoint()) |cp| switch (cp) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        '\n' => try w.writeAll("\\n"),
        '\r' => try w.writeAll("\\r"),
        '\t' => try w.writeAll("\\t"),
        8 => try w.writeAll("\\b"),
        12 => try w.writeAll("\\f"),
        0...7, 11, 14...31, 127...0xffff => try w.print("\\u{x:0>4}", .{cp}),
        0x10000...0x10ffff => {
            const u = cp - 0x10000;
            try w.print("\\u{x:0>4}\\u{x:0>4}", .{ 0xd800 + (u >> 10), 0xdc00 + (u & 1023) });
        },
        else => try w.writeByte(@intCast(cp)),
    };
    try w.writeByte('"');
}

test "replay canonical JSON equals Python ASCII encoding" {
    const a = std.testing.allocator;
    const p = try std.json.parseFromSlice(Value, a, "{\"z\":true,\"a\":\"é😀\\n\",\"n\":[1,null]}", .{});
    defer p.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try canonical(a, &out.writer, p.value);
    try std.testing.expectEqualStrings("{\"a\":\"\\u00e9\\ud83d\\ude00\\n\",\"n\":[1,null],\"z\":true}", out.written());
}
